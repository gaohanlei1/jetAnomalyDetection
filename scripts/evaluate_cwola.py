"""Evaluate a CWoLa run on test data, pooling backgrounds in every plot.

python scripts/evaluate_cwola.py plots/cwola
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from models.cwola import CWoLaMLP
from scripts.lejepa_run import LeJEPARun
from scripts.diagnose_lejepa_latents import read_state_dict
from scripts.cwola_utils import validate_label_sets, configure_cwola_labels, batch_sizes, make_loaders, evaluate, write_json


def plot_pair(output, name, signal_name, background_scores, signal_scores):
    truth = np.r_[np.zeros(len(background_scores)), np.ones(len(signal_scores))]
    scores = np.r_[background_scores, signal_scores]
    auc = float(roc_auc_score(truth, scores))
    fpr, tpr, _ = roc_curve(truth, scores)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="Background efficiency (FPR)", ylabel="Signal efficiency (TPR)",
           title=f"All backgrounds vs {signal_name}", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    roc_path = output / f"roc_{name}.png"
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    bins = np.linspace(0, 1, 51)
    ax.hist(background_scores, bins=bins, density=True, histtype="step", label="All backgrounds")
    ax.hist(signal_scores, bins=bins, density=True, histtype="step", label=signal_name)
    ax.set(xlabel="CWoLa score (sigmoid; higher = more signal-like)", ylabel="Density",
           title=f"All backgrounds vs {signal_name}", xlim=(0, 1))
    ax.legend()
    score_path = output / f"score_distribution_{name}.png"
    fig.savefig(score_path, dpi=160)
    plt.close(fig)
    return dict(auc=auc, background_events=len(background_scores), signal_events=len(signal_scores),
                roc_plot=str(roc_path), score_distribution_plot=str(score_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="CWoLa run containing summary.json.")
    parser.add_argument("--checkpoint", type=Path, help="CWoLa checkpoint; default: best_model.pth.")
    parser.add_argument("--dataset-root", type=Path, help="Optional relocation of the saved dataset root.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="Default: CWoLa summary eval_steps; 0: complete finite test splits.")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Evaluation uses one rank; launch with python.")
    output_run = args.run_dir.expanduser().resolve()
    with (output_run / "summary.json").open() as handle:
        summary = json.load(handle)
    if summary.get("model") != "cwola-mlp":
        parser.error("run_dir must be a CWoLa run.")
    try:
        backgrounds, signals = validate_label_sets(summary)
        bs = args.batch_size if args.batch_size is not None else int(summary["batch_size"])
        n_bg, n_sg = batch_sizes(bs, summary["signal_fraction"], for_roc=True)
    except ValueError as exc:
        parser.error(str(exc))
    steps = args.eval_steps if args.eval_steps is not None else int(summary["eval_steps"])
    if steps < 0:
        parser.error("eval-steps must be nonnegative.")
    run = LeJEPARun(SimpleNamespace(
        run_dir=Path(summary["backbone_run_dir"]), checkpoint=Path(summary["backbone_checkpoint"]),
        dataset_root=args.dataset_root or Path(summary["dataset_root"]), batch_size=bs,
        num_workers=args.num_workers if args.num_workers is not None else int(summary["num_workers"]),
        device=args.device), summary=summary["backbone_summary"])
    configure_cwola_labels(run, backgrounds, signals)
    if run.model.config.representation_dim != summary["cwola_config"]["input_dim"]:
        raise ValueError("Backbone representation dimension differs from the trained CWoLa input.")
    # Saved training settings, including event limits and precision, govern eval.
    run.summary = dict(summary["backbone_summary"])
    run.precision = summary["backbone_precision"]
    model = CWoLaMLP(**summary["cwola_config"]).to(run.device)
    checkpoint = (args.checkpoint or output_run / "best_model.pth").expanduser().resolve()
    model.load_state_dict(read_state_dict(checkpoint, run.device), strict=True)
    loaders = make_loaders(run, "test", summary["signal_fraction"], steps=steps, for_roc=True)
    result = evaluate(model, run, loaders, steps, per_signal=True, require_full_budget=False,
                      description="Test pooled background and signal")
    background_scores = result["scores"][result["truth"] == 0]
    # Group scores AFTER the same mixed forward pass; reuse the identical pooled
    # background sample for every signal-specific ROC and distribution.
    signal_scores = {}
    for signal in signals:
        signal_scores[signal] = result["scores"][result["signal_ids"] == run.backend.label_axis.index(signal)]
        if not len(signal_scores[signal]):
            raise RuntimeError(f"No {signal} events in the test sample. Increase --eval-steps or use 0.")
    output = output_run / "latent_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"pooled": plot_pair(output, "pooled", "All signals", background_scores,
                                   result["scores"][result["truth"] == 1])}
    for signal, scores in signal_scores.items():
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", signal)
        metrics[signal] = plot_pair(output, name, signal.removeprefix("label_"), background_scores, scores)
    write_json(output / "evaluation.json", {
        "checkpoint": str(checkpoint), "backbone_checkpoint": str(run.checkpoint),
        "split": "test", "dataset_root": str(run.backend.dataset_root),
        "background_labels": backgrounds, "signal_labels": signals,
        "requested_eval_steps": steps, "batch_size": bs,
        "signal_fraction": summary["signal_fraction"],
        "background_batch_size": n_bg, "signal_batch_size": n_sg,
        "roc_signal_fraction": n_sg / (bs // 2),
        "anomaly_score": "sigmoid", "roc_truth": "background_vs_signal", "metrics": metrics,
    })
    print(f"Pooled truth ROC AUC={result['auc']:.6f}; plots written to {output}")


if __name__ == "__main__":
    main()
