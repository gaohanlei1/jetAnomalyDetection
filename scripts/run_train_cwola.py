"""Train CWoLa on runtime representations from a frozen LeJEPA backbone.

python scripts/run_train_cwola.py plots/backbone --output-dir plots/cwola --signal-fraction 0.5
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from tqdm import tqdm

from models.cwola import CWoLaMLP
from scripts.lejepa_run import LeJEPARun, add_backbone_arguments
from scripts.run_train_lejepa_part import make_warmup_cosine_scheduler
from scripts.cwola_utils import (
    POOLED_BACKGROUND, POOLED_SIGNAL, resolve_label_sets, parse_label_override,
    configure_cwola_labels, batch_sizes,
    make_loaders, combine_batches, paired_batches, evaluate, encode_finite, write_json,
)
from visualize.training_progress import plot_progress


def resolve_training_config(args, summary):
    defaults = dict(epochs=20, steps_per_epoch=1000, val_steps=100, eval_steps=100,
                    learning_rate=1e-3, weight_decay=0.05, final_lr_ratio=1e-3)
    config = {key: getattr(args, key) if getattr(args, key) is not None else summary.get(key, value)
              for key, value in defaults.items()}
    config["batch_size"] = args.batch_size if args.batch_size is not None else int(
        summary.get("global_batch_size", summary.get("batch_size", 128)))
    total = config["epochs"] * config["steps_per_epoch"]
    config["warmup_steps"] = (args.warmup_steps if args.warmup_steps is not None else
                              summary.get("warmup_steps"))
    if config["warmup_steps"] is None:
        config["warmup_steps"] = int(summary.get("warmup_epochs", 10)) * config["steps_per_epoch"]
    if any(config[key] < 1 for key in ("epochs", "steps_per_epoch", "val_steps", "eval_steps")):
        raise ValueError("epochs, steps-per-epoch, val-steps and eval-steps must be positive.")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight-decay must be finite and nonnegative.")
    if not 0 <= config["final_lr_ratio"] <= 1:
        raise ValueError("final-lr-ratio must be in [0, 1].")
    if not 0 <= config["warmup_steps"] < total:
        raise ValueError("warmup-steps must be in [0, epochs * steps-per-epoch). "
                         "Override --warmup-steps when shortening a backbone schedule.")
    config["total_training_steps"] = total
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backbone_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True, help="New CWoLa run directory.")
    parser.add_argument("--signal-fraction", type=float, default=0.5,
                        help="Signal fraction WITHIN the mixture half, in [0, 1]. Default: 0.5.")
    parser.add_argument("--background-labels", type=parse_label_override, default=None,
                        help="Comma-separated CWoLa background labels; default: backbone summary.")
    parser.add_argument("--signal-labels", type=parse_label_override, default=None,
                        help="Comma-separated CWoLa signal labels; default: backbone summary.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-num-workers", type=int, default=None,
                        help="Workers per validation loader; default: min(training workers, 1).")
    parser.add_argument("--no-cache-validation", action="store_true",
                        help="Reload validation ROOT data each epoch instead of caching frozen CPU representations.")
    for name in ("epochs", "steps-per-epoch", "val-steps", "eval-steps", "warmup-steps"):
        parser.add_argument(f"--{name}", type=int, default=None, help="Default: backbone summary.json.")
    for name in ("learning-rate", "weight-decay", "final-lr-ratio"):
        parser.add_argument(f"--{name}", type=float, default=None, help="Default: backbone summary.json.")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("CWoLa uses one rank; launch with python, not multi-rank torchrun.")
    with (args.run_dir.expanduser() / "summary.json").open() as handle:
        source_summary = json.load(handle)
    try:
        backgrounds, signals = resolve_label_sets(source_summary, args.background_labels, args.signal_labels)
        config = resolve_training_config(args, source_summary)
        if args.eval_num_workers is not None and args.eval_num_workers < 0:
            raise ValueError("eval-num-workers must be nonnegative.")
        n_bg, n_sg = batch_sizes(config["batch_size"], args.signal_fraction)
        if not 0 <= args.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir.expanduser().resolve()
    if output == args.run_dir.expanduser().resolve() or (output.exists() and any(output.iterdir())):
        parser.error("--output-dir must be new or empty and differ from the backbone run.")
    args.batch_size = config["batch_size"]
    run = LeJEPARun(args)
    try:
        configure_cwola_labels(run, backgrounds, signals)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"CWoLa task: backgrounds={run.backgrounds}; signals={run.signals}", flush=True)
    model_config = dict(input_dim=run.model.config.representation_dim, dropout=args.dropout)
    model = CWoLaMLP(**model_config).to(run.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = make_warmup_cosine_scheduler(optimizer, config["total_training_steps"],
                                            config["warmup_steps"], config["final_lr_ratio"])
    effective_fraction = n_sg / (run.batch_size // 2)
    if effective_fraction != args.signal_fraction:
        warnings.warn(f"Integer batch allocation rounds signal fraction {args.signal_fraction} "
                      f"to {effective_fraction}; increase --batch-size for finer resolution.")
    if n_sg == 0:
        warnings.warn("Training mixture has no signal. ROC uses a separate diagnostic mixture "
                      "with one signal per batch; these signals do not enter training or validation loss.")
    print(f"CWoLa {model_config['input_dim']} -> 64 -> 32 -> 1; frozen backbone {run.checkpoint}")
    print(f"Batch: {n_bg} background + {n_sg} signal; first {run.batch_size // 2} rows "
          f"reference, last {run.batch_size // 2} rows mixture (effective fraction {effective_fraction:g}).")
    train_loaders = make_loaders(run, "train", args.signal_fraction,
                                 steps=config["steps_per_epoch"], training=True)
    eval_workers = min(run.num_workers, 1) if args.eval_num_workers is None else args.eval_num_workers
    val_loaders = make_loaders(run, "val", args.signal_fraction, steps=config["val_steps"],
                               num_workers=eval_workers)
    # Dedicated validation datasets for truth ROC, both with all configured types pooled.
    roc_loaders = make_loaders(run, "val", args.signal_fraction, steps=config["eval_steps"], for_roc=True,
                               num_workers=eval_workers)
    val_cache = None if args.no_cache_validation else []
    roc_cache = None if args.no_cache_validation else []
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "cwola-mlp", "status": "initialized", **config,
        "cwola_config": model_config, "signal_fraction": args.signal_fraction,
        "effective_signal_fraction": effective_fraction,
        "background_batch_size": n_bg, "signal_batch_size": n_sg,
        "roc_background_batch_size": batch_sizes(run.batch_size, args.signal_fraction, for_roc=True)[0],
        "roc_signal_batch_size": batch_sizes(run.batch_size, args.signal_fraction, for_roc=True)[1],
        "weak_label_definition": "0=background reference; 1=background+signal mixture",
        "batch_order": "reference_background, mixture_background, mixture_signal",
        "backbone_run_dir": str(run.run_dir), "backbone_checkpoint": str(run.checkpoint),
        "backbone_summary": source_summary, "backbone_frozen": True,
        "latent_space": "representation_head_of_cls", "backbone_precision": run.precision,
        "optimizer": "AdamW", "loss": "binary_cross_entropy_with_logits",
        "anomaly_score": "sigmoid", "score_direction": "higher_is_more_signal_like",
        "schedule": "linear_warmup_then_cosine_decay", "early_stopping": False,
        "dataset": run.backend.dataset_name, "dataset_root": str(run.backend.dataset_root),
        "dataset_label_axis": run.backend.label_axis,
        "background_labels": run.backgrounds, "signal_labels": run.signals,
        "background_labels_overridden": args.background_labels is not None,
        "signal_labels_overridden": args.signal_labels is not None,
        "particle_features": run.backend.feature_names,
        "batch_standardized_particle_features": run.backend.batch_standardized_feature_names,
        "max_num_particles": run.backend.max_num_particles,
        "max_train_events": run.summary.get("max_train_events"),
        "max_val_events": run.summary.get("max_val_events"),
        "max_test_background_events": run.summary.get("max_test_background_events"),
        "max_test_signal_events": run.summary.get("max_test_signal_events"),
        "num_workers": run.num_workers, "base_seed": run.seed,
        "prefetch_factor": run.summary.get("prefetch_factor", 2),
        "shuffle_active_shards": run.summary.get("shuffle_active_shards", 3),
        "cms_split_manifest_sha256": run.backend.cms_manifest_sha256,
        "validation_num_workers": eval_workers, "validation_prefetch_factor": 1,
        "validation_active_shards_per_label_per_worker": 1,
        "validation_representation_cache": not args.no_cache_validation,
        "validation_split": "val", "test_split": "test", "roc_truth": "background_vs_signal",
        "precision": "fp32", "device": str(run.device), "distributed": False, "world_size": 1,
        "global_batch_size": run.batch_size, "num_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_model_path": str(output / "best_model.pth"), "last_model_path": str(output / "last_model.pth"),
        "best_checkpoint_path": str(output / "best_checkpoint.pt"),
        "last_checkpoint_path": str(output / "last_checkpoint.pt"),
        "completed_epochs": 0, "best_val_loss": None,
    }
    if run.backend.cms_splits is not None:
        from datasets.cms_streaming import cms_split_manifest
        write_json(output / "cms_split_manifest.json", cms_split_manifest(run.backend.cms_splits))
    write_json(output / "summary.json", summary)
    train_history, val_history = {"total_loss": []}, {"total_loss": []}
    auc_history = {POOLED_SIGNAL: {POOLED_BACKGROUND: {"val": []}}}
    epoch_end_steps, roc_eval_steps = [], []
    plot_context = SimpleNamespace(ssl_metric_keys=["total_loss"], signal_labels=[POOLED_SIGNAL],
                                   background_labels=[POOLED_BACKGROUND], output_dir=str(output))
    best = math.inf
    train_iterator = paired_batches(train_loaders, config["total_training_steps"])
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for _ in tqdm(range(config["steps_per_epoch"]), desc=f"Epoch {epoch}/{config['epochs']}"):
            batch, weak, _ = combine_batches(*next(train_iterator))
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(encode_finite(run, batch))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, weak.to(run.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CWoLa training loss.")
            loss.backward()
            # Check gradients without imposing an extra clipping hyperparameter.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            train_history["total_loss"].append(loss.item())
        validation = evaluate(model, run, val_loaders, config["val_steps"], collect_scores=False,
                              description="Val background vs mixture loss", cache=val_cache)
        diagnostic = evaluate(model, run, roc_loaders, config["eval_steps"],
                              description="Val pooled background vs signal ROC", cache=roc_cache)
        val_loss = validation["loss"]
        val_history["total_loss"].append(val_loss)
        epoch_end_steps.append(len(train_history["total_loss"]))
        roc_eval_steps.append(epoch_end_steps[-1])
        auc_history[POOLED_SIGNAL][POOLED_BACKGROUND]["val"].append(diagnostic["auc"])
        improved = val_loss < best
        best = min(best, val_loss)
        summary.update(status="training", current_epoch=epoch, completed_epochs=epoch,
                       global_step=epoch_end_steps[-1], best_val_loss=best, final_val_loss=val_loss,
                       current_learning_rate=optimizer.param_groups[0]["lr"],
                       latest_train_losses={"total_loss": float(np.mean(train_history["total_loss"][-config["steps_per_epoch"]:]))},
                       latest_val_losses={"total_loss": val_loss}, latest_val_auc=diagnostic["auc"],
                       validation_loss_events=validation["num_events"],
                       validation_roc_background_events=int((diagnostic["truth"] == 0).sum()),
                       validation_roc_signal_events=int((diagnostic["truth"] == 1).sum()))
        history = dict(train_history=train_history, val_history=val_history, auc_history=auc_history,
                       epoch_end_steps=epoch_end_steps, roc_eval_steps=roc_eval_steps)
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          scheduler_state_dict=scheduler.state_dict(), epoch=epoch, metadata=dict(summary), **history)
        for prefix in (["last", "best"] if improved else ["last"]):
            torch.save(model.state_dict(), output / f"{prefix}_model.pth")
            torch.save(checkpoint, output / f"{prefix}_checkpoint.pt")
        write_json(output / "history.json", history)
        write_json(output / "summary.json", summary)
        plot_progress(plot_context, train_history, val_history, epoch_end_steps, best, auc_history,
                      roc_eval_steps, suptitle="CWoLa: weak-label loss and pooled signal/background ROC")
        print(f"Epoch {epoch}: val BCE={val_loss:.6g}; truth ROC AUC={diagnostic['auc']:.6f}", flush=True)
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
