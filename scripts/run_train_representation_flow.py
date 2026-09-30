"""Train a single-rank affine flow on runtime projected CLS states of a frozen LeJEPA run.

python scripts/run_train_representation_flow.py plots/backbone --output-dir plots/flow
NLL (nats/dimension, higher = more anomalous) is the fixed anomaly score.
"""
import argparse
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.representation_flow import RepresentationFlow
from scripts.lejepa_run import LeJEPARun, add_backbone_arguments
from scripts.run_train_lejepa_part import make_warmup_cosine_scheduler
from visualize.training_progress import plot_progress


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    temporary.replace(path)


@torch.no_grad()
def evaluate(flow, run, loader, steps):
    scores = []
    batches = loader if steps == 0 else islice(loader, steps)
    for batch in batches:
        nll = flow.forward_pretrain(run.encode(batch))["nll"]
        if not torch.isfinite(nll).all():
            raise FloatingPointError("Non-finite validation NLL.")
        scores.append(nll.cpu().numpy())
    if not scores:
        raise RuntimeError("Validation stream has no events.")
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backbone_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=8, help="Number of affine couplings (>=2).")
    parser.add_argument("--hidden-dim", type=int, default=128, help="SwiGLU hidden width.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Default: backbone run value.")
    parser.add_argument("--val-steps", type=int, default=None,
                        help="Batches per validation jet type; 0 means the complete split.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=None, help="Default: 10%% of training steps.")
    parser.add_argument("--final-lr-ratio", type=float, default=1e-3)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This script uses one rank; launch with python, not multi-rank torchrun.")
    if args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("epochs and learning-rate must be positive; weight-decay must be nonnegative.")
    if not 0 <= args.final_lr_ratio <= 1:
        parser.error("final-lr-ratio must be in [0, 1].")
    output = args.output_dir.expanduser().resolve()
    if output == args.run_dir.expanduser().resolve():
        parser.error("--output-dir must differ from the backbone run.")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be new or empty to protect existing runs.")
    run = LeJEPARun(args)
    steps = args.steps_per_epoch if args.steps_per_epoch is not None else int(run.summary.get("steps_per_epoch", 1000))
    val_steps = args.val_steps if args.val_steps is not None else int(run.summary.get("val_steps", 100))
    total_steps = steps * args.epochs
    warmup = args.warmup_steps if args.warmup_steps is not None else total_steps // 10
    if steps < 1 or val_steps < 0 or not 0 <= warmup < total_steps:
        parser.error("steps-per-epoch must be positive, val-steps nonnegative, and warmup-steps in [0, total steps).")
    flow_config = dict(dim=run.model.config.representation_dim, num_layers=args.num_layers,
                       hidden_dim=args.hidden_dim, seed=run.seed)
    flow = RepresentationFlow(**flow_config).to(run.device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_steps, warmup, args.final_lr_ratio)
    train_loader = run.loader("train", run.backgrounds, infinite=True,
                              max_events=run.summary.get("max_train_events"))
    # Finite validation streams never duplicate samples to meet a step budget.
    val_loaders = {label: run.loader("val", [label]) for label in run.backgrounds + run.signals}
    output.mkdir(parents=True, exist_ok=True)
    train_history, val_history = {"total_loss": []}, {"total_loss": []}
    # A display-only group: never add it to the dataset's physical jet labels.
    pooled_background_label = "All backgrounds"
    auc_background_labels = [*run.backgrounds, pooled_background_label]
    auc_history = {signal: {bg: {"val": []} for bg in auc_background_labels} for signal in run.signals}
    epoch_end_steps, roc_eval_steps = [], []
    plot_context = SimpleNamespace(ssl_metric_keys=["total_loss"], signal_labels=run.signals,
                                   background_labels=auc_background_labels, output_dir=str(output))
    summary = {
        "model": "representation-affine-flow", "status": "initialized",
        "backbone_run_dir": str(run.run_dir), "backbone_checkpoint": str(run.checkpoint),
        "backbone_summary": str(run.run_dir / "summary.json"), "backbone_frozen": True,
        "latent_space": "representation_head_of_cls", "flow_config": flow_config,
        "scale_multiplier": 2, "initialization": "zero final linear weights and biases; identity flow",
        "loss": "negative_log_likelihood", "loss_units": "nats_per_dimension",
        "log_prob_definition": "-representation_dim * per_event_nll", "anomaly_score": "nll",
        "dataset": run.backend.dataset_name, "dataset_root": str(run.backend.dataset_root),
        "background_labels": run.backgrounds, "signal_labels": run.signals,
        "pooled_auc_background_labels": run.backgrounds,
        "pooled_auc_history_key": pooled_background_label,
        "particle_features": run.backend.feature_names,
        "batch_standardized_particle_features": run.backend.batch_standardized_feature_names,
        "max_num_particles": run.backend.max_num_particles,
        "max_train_events": run.summary.get("max_train_events"),
        "cms_split_manifest_sha256": run.backend.cms_manifest_sha256,
        "cms_split_manifest_source": run.backend.cms_manifest_source,
        "validation_split": "val", "val_steps_per_label": val_steps,
        "batch_size": run.batch_size, "num_workers": run.num_workers,
        "prefetch_factor": run.summary.get("prefetch_factor", 2),
        "epochs": args.epochs, "steps_per_epoch": steps, "warmup_steps": warmup,
        "total_training_steps": total_steps, "base_seed": run.seed,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "final_lr_ratio": args.final_lr_ratio, "precision": "fp32",
        "backbone_precision": run.precision, "device": str(run.device),
        "distributed": False, "world_size": 1,
        "num_trainable_parameters": sum(p.numel() for p in flow.parameters()),
        "best_model_path": str(output / "best_model.pth"),
        "last_model_path": str(output / "last_model.pth"),
        "best_checkpoint_path": str(output / "best_checkpoint.pt"),
        "last_checkpoint_path": str(output / "last_checkpoint.pt"),
        "completed_epochs": 0, "best_val_loss": None,
    }
    if run.backend.cms_splits is not None:
        from datasets.cms_streaming import cms_split_manifest
        write_json(output / "cms_split_manifest.json", cms_split_manifest(run.backend.cms_splits))
    write_json(output / "summary.json", summary)
    best = math.inf
    train_iterator = iter(train_loader)
    for epoch in range(1, args.epochs + 1):
        flow.train()
        progress = tqdm(range(steps), desc=f"Epoch {epoch}/{args.epochs}")
        for _ in progress:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as exc:
                    raise RuntimeError("Training stream has no events.") from exc
            # Recompute CLS dynamically; backbone remains eval + no_grad.
            cls = run.encode(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = flow.forward_pretrain(cls)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training NLL.")
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_history["total_loss"].append(loss.item())
            progress.set_postfix(nll=f"{loss.item():.5g}")
        flow.eval() # evaluate roc 
        by_label = {label: evaluate(flow, run, loader, val_steps)
                    for label, loader in val_loaders.items()}
        # Pool event scores, not per-type AUCs: preserve the sampled class counts.
        pooled_background = np.concatenate([by_label[label] for label in run.backgrounds])
        background_scores = {label: by_label[label] for label in run.backgrounds}
        background_scores[pooled_background_label] = pooled_background
        val_loss = float(pooled_background.mean())
        val_history["total_loss"].append(val_loss)
        epoch_end_steps.append(len(train_history["total_loss"]))
        roc_eval_steps.append(epoch_end_steps[-1])
        latest_auc = {}
        for signal in run.signals:
            latest_auc[signal] = {}
            for background, bg in background_scores.items():
                sg = by_label[signal]
                auc = float(roc_auc_score(np.r_[np.zeros(len(bg)), np.ones(len(sg))], np.r_[bg, sg]))
                auc_history[signal][background]["val"].append(auc)
                latest_auc[signal][background] = auc
                print(f"{background} vs {signal}: validation ROC AUC = {auc:.6f}")
        improved = val_loss < best
        best = min(best, val_loss)
        summary.update(status="training", current_epoch=epoch, completed_epochs=epoch,
                       best_val_loss=best, final_val_loss=val_loss,
                       current_learning_rate=optimizer.param_groups[0]["lr"],
                       latest_train_losses={"total_loss": float(np.mean(train_history["total_loss"][-steps:]))},
                       latest_val_losses={"total_loss": val_loss}, latest_val_auc=latest_auc,
                       validation_events_by_label={key: len(value) for key, value in by_label.items()})
        history = dict(train_history=train_history, val_history=val_history,
                       auc_history=auc_history, epoch_end_steps=epoch_end_steps,
                       roc_eval_steps=roc_eval_steps)
        checkpoint = dict(model_state_dict=flow.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          scheduler_state_dict=scheduler.state_dict(), epoch=epoch,
                          metadata=dict(summary), **history)
        # Flow parameters only: the frozen backbone is referenced, never copied.
        for prefix in (["last", "best"] if improved else ["last"]):
            torch.save(flow.state_dict(), output / f"{prefix}_model.pth")
            torch.save(checkpoint, output / f"{prefix}_checkpoint.pt")
        write_json(output / "history.json", history)
        write_json(output / "summary.json", summary)
        plot_progress(plot_context, train_history, val_history, epoch_end_steps,
                      best, auc_history, roc_eval_steps, suptitle="Representation Flow Training Progress")
        print(f"Epoch {epoch}: validation NLL={val_loss:.6f}, best={best:.6f}")
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)
    print(f"Saved flow run to {output}")


if __name__ == "__main__":
    main()
