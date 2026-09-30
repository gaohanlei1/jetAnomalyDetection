"""Train a single-rank affine flow on runtime projected CLS states of a frozen LeJEPA run.

python scripts/run_train_representation_flow.py plots/backbone --output-dir plots/flow
NLL (nats/dimension, higher = more anomalous) is the fixed anomaly score.
"""
import argparse
from itertools import count
import json
import math
import os
from pathlib import Path
import sys
import warnings
from types import SimpleNamespace
from time import perf_counter

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


def encode_finite(run, batch):
    """Keep the backbone frozen; retry reduced-precision failures in FP32."""
    if batch["padding_mask"].all(dim=-1).any():
        raise FloatingPointError("Input contains an entirely padded jet (no valid particles).")
    representations = run.encode(batch)
    if torch.isfinite(representations).all():
        return representations
    if not torch.isfinite(batch["x_particles"]).all():
        raise FloatingPointError("Input particle features contain NaN/Inf.")
    if run.device.type == "cuda" and run.precision != "fp32":
        representations = run.encode(batch, precision="fp32")
        if torch.isfinite(representations).all():
            warnings.warn("Non-finite backbone output in reduced precision; this batch was recomputed in FP32.")
            return representations
    raise FloatingPointError("Frozen backbone representations contain NaN/Inf, before the flow.")


def checked_optimizer_step(loss, flow, optimizer, max_norm):
    """A finite scalar loss does not guarantee finite backward gradients."""
    loss.backward()
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            flow.parameters(), max_norm=max_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise FloatingPointError("Non-finite flow gradient norm; optimizer step was NOT executed.") from exc
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in flow.parameters()]).all():
        raise FloatingPointError("Optimizer produced non-finite flow parameters.")
    return float(grad_norm)


def tensor_statistics(tensor):
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    return {"shape": list(tensor.shape), "nonfinite": int((~finite).sum()),
            "finite_abs_max": float(tensor[finite].abs().max()) if finite.any() else None}


@torch.no_grad()
def save_numerical_failure(path, flow, optimizer, batch, representations, *, epoch, step, error):
    """Save a reproducible failing batch and flow, never the backbone weights."""
    statistics = {"input": tensor_statistics(batch["x_particles"])}
    if representations is not None:
        statistics["representation"] = tensor_statistics(representations)
        x = representations
        statistics["couplings"] = []
        for index, layer in enumerate(flow.layers):
            x, logits = layer._transform(x)
            statistics["couplings"].append({"layer": index,
                "scale_logits": tensor_statistics(logits), "output": tensor_statistics(x)})
    torch.save({"epoch": epoch, "step": step, "error": str(error),
                "statistics": statistics,
                "batch": {key: value.detach().cpu() for key, value in batch.items()},
                "representations": None if representations is None else representations.detach().cpu(),
                "model_state_dict": flow.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, path)
    return statistics


def make_eval_loader(run, labels, steps, *, seed_offset=0, signal=False):
    """Match the main trainer: mixed-label active pool and fresh eval workers.

    Background streams repeat for bounded evaluation. Like the main trainer,
    JetClass signals are finite, while CMS signals repeat. The steps=0
    extension visits any split once, including its final partial batch.
    """
    return run.loader(
        "val", labels,
        infinite=steps > 0 and (not signal or run.backend.dataset_name == "cms"),
        seed_offset=seed_offset,
        max_events=run.summary.get("max_test_signal_events" if signal else "max_val_events")
                   if steps > 0 else None,
        drop_last=steps > 0, persistent_workers=False,
        prefetch_factor=int(run.summary.get("prefetch_factor", 2)),
        active_shards=int(run.summary.get("shuffle_active_shards", 3)),
    )


@torch.no_grad()
def evaluate(flow, run, loader, steps, *, label="validation"):
    """Forward mixed batches first; retain labels in exactly the score order."""
    scores, label_ids = [], []
    started = perf_counter()
    budget = f"{steps} mixed batches" if steps else "complete finite split"
    print(f"Validation {label}: {budget}; starting streaming loader "
          f"({loader.num_workers} workers).", flush=True)
    iterator = iter(loader)
    with tqdm(range(steps) if steps else count(), total=steps or None,
              desc=f"Val {label}", unit="batch") as progress:
        for _ in progress:
            try:
                batch = next(iterator)
            except StopIteration as exc:
                if steps:
                    raise RuntimeError(
                        f"{label} stream ended before the {steps}-batch budget. "
                        "Check max_val_events and dataset availability."
                    ) from exc
                break
            nll = flow.forward_pretrain(encode_finite(run, batch))["nll"]
            if not torch.isfinite(nll).all():
                raise FloatingPointError("Non-finite validation NLL.")
            scores.append(nll.cpu().numpy())
            label_ids.append(batch["y"].argmax(dim=-1).cpu().numpy())
    if not scores:
        raise RuntimeError("Validation stream has no events.")
    result = np.concatenate(scores)
    print(f"Validation {label}: {len(result):,} events in "
          f"{perf_counter() - started:.1f}s.", flush=True)
    return result, np.concatenate(label_ids)


def group_scores(scores, label_ids, labels, label_axis):
    """Split only after full mixed-batch backbone/flow inference."""
    grouped = {}
    for label in labels:
        selected = scores[label_ids == label_axis.index(label)]
        if len(selected) == 0:
            raise RuntimeError(f"No {label} events in the ROC sample; increase --eval-steps.")
        grouped[label] = selected
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backbone_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=8, help="Number of affine couplings (>=2).")
    parser.add_argument("--hidden-dim", type=int, default=128, help="SwiGLU hidden width.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Default: backbone run value.")
    parser.add_argument("--val-steps", type=int, default=None,
                        help="Mixed-background batches for validation loss; defaults to backbone val_steps. 0: full split.")
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="ROC batches per mixed background/signal loader; defaults to backbone eval_steps. 0: full split.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="Clip flow gradient norm before AdamW; default 1.0.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Default: 10%% of training steps.")
    parser.add_argument("--final-lr-ratio", type=float, default=1e-3)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This script uses one rank; launch with python, not multi-rank torchrun.")
    if args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("epochs and learning-rate must be positive; weight-decay must be nonnegative.")
    if not math.isfinite(args.grad_clip_norm) or args.grad_clip_norm <= 0:
        parser.error("grad-clip-norm must be finite and positive.")
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
    eval_steps = args.eval_steps if args.eval_steps is not None else int(run.summary.get("eval_steps", 100))
    total_steps = steps * args.epochs
    warmup = args.warmup_steps if args.warmup_steps is not None else total_steps // 10
    if steps < 1 or val_steps < 0 or eval_steps < 0 or not 0 <= warmup < total_steps:
        parser.error("steps-per-epoch must be positive, val-steps and eval-steps nonnegative, and warmup-steps in [0, total steps).")
    flow_config = dict(dim=run.model.config.representation_dim, num_layers=args.num_layers,
                       hidden_dim=args.hidden_dim, seed=run.seed)
    flow = RepresentationFlow(**flow_config).to(run.device)
    print(f"Number of params: {sum(p.numel() for p in flow.parameters())}, trainable: {sum(p.numel() for p in flow.parameters() if p.requires_grad)}")
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_steps, warmup, args.final_lr_ratio)
    train_loader = run.loader("train", run.backgrounds, infinite=True,
                              max_events=run.summary.get("max_train_events"), drop_last=True)
    # Main trainer convention: backgrounds share batches; signals have their
    # own mixed loader. Keep the user-requested VAL split for signal events.
    val_seed_offset = 202 if run.backend.dataset_name == "cms" else 0
    signal_seed_offset = 404 if run.backend.dataset_name == "cms" else 0
    bg_val_loader = make_eval_loader(run, run.backgrounds, val_steps, seed_offset=val_seed_offset)
    bg_roc_loader = make_eval_loader(run, run.backgrounds, eval_steps, seed_offset=val_seed_offset)
    signal_loader = (make_eval_loader(run, run.signals, eval_steps, seed_offset=signal_seed_offset, signal=True)
                     if run.signals else None)
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
        "validation_split": "val", "signal_eval_split": "val",
        "val_steps": val_steps, "eval_steps": eval_steps,
        "validation_batching": "mixed_background_labels; separate_mixed_signal_loader",
        "roc_grouping": "group_by_label_after_forward",
        "max_val_events": run.summary.get("max_val_events"),
        "max_signal_eval_events": run.summary.get("max_test_signal_events"),
        "batch_size": run.batch_size, "num_workers": run.num_workers,
        "prefetch_factor": run.summary.get("prefetch_factor", 2),
        "validation_streaming": True,
        "shuffle_active_shards": run.summary.get("shuffle_active_shards", 3),
        "validation_prefetch_factor": run.summary.get("prefetch_factor", 2),
        "validation_persistent_workers": False, "worker_start_method": "spawn",
        "epochs": args.epochs, "steps_per_epoch": steps, "warmup_steps": warmup,
        "total_training_steps": total_steps, "base_seed": run.seed,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
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
        for step in progress:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as exc:
                    raise RuntimeError("Training stream has no events.") from exc
            # Recompute projected CLS dynamically; backbone stays eval/no_grad.
            cls = None
            optimizer.zero_grad(set_to_none=True)
            try:
                cls = encode_finite(run, batch)
                output_dict = flow.forward_pretrain(cls)
                loss = output_dict["loss"]
                if not torch.isfinite(output_dict["nll"]).all() or not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite flow NLL with finite backbone representations.")
                grad_norm = checked_optimizer_step(loss, flow, optimizer, args.grad_clip_norm)
            except FloatingPointError as exc:
                diagnostic_path = output / "numerical_failure.pt"
                statistics = save_numerical_failure(
                    diagnostic_path, flow, optimizer, batch, cls,
                    epoch=epoch, step=step + 1, error=exc)
                summary.update(status="failed", current_epoch=epoch, failed_step=step + 1,
                               numerical_error=str(exc), numerical_statistics=statistics,
                               numerical_failure_path=str(diagnostic_path))
                write_json(output / "summary.json", summary)
                raise FloatingPointError(
                    f"Epoch {epoch}, step {step + 1}: {exc} "
                    f"Reproduction data saved to {diagnostic_path}."
                ) from exc
            scheduler.step()
            train_history["total_loss"].append(loss.item())
            progress.set_postfix(nll=f"{loss.item():.5g}", grad=f"{grad_norm:.3g}")
        flow.eval()
        print(f"Epoch {epoch}: training done; validating mixed backgrounds.", flush=True)
        # As in run_train_lejepa_part: val_steps controls the validation loss;
        # eval_steps separately controls representation collection for ROC.
        val_scores, val_ids = evaluate(flow, run, bg_val_loader, val_steps, label="background loss")
        val_loss = float(val_scores.mean())
        val_history["total_loss"].append(val_loss)
        epoch_end_steps.append(len(train_history["total_loss"]))
        by_label, background_scores = {}, {}
        if signal_loader is not None:
            bg_scores, bg_ids = evaluate(flow, run, bg_roc_loader, eval_steps, label="background ROC")
            sg_scores, sg_ids = evaluate(flow, run, signal_loader, eval_steps, label="signal ROC")
            by_label.update(group_scores(bg_scores, bg_ids, run.backgrounds, run.backend.label_axis))
            by_label.update(group_scores(sg_scores, sg_ids, run.signals, run.backend.label_axis))
            background_scores = {label: by_label[label] for label in run.backgrounds}
            background_scores[pooled_background_label] = np.concatenate(
                [background_scores[label] for label in run.backgrounds])
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
                       validation_loss_events=len(val_scores),
                       validation_events_by_label={label: int((val_ids == run.backend.label_axis.index(label)).sum())
                                                   for label in run.backgrounds},
                       roc_events_by_label={key: len(value) for key, value in by_label.items()})
        history = dict(train_history=train_history, val_history=val_history,
                       auc_history=auc_history, epoch_end_steps=epoch_end_steps,
                       roc_eval_steps=roc_eval_steps)
        checkpoint = dict(model_state_dict=flow.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          scheduler_state_dict=scheduler.state_dict(), epoch=epoch,
                          metadata=dict(summary), **history)
        print(f"Epoch {epoch}: saving flow checkpoints and history.", flush=True)
        # Flow parameters only: the frozen backbone is referenced, never copied.
        for prefix in (["last", "best"] if improved else ["last"]):
            torch.save(flow.state_dict(), output / f"{prefix}_model.pth")
            torch.save(checkpoint, output / f"{prefix}_checkpoint.pt")
        write_json(output / "history.json", history)
        write_json(output / "summary.json", summary)
        print(f"Epoch {epoch}: plotting progress.", flush=True)
        plot_progress(plot_context, train_history, val_history, epoch_end_steps,
                      best, auc_history, roc_eval_steps, suptitle="Representation Flow Training Progress")
        print(f"Epoch {epoch}: validation NLL={val_loss:.6f}, best={best:.6f}")
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)
    print(f"Saved flow run to {output}")


if __name__ == "__main__":
    main()
