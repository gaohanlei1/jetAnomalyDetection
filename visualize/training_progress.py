"""Shared training plot, extracted unchanged from run_train_lejepa_part."""
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_progress(
    self,
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    epoch_end_steps: List[int],
    best_val_loss: float,
    auc_history: Dict[str, Dict[str, Dict[str, List[float]]]],
    roc_eval_steps: List[int],
    suptitle: str="LeJEPA + Triplet + Semi-Supervised Classification Training Progress",
) -> None:
    """Plot loss curves and one pairwise ROC-AUC panel per signal type."""

    if len(train_history["total_loss"]) == 0:
        return

    title_map = {
        "total_loss": "Total Loss",
        "invariant_loss": "Invariant Loss",
        "sigreg_loss": "SIGReg Loss",
        "classification_loss": "Classification Loss",
        "triplet_loss": "Triplet Loss",
        "triplet_pos_distance": "Triplet Positive Distance",
        "triplet_neg_distance": "Triplet Negative Distance",
    }
    loss_keys = list(self.ssl_metric_keys)
    signal_panels = [
        signal
        for signal in self.signal_labels
        if signal in auc_history
    ]
    num_subplots = len(loss_keys) + len(signal_panels)

    fig, axes = plt.subplots(
        num_subplots,
        1,
        figsize=(10, 3.8 * num_subplots),
        sharex=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)

    step_axis = np.arange(1, len(train_history["total_loss"]) + 1)
    epoch_end_steps_np = np.asarray(epoch_end_steps)

    for ax, key in zip(axes[: len(loss_keys)], loss_keys):
        train_values = np.asarray(train_history[key], dtype=np.float64)
        val_values = np.asarray(val_history[key], dtype=np.float64)

        ax.plot(step_axis, train_values, label="Train", alpha=0.75)
        if len(val_values) > 0:
            repeat_count = int(np.ceil(len(train_values) / len(val_values)))
            repeated_val = np.repeat(val_values, repeat_count)[: len(train_values)]
            ax.plot(step_axis, repeated_val, label="Validation", alpha=0.75)

        if key == "total_loss" and np.isfinite(best_val_loss):
            ax.axhline(
                y=best_val_loss,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.25,
                label=f"Best Val: {best_val_loss:.4g}",
            )

        if len(epoch_end_steps_np) > 0:
            stride = max(1, int(np.ceil(len(epoch_end_steps_np) / 12)))
            for step_idx in epoch_end_steps_np[::stride]:
                ax.axvline(step_idx, color="gray", ls="--", lw=0.6, alpha=0.25)

        ax.set_ylabel(title_map[key])
        if np.all(train_values > 0):
            ax.set_yscale("log")
        ax.legend()
        ax.grid(False)

    eval_steps = np.asarray(roc_eval_steps, dtype=np.int64)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for panel_index, signal_label in enumerate(signal_panels):
        ax = axes[len(loss_keys) + panel_index]
        signal_name = signal_label.removeprefix("label_")
        pair_history = auc_history[signal_label]

        for background_index, background_label in enumerate(self.background_labels):
            # Background classes are shown against the configured test signal.
            #     continue
            if background_label not in pair_history:
                continue
            background_name = background_label.removeprefix("label_")
            pair_values = pair_history[background_label]
            train_heldout_auc = np.asarray(
                pair_values.get("train_heldout", []), dtype=np.float64
            )
            val_auc = np.asarray(pair_values.get("val", []), dtype=np.float64)
            color = (
                default_colors[background_index % len(default_colors)]
                if default_colors
                else None
            )

            if len(train_heldout_auc) > 0:
                train_heldout_last = float(train_heldout_auc[-1])
                train_heldout_steps = np.concatenate(
                    ([0], eval_steps[: len(train_heldout_auc)])
                )  # Show the first epoch value over the 0-1 epoch interval.
                train_heldout_auc = np.concatenate(
                    ([train_heldout_auc[0]], train_heldout_auc)
                )
                ax.step(
                    train_heldout_steps,
                    train_heldout_auc,
                    where="pre",
                    linestyle="-",
                    color=color,
                    alpha=0.85,
                    label=(
                        f"{background_name} Held-out train vs {signal_name} "
                        f"(Last: {train_heldout_last:.4f})"
                    ),
                )
            if len(val_auc) > 0:
                val_last = float(val_auc[-1])
                val_steps = np.concatenate(([0], eval_steps[: len(val_auc)]))
                val_auc = np.concatenate(([val_auc[0]], val_auc))
                ax.step(
                    val_steps,
                    val_auc,
                    where="pre",
                    linestyle="--",
                    color=color,
                    alpha=0.85,
                    label=(
                        f"{background_name} Val vs {signal_name} "
                        f"(Last: {val_last:.4f})"
                    ),
                )

        ax.axhline(
            0.5,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.45,
        )
        if len(epoch_end_steps_np) > 0:
            stride = max(1, int(np.ceil(len(epoch_end_steps_np) / 12)))
            for step_idx in epoch_end_steps_np[::stride]:
                ax.axvline(step_idx, color="gray", ls="--", lw=0.6, alpha=0.25)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"Pairwise ROC AUC with {signal_name} as visualization signal")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, ncol=2)
        ax.grid(False)

    axes[-1].set_xlabel("Step Number")

    if len(epoch_end_steps_np) > 0:
        epoch_ids = np.arange(1, len(epoch_end_steps_np) + 1)
        stride = max(1, int(np.ceil(len(epoch_end_steps_np) / 12)))
        top_ax = axes[0].secondary_xaxis("top")
        top_ax.set_xticks(epoch_end_steps_np[::stride])
        top_ax.set_xticklabels(epoch_ids[::stride])
        top_ax.set_xlabel("Epoch")

    fig.suptitle(
        suptitle,
        fontsize=15
    )
    fig.savefig(
        os.path.join(self.output_dir, "loss.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

