#!/usr/bin/env python3
"""Plot random CMS or JetClass jets directly from ROOT shards."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IDENTITY_FEATURES = (
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)
IDENTITY_MARKERS = {
    "part_isChargedHadron": "o",
    "part_isNeutralHadron": "h",
    "part_isPhoton": "^",
    "part_isElectron": "s",
    "part_isMuon": "D",
    "unidentified": "X",
}
IDENTITY_LABELS = {
    "part_isChargedHadron": "charged hadron",
    "part_isNeutralHadron": "neutral hadron",
    "part_isPhoton": "photon",
    "part_isElectron": "electron",
    "part_isMuon": "muon",
    "unidentified": "other",
}
CHARGE_COLORS = {
    -1: "#4F78C8",
    0: "#8B8F97",
    1: "#D65A5A",
}


def linear_nonnegative_unit(
    values: np.ndarray,
    *,
    upper_quantile: float = 0.95,
    feature_name: str,
) -> Tuple[np.ndarray, float, float]:
    """Linearly map a nonnegative significance feature to [0, 1].

    The visualization uses the native significance values directly:

        low  = 0
        high = the requested upper quantile of the finite raw values

    Values above ``high`` are saturated at 1. No logarithm, asinh, signed
    transformation, centering, or other nonlinear mapping is applied.
    """
    raw = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(raw)
    finite = raw[finite_mask]
    if len(finite) == 0:
        raise ValueError(
            f"Cannot color by {feature_name}: no finite significance values."
        )

    negative = finite[finite < 0.0]
    if len(negative) > 0:
        raise ValueError(
            f"{feature_name} is expected to be nonnegative, but found "
            f"{len(negative)} negative values; minimum={float(negative.min()):.6g}."
        )

    low = 0.0
    high = float(np.quantile(finite, upper_quantile))
    if not np.isfinite(high):
        raise ValueError(
            f"Cannot color by {feature_name}: non-finite {upper_quantile:.3f} quantile."
        )

    if high <= low:
        unit = np.zeros_like(raw, dtype=np.float64)
        return unit, low, high

    unit = (raw - low) / (high - low)
    unit = np.clip(unit, 0.0, 1.0)
    unit[~finite_mask] = 0.0
    return unit, low, high


def bivariate_edge_colors(dxy_unit: np.ndarray, dz_unit: np.ndarray) -> np.ndarray:
    # Muted four-corner palette. Horizontal motion is controlled only by dxy;
    # vertical motion is controlled only by dz, with bilinear interpolation.
    c00 = np.asarray(to_rgb("#0D0D0D"))  # low dxy, low dz
    c10 = np.asarray(to_rgb("#3BF707"))  # high dxy, low dz
    c01 = np.asarray(to_rgb("#082DFD"))  # low dxy, high dz
    c11 = np.asarray(to_rgb("#F22704"))  # high dxy, high dz
    x = np.asarray(dxy_unit, dtype=np.float64)[..., None]
    y = np.asarray(dz_unit, dtype=np.float64)[..., None]
    return (
        (1.0 - x) * (1.0 - y) * c00
        + x * (1.0 - y) * c10
        + (1.0 - x) * y * c01
        + x * y * c11
    )


def plot_random_jet_events(
    *,
    events_path: Optional[Path] = None,
    feature_names: Sequence[str],
    output_path: Path,
    num_events: int,
    seed: int,
    scores_path: Optional[Path] = None,
    events: Optional[np.ndarray] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    event_labels: Optional[Sequence[str]] = None,
    title: str = "Random sample from highest-scoring held-out real-data jets",
) -> List[int]:
    if events is None:
        if events_path is None:
            raise ValueError("Provide events or events_path.")
        events = np.load(events_path, mmap_mode="r")
    feature_names = list(feature_names)
    dxy_name = "Cpfcan_dxysig" if "Cpfcan_dxysig" in feature_names else "d0_sig"
    dz_name = "Cpfcan_dzsig" if "Cpfcan_dzsig" in feature_names else "dz_sig"
    if event_labels is not None and len(event_labels) != len(events):
        raise ValueError("event_labels must have one entry per event.")
    if events.ndim != 3:
        raise ValueError(f"Expected events array shape (E, N, F), got {events.shape}.")
    if events.shape[2] != len(feature_names):
        raise ValueError(
            f"Events feature dimension {events.shape[2]} does not match feature list "
            f"length {len(feature_names)}."
        )
    required = {
        "part_deta",
        "part_dphi",
        "part_pt",
        "part_charge",
        dxy_name,
        dz_name,
        *IDENTITY_FEATURES,
    }
    missing = sorted(required - set(feature_names))
    if missing:
        raise ValueError(f"Cannot visualize jets; missing features: {missing}")
    if len(events) == 0:
        raise ValueError(f"Events file is empty: {events_path}")
    n = min(max(1, int(num_events)), len(events))
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(events), size=n, replace=False))
    selected = np.asarray(events[selected_indices], dtype=np.float32)
    scores = None
    if scores_path is not None:
        score_array = np.load(scores_path, mmap_mode="r")
        if len(score_array) != len(events):
            raise ValueError(
                f"Score array length {len(score_array)} does not match events {len(events)}."
            )
        scores = np.asarray(score_array[selected_indices], dtype=np.float64)

    index = {name: feature_names.index(name) for name in feature_names}
    valid_masks = ~np.all(selected == 0, axis=-1)
    all_valid = selected[valid_masks]
    if len(all_valid) == 0:
        raise ValueError("Selected jets contain no non-padding particle candidates.")

    all_dxy_unit, dxy_low, dxy_high = linear_nonnegative_unit(
        np.abs(all_valid[:, index[dxy_name]]) if dxy_name == "d0_sig" else all_valid[:, index[dxy_name]],
        upper_quantile=0.95,
        feature_name=dxy_name,
    )
    all_dz_unit, dz_low, dz_high = linear_nonnegative_unit(
        np.abs(all_valid[:, index[dz_name]]) if dz_name == "dz_sig" else all_valid[:, index[dz_name]],
        upper_quantile=0.95,
        feature_name=dz_name,
    )

    coord_abs = np.concatenate(
        [
            np.abs(all_valid[:, index["part_deta"]]),
            np.abs(all_valid[:, index["part_dphi"]]),
        ]
    )
    axis_limit = max(0.15, float(np.max(coord_abs)) * 1.08)

    if nrows is not None and nrows < 1 or ncols is not None and ncols < 1:
        raise ValueError("Subplot rows and columns must be positive.")
    if ncols is None:
        ncols = int(math.ceil(n / nrows)) if nrows else min(4, int(math.ceil(math.sqrt(n))))
    if nrows is None:
        nrows = int(math.ceil(n / ncols))
    if nrows * ncols < n:
        raise ValueError("Subplot grid is too small for num_events.")
    if event_labels is not None and "label_fQCD" in event_labels:
        from datasets.jetclass_streaming import (
            FQCD_RECT_TOP_LEFT, FQCD_RECT_ETA_LENGTH, FQCD_RECT_PHI_LENGTH,
        )
        eta_left, phi_top = FQCD_RECT_TOP_LEFT
        axis_limit = max(axis_limit, 1.08 * max(abs(eta_left),
            abs(eta_left + FQCD_RECT_ETA_LENGTH), abs(phi_top),
            abs(phi_top - FQCD_RECT_PHI_LENGTH)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols + 1.8, 3.35 * nrows + 1.1),
        squeeze=False,
    )

    global_offset = 0
    for panel_index, event_index in enumerate(selected_indices):
        ax = axes.flat[panel_index]
        event = selected[panel_index]
        valid = valid_masks[panel_index]
        particles = event[valid]
        count = len(particles)
        dxy_unit = all_dxy_unit[global_offset : global_offset + count]
        dz_unit = all_dz_unit[global_offset : global_offset + count]
        global_offset += count
        edge_colors = bivariate_edge_colors(dxy_unit, dz_unit)

        identities = particles[:, [index[name] for name in IDENTITY_FEATURES]]
        identity_argmax = identities.argmax(axis=1)
        identity_strength = identities.max(axis=1)
        identity_names = np.asarray(
            [IDENTITY_FEATURES[i] for i in identity_argmax], dtype=object
        )
        identity_names[identity_strength <= 0.5] = "unidentified"

        charge_values = particles[:, index["part_charge"]]
        charge_class = np.where(charge_values > 0.5, 1, np.where(charge_values < -0.5, -1, 0))
        face_colors = np.asarray([CHARGE_COLORS[int(c)] for c in charge_class])
        pt_frac = np.clip(particles[:, index["part_pt"]], 0.0, 1.0)
        sizes = 8 + 240.0 * pt_frac

        for identity_name, marker in IDENTITY_MARKERS.items():
            mask = identity_names == identity_name
            if not np.any(mask):
                continue
            ax.scatter(
                particles[mask, index["part_deta"]],
                particles[mask, index["part_dphi"]],
                s=sizes[mask],
                marker=marker,
                c=face_colors[mask],
                edgecolors=edge_colors[mask],
                linewidths=1.25,
                alpha=0.5,
            )

        panel_title = f"event {int(event_index)}"
        if event_labels is not None:
            label = event_labels[event_index]
            panel_title = f"{label} — {panel_title}"
            if label == "label_fQCD":
                ax.add_patch(Rectangle(
                    (eta_left, phi_top - FQCD_RECT_PHI_LENGTH),
                    FQCD_RECT_ETA_LENGTH, FQCD_RECT_PHI_LENGTH,
                    fill=False, edgecolor="#A43D38", linestyle="--", linewidth=1,
                ))
        if scores is not None:
            panel_title += f"\nscore {scores[panel_index]:.1f}"
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xticks([-axis_limit, 0, axis_limit], labels=[f"{-axis_limit:.2f}", "0.00", f"{axis_limit:.2f}"], fontsize=8)
        ax.set_yticks([-axis_limit, 0, axis_limit], labels=[f"{-axis_limit:.2f}", "0.00", f"{axis_limit:.2f}"], fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.axhline(0, color="#D7DADF", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#D7DADF", linewidth=0.6, zorder=0)
        ax.set_xlabel(r"$\Delta\eta$")
        ax.set_ylabel(r"$\Delta\phi$")
        ax.grid(False)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    identity_handles = [
        Line2D(
            [0],
            [0],
            marker=IDENTITY_MARKERS[name],
            color="none",
            markerfacecolor="#8B8F97",
            markeredgecolor="#59636C",
            markersize=7,
            label=IDENTITY_LABELS[name],
        )
        for name in IDENTITY_FEATURES
    ]
    charge_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CHARGE_COLORS[value],
            markeredgecolor="none",
            markersize=7,
            label=label,
        )
        for value, label in ((1, "charge +1"), (0, "charge 0"), (-1, "charge -1"))
    ]
    fig.legend(
        handles=identity_handles + charge_handles,
        loc="lower center",
        bbox_to_anchor=(0.43, 0.01),
        ncol=min(4, len(identity_handles) + len(charge_handles)),
        frameon=False,
        fontsize=8,
    )

    # A compact bivariate key for the outline color.
    key_ax = fig.add_axes([0.865, 0.06, 0.105, 0.105])
    grid = np.linspace(0.0, 1.0, 64)
    xx, yy = np.meshgrid(grid, grid)
    key_ax.imshow(
        bivariate_edge_colors(xx, yy),
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
    )
    key_ax.set_xlabel("|d0 sig|" if dxy_name == "d0_sig" else "dxy sig (raw)", fontsize=7, labelpad=1)
    key_ax.set_ylabel("|dz sig|" if dz_name == "dz_sig" else "dz sig (raw)", fontsize=7, labelpad=1)
    key_ax.set_xticks(
        [0, 1],
        [f"{dxy_low:.3g}", f"{dxy_high:.3g}"],
        fontsize=6,
    )
    key_ax.set_yticks(
        [0, 1],
        [f"{dz_low:.3g}", f"{dz_high:.3g}"],
        fontsize=6,
    )
    key_ax.set_title(
        "outline color\nlinear; high = q95",
        fontsize=7,
        pad=2,
    )

    fig.suptitle(
        title,
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.84,
        bottom=0.15,
        top=0.91,
        wspace=0.34,
        hspace=0.45,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)
    return [int(x) for x in selected_indices]


def main() -> None:
    from datasets import cms_streaming, jetclass_streaming

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("jetclass", "cms"), default="jetclass")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="JetClass root (or a split directory), or CMS root with class directories.")
    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument("--labels", default="label_fQCD", help="Comma-separated labels.")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None,
                        help="JetClass only: defaults to val when dataset-root is not a split directory.")
    parser.add_argument("--sample-events", type=int, default=256,
                        help="Size of the random candidate pool (at least rows * cols).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("plots/random_jet_events.png"))
    args = parser.parse_args()
    if min(args.rows, args.cols, args.sample_events, args.max_num_particles) < 1:
        parser.error("rows, cols, sample-events and max-num-particles must be positive")
    labels = list(dict.fromkeys(x.strip() for x in args.labels.split(",") if x.strip()))
    if not labels:
        parser.error("labels must not be empty")
    count = max(args.sample_events, args.rows * args.cols)
    root = args.dataset_root.expanduser().resolve()
    common = dict(labels_to_load=labels, max_num_particles=args.max_num_particles,
                  max_events=count, shuffle_files=True, shuffle_active_shards=len(labels),
                  infinite=False, seed=args.seed)
    if args.dataset == "jetclass":
        module = jetclass_streaming
        module.validate_requested_labels(labels)
        split_dirs = dict(train="train_100M", val="val_5M", test="test_20M")
        if args.split is not None or not any(root.glob("*.root")):
            root = root / split_dirs[args.split or "val"]
        label_axis = module.JETCLASS_LABELS
        dataset = module.JetClassIterableDataset(
            split_dir=str(root), particle_features=module.PARTICLE_FEATURES, **common)
    else:
        if args.split is not None:
            parser.error("--split is JetClass-only; CMS reads the supplied class directories directly")
        module = cms_streaming
        label_axis = module.CMS_LABELS
        dataset = module.CMSIterableDataset(
            files_by_label=module.discover_cms_files(str(root), labels),
            label_axis=label_axis, particle_features=module.PARTICLE_FEATURES, **common)
    chunks, label_indices = [], []
    for particles, target in dataset:
        chunks.append(particles.numpy())
        label_indices.append(int(target.argmax()))
    if not chunks:
        raise RuntimeError("No events available for the requested labels/split.")
    events = np.stack(chunks)
    event_labels = [label_axis[i] for i in label_indices]
    selected = plot_random_jet_events(
        events=events, feature_names=module.PARTICLE_FEATURES, output_path=args.output,
        num_events=args.rows * args.cols, seed=args.seed, nrows=args.rows, ncols=args.cols,
        event_labels=event_labels, title=f"{args.dataset}: {', '.join(labels)}",
    )
    print(f"Saved {len(selected)} jets to {args.output}")


if __name__ == "__main__":
    main()
