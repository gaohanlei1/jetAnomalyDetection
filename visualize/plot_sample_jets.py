"""Plot one random jet per CMS jet type for visual inspection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.cms_streaming import (
    CMS_LABELS,
    PARTICLE_FEATURES,
    discover_cms_files,
    load_and_preprocess_cms_file,
)
from visualize.plot_jet_eta_phi import plot_jet_eta_phi


CMS_MC_LABELS = [label for label in CMS_LABELS if label != "label_Real"]

CMS_PARTICLE_TYPE_INDICES = {
    "electron": PARTICLE_FEATURES.index("part_isElectron"),
    "muon": PARTICLE_FEATURES.index("part_isMuon"),
    "photon": PARTICLE_FEATURES.index("part_isPhoton"),
    "charged_hadron": PARTICLE_FEATURES.index("part_isChargedHadron"),
    "neutral_hadron": PARTICLE_FEATURES.index("part_isNeutralHadron"),
}


def plot_random_jets_per_label(
    dataset_root: str,
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
    seed: int = 42,
    max_num_particles: int = 128,
    min_nodes: int = 4,
    max_events_per_file: int = 64,
) -> list[Path]:
    """Load and plot one random jet from each available CMS jet type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_to_plot = list(CMS_MC_LABELS if labels is None else labels)
    files_by_label = discover_cms_files(dataset_root, labels_to_plot)
    rng = np.random.default_rng(seed)
    saved_paths: list[Path] = []

    for label in labels_to_plot:
        files = files_by_label[label]
        filepath = files[int(rng.integers(len(files)))]
        x_particles, _ = load_and_preprocess_cms_file(
            filepath,
            label_name=label,
            label_axis=CMS_LABELS,
            particle_features=PARTICLE_FEATURES,
            max_num_particles=max_num_particles,
            lowerpt=None,
            upperpt=None,
            min_nodes=min_nodes,
            max_events=max_events_per_file,
        )
        if len(x_particles) == 0:
            print(f"Skipping {label}: no valid events in {filepath}")
            continue

        event_idx = int(rng.integers(len(x_particles)))
        jet = torch.from_numpy(x_particles[event_idx : event_idx + 1])
        output_path = output_dir / f"{label}.png"
        plot_jet_eta_phi(
            jet,
            output_path,
            title=label,
            pt_frac_label="log_pt_fraction",
            particle_type_indices=CMS_PARTICLE_TYPE_INDICES,
        )
        saved_paths.append(output_path)
        print(f"Saved {label} jet from {filepath} (event {event_idx}) -> {output_path}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/HEP/export/home/hgao50/jet-anomaly-data/ak8-v4",
        help="CMS dataset root with one subdirectory per jet type.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/sample_jets",
        help="Directory where jet plots will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot_random_jets_per_label(
        args.dataset_root,
        args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
