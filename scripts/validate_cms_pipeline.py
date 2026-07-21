#!/usr/bin/env python3
"""Validate CMS discovery, family splits, feature schema, and one-hot labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from helpers.cms_streaming import (
    CMS_BATCH_NORMALIZED_FEATURES,
    CMS_PARTICLE_FEATURES,
    CMSIterableDataset,
    CMS_LABELS,
    cms_split_manifest,
    discover_cms_files_by_label_family,
    read_file,
    split_cms_files_by_family,
)


def parse_csv(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated label.")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--labels", default=",".join(CMS_LABELS))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pt-min", type=float, default=None)
    parser.add_argument("--pt-max", type=float, default=None)
    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument(
        "--max-families-per-label",
        type=int,
        default=0,
        help="0 validates one shard from every production family.",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    args = parser.parse_args()

    labels = parse_csv(args.labels)
    discovered = discover_cms_files_by_label_family(
        str(args.dataset_root), labels
    )
    splits = split_cms_files_by_family(
        discovered,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    manifest = cms_split_manifest(splits)

    metadata_dataset = CMSIterableDataset(
        files_by_label_family=splits["train"],
        labels_to_load=labels,
        label_axis=CMS_LABELS,
        particle_features=CMS_PARTICLE_FEATURES,
        max_num_particles=args.max_num_particles,
        lowerpt=args.pt_min,
        upperpt=args.pt_max,
        infinite=False,
        shuffle_files=False,
        shuffle_active_shards=max(1, len(labels)),
        min_active_families_per_class=1,
        rank=0,
        world_size=1,
    )
    assert metadata_dataset.feature_names == CMS_PARTICLE_FEATURES
    assert (
        metadata_dataset.batch_normalized_feature_names
        == CMS_BATCH_NORMALIZED_FEATURES
    )
    assert metadata_dataset.feature_names[:4] == [
        "part_px", "part_py", "part_pz", "part_energy"
    ]

    print("CMS particle feature order:")
    for index, name in enumerate(CMS_PARTICLE_FEATURES):
        print(f"  {index:2d}: {name}")
    print("CMS batch-normalized features:")
    for name in CMS_BATCH_NORMALIZED_FEATURES:
        print(f"  - {name}")

    print("\nROOT-shard split counts by class × family:")
    for label in labels:
        for family in sorted(discovered[label]):
            counts = {
                split: len(splits[split][label].get(family, []))
                for split in ("train", "val", "test")
            }
            total = len(discovered[label][family])
            assert sum(counts.values()) == total
            print(
                f"  {label:12s} {family:32s} total={total:4d} "
                f"train={counts['train']:4d} val={counts['val']:3d} "
                f"test={counts['test']:3d}"
            )

    # Verify global disjointness explicitly.
    split_paths = {}
    for split_name, label_map in splits.items():
        split_paths[split_name] = {
            path
            for family_map in label_map.values()
            for paths in family_map.values()
            for path in paths
        }
    assert split_paths["train"].isdisjoint(split_paths["val"])
    assert split_paths["train"].isdisjoint(split_paths["test"])
    assert split_paths["val"].isdisjoint(split_paths["test"])
    print(f"\nSplit manifest SHA256: {manifest['sha256']}")

    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote manifest: {args.manifest_output}")

    print("\nReading one training shard per selected production family:")
    for label in labels:
        families = sorted(splits["train"][label])
        if args.max_families_per_label > 0:
            families = families[: args.max_families_per_label]
        for family in families:
            filepath = splits["train"][label][family][0]
            x_particles, x_jets, y = read_file(
                filepath,
                max_num_particles=args.max_num_particles,
                particle_features=CMS_PARTICLE_FEATURES,
                jet_features=["jet_pt"],
                labels=CMS_LABELS,
                lowerpt=args.pt_min,
                upperpt=args.pt_max,
                label_name=label,
            )
            expected_label_index = CMS_LABELS.index(label)
            if len(y) > 0:
                assert np.allclose(y.sum(axis=1), 1.0)
                assert np.all(y[:, expected_label_index] == 1.0)
                assert x_particles.shape[1] == len(CMS_PARTICLE_FEATURES)
                assert np.isfinite(x_particles).all()
                assert np.isfinite(x_jets).all()
                valid_counts = np.count_nonzero(
                    np.any(np.transpose(x_particles, (0, 2, 1)) != 0, axis=-1),
                    axis=1,
                )
                pt_text = (
                    f"jet_pt=[{x_jets[:, 0].min():.3f}, "
                    f"{x_jets[:, 0].max():.3f}]"
                )
                node_text = (
                    f"valid_nodes=[{valid_counts.min()}, {valid_counts.max()}]"
                )
            else:
                pt_text = "jet_pt=[empty after cuts]"
                node_text = "valid_nodes=[empty]"
            print(
                f"  {label:12s} {family:32s} events={len(y):7d} "
                f"x={tuple(x_particles.shape)} y={tuple(y.shape)} "
                f"{pt_text} {node_text}"
            )
            if len(y) > 0:
                for feature_name in (
                    "Cpfcan_dxysig",
                    "log_Cpfcan_dxysig",
                    "Cpfcan_dz",
                ):
                    feature_index = CMS_PARTICLE_FEATURES.index(feature_name)
                    values = x_particles[:, feature_index, :]
                    valid_values = values[np.isfinite(values)]
                    print(
                        f"      {feature_name:22s} "
                        f"min={valid_values.min():.6g} "
                        f"max={valid_values.max():.6g} "
                        f"mean={valid_values.mean():.6g}"
                    )

    print("\nCMS pipeline validation completed successfully.")


if __name__ == "__main__":
    main()
