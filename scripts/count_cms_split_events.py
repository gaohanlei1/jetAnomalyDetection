#!/usr/bin/env python3
"""Count raw and usable jet entries for each label and split in a CMS manifest.

Uses cms_streaming.load_and_preprocess_cms_file without duplicating its cuts.
Requires the same Python environment as cms_streaming (torch, uproot, awkward,
numpy). Place in the repository or scripts/, or pass --cms-streaming explicitly.
Manifest basenames are resolved under DATA_DIR/<label directory>/ so --data-dir
always selects the dataset being counted, even when old absolute paths exist.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_cms(path):
    if path is None:
        roots = [Path.cwd(), *Path(__file__).resolve().parents]
        path = next((root / "datasets/cms_streaming.py" for root in roots
                     if (root / "datasets/cms_streaming.py").is_file()), None)
        if path is None:
            raise FileNotFoundError("Cannot find datasets/cms_streaming.py; pass --cms-streaming PATH.")
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("_cms_split_counter_backend", path)
    module = importlib.util.module_from_spec(spec)
    # dataclass definitions require the module to be registered before execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def resolve_splits(manifest, data_dir, cms):
    if manifest.get("version") != 2:
        raise ValueError("Expected a version-2 CMS split manifest.")
    splits = manifest["splits"]
    if manifest.get("sha256") != cms.cms_split_manifest(splits)["sha256"]:
        raise ValueError("Manifest hash does not match its file assignments.")
    result, seen = {}, set()
    for split in ("train", "val", "test"):
        result[split] = {}
        for label, paths in splits[split].items():
            cms.validate_cms_labels([label])
            if not isinstance(paths, list):
                raise ValueError(f"Expected file list for {split}/{label}.")
            resolved = []
            for original in paths:
                path = (data_dir / cms.CMS_LABEL_TO_DIRECTORY[label] / Path(original).name).resolve()
                if path in seen:
                    raise ValueError(f"Duplicate file or overlapping splits: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                seen.add(path)
                resolved.append(path)
            result[split][label] = resolved
    return result


def print_counts(title, files, raw, usable):
    retention = f"{usable / raw:.6%}" if raw else "N/A"
    print(f"{title} files={files} raw_events={raw} usable_events={usable} "
          f"dropped_events={raw-usable} retention={retention}", flush=True)


def count_file(path, label, cms, args):
    import uproot

    with uproot.open(path) as handle:
        raw = int(cms._resolve_tree(handle).num_entries)
    usable = 0
    for start in range(0, raw, args.chunk_entries):
        x, _ = cms.load_and_preprocess_cms_file(
            str(path), label_name=label, label_axis=cms.CMS_LABELS,
            particle_features=cms.PARTICLE_FEATURES,
            max_num_particles=args.max_num_particles, min_nodes=args.min_nodes,
            lowerpt=args.pt_min, upperpt=args.pt_max,
            entry_start=start, entry_stop=min(start + args.chunk_entries, raw),
        )
        usable += len(x)
        del x
    return raw, usable


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", "--dataset-root", type=Path,
                        default=Path("/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5"))
    parser.add_argument("--cms-streaming", type=Path,
                        help="Path to the repository's datasets/cms_streaming.py")
    parser.add_argument("--chunk-entries", type=int, default=10000,
                        help="Raw entries processed at once; does not limit total events")
    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--pt-min", type=float, default=None)
    parser.add_argument("--pt-max", type=float, default=None)
    args = parser.parse_args()
    if args.chunk_entries < 1 or args.max_num_particles < 1 or args.min_nodes < 0:
        parser.error("chunk-entries/max-num-particles must be positive; min-nodes must be nonnegative")
    if args.pt_min is not None and args.pt_max is not None and args.pt_min >= args.pt_max:
        parser.error("pt-min must be smaller than pt-max")

    cms, backend_path = load_cms(args.cms_streaming)
    manifest = json.loads(args.manifest.expanduser().read_text())
    data_dir = args.data_dir.expanduser().resolve()
    splits = resolve_splits(manifest, data_dir, cms)
    print(f"# cms_streaming={backend_path}\n# data_dir={data_dir}")
    print(f"# max_num_particles={args.max_num_particles} min_nodes={args.min_nodes} "
          f"pt_min={args.pt_min} pt_max={args.pt_max}")
    print("# Other jet/particle cuts use the imported cms_streaming defaults.")
    print(f"# particle_features={cms.PARTICLE_FEATURES}", flush=True)
    totals = {}
    for split, labels in splits.items():
        split_files = split_raw = split_usable = 0
        for label, paths in labels.items():
            label_raw = label_usable = 0
            for path in paths:
                raw, usable = count_file(path, label, cms, args)
                label_raw += raw
                label_usable += usable
                print_counts(f"FILE split={split} label={label} path={path}", 1, raw, usable)
            print_counts(f"LABEL SUMMARY split={split} label={label}", len(paths), label_raw, label_usable)
            split_files += len(paths)
            split_raw += label_raw
            split_usable += label_usable
        totals[split] = (split_files, split_raw, split_usable)
        print_counts(f"SPLIT SUMMARY split={split}", *totals[split])

    grand = tuple(sum(row[i] for row in totals.values()) for i in range(3))
    print("\n# FINAL SPLIT COUNTS", flush=True)
    for split, counts in totals.items():
        print_counts(f"SPLIT split={split}", *counts)
        raw_share = f"{counts[1] / grand[1]:.6%}" if grand[1] else "N/A"
        usable_share = f"{counts[2] / grand[2]:.6%}" if grand[2] else "N/A"
        print(f"  raw_share={raw_share} usable_share={usable_share}", flush=True)
    print_counts("ALL SPLITS", *grand)


if __name__ == "__main__":
    main()
