#!/usr/bin/env python3
"""Compare raw constituent scalar-pT sums with jet_pt in every CMS shard.

Requires numpy, awkward, uproot. Discovery and tree fallback follow
datasets/cms_streaming.py; no torch dependency or preprocessing is needed.
An 'event' here means one ROOT tree entry (one jet), not a unique collision.
All matching shards are scanned, without a train/val/test manifest split.
Print statistics after each file and each label; never print individual entries.
"""

import argparse
from pathlib import Path


# Same directory/prefix mapping as cms_streaming.py.
LABEL_FILES = {
    "label_QCD": ("qcd", "qcd_"),
    "label_Hbb": ("hbb", "hbb_"),
    "label_Wqq": ("wjets", "wqq_"),
    "label_Zqq": ("zjets", "zqq_"),
    "label_Tbqq": ("ttbar", "tbqq_"),
    "label_Real": ("real", "data_"),
}


def discover_files(root, labels):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    result = {}
    for label in labels:
        directory, prefix = LABEL_FILES[label]
        folder = root / directory
        files = sorted(p.resolve() for p in folder.glob(f"{prefix}*.root")
                       if p.is_file() and p.stat().st_size > 0)
        if not files:
            raise FileNotFoundError(f"No non-empty files: {folder}/{prefix}*.root")
        result[label] = files
    return result


class Stats:
    """Streaming population statistics; invalid values are counted explicitly."""

    def __init__(self):
        self.n = self.invalid = 0
        self.mean = self.m2 = 0.0
        self.low, self.high = float("inf"), float("-inf")

    def add(self, values, np):
        good = np.isfinite(values)
        self.invalid += int((~good).sum())
        values = values[good]
        n = len(values)
        if not n:
            return
        mean = float(values.mean())
        delta = mean - self.mean
        total = self.n + n
        self.m2 += float(((values - mean) ** 2).sum()) + delta**2 * self.n * n / total
        self.mean += delta * n / total
        self.n = total
        self.low = min(self.low, float(values.min()))
        self.high = max(self.high, float(values.max()))

    def __str__(self):
        if not self.n:
            return f"finite=0 invalid={self.invalid}; no finite values"
        return (f"finite={self.n} invalid={self.invalid} mean={self.mean:.9g} "
                f"std={(self.m2 / self.n)**0.5:.9g} min={self.low:.9g} max={self.high:.9g}")


def print_summary(title, stats, total, above, equal, below):
    lines = [f"# {title} total_events={total}"]
    lines.extend(f"# {name}: {summary}" for name, summary in stats.items())
    valid = above + equal + below
    lines.append(f"# delta>0: {above}; delta==0: {equal}; delta<0: {below}; invalid: {total-valid}")
    if valid:
        lines.append(f"# fraction(sum_pt > jet_pt), finite entries: {above / valid:.6%}")
    print("\n".join(lines), flush=True)


def scan_label(label, files, args):
    import awkward as ak
    import numpy as np
    import uproot

    stats = {name: Stats() for name in ("jet_pt", "sum_pt", "delta_pt", "sum_pt/jet_pt")}
    total = above = equal = below = 0
    print(f"\n# label={label} files={len(files)}", flush=True)
    for path in files:
        count = 0
        file_stats = {name: Stats() for name in stats}
        file_above = file_equal = file_below = 0
        with uproot.open(path) as handle:
            tree = handle["deepntuplizerAK8/tree"] if "deepntuplizerAK8/tree" in handle else handle["tree"]
            for arrays in tree.iterate(["jet_pt", "Cpfcan_pt", "Npfcan_pt"],
                                       step_size=args.step_size, library="ak"):
                jet = ak.to_numpy(arrays["jet_pt"]).astype(np.float64)
                # Cast BEFORE reduction; retain every stored charged/neutral candidate.
                summed = ak.to_numpy(
                    ak.sum(ak.values_astype(arrays["Cpfcan_pt"], np.float64), axis=1)
                    + ak.sum(ak.values_astype(arrays["Npfcan_pt"], np.float64), axis=1)
                )
                delta = summed - jet
                ratio = np.full(len(jet), np.nan)
                np.divide(summed, jet, out=ratio, where=np.isfinite(jet) & (jet != 0))
                for name, values in zip(stats, (jet, summed, delta, ratio)):
                    stats[name].add(values, np)
                    file_stats[name].add(values, np)
                finite = np.isfinite(delta)
                file_above += int((finite & (delta > 0)).sum())
                file_equal += int((finite & (delta == 0)).sum())
                file_below += int((finite & (delta < 0)).sum())
                count += len(jet)
            if count != tree.num_entries:
                raise RuntimeError(f"Entry count mismatch: {path}: {count} != {tree.num_entries}")
        total += count
        above += file_above
        equal += file_equal
        below += file_below
        print_summary(f"FILE label={label} file={path}", file_stats, count,
                      file_above, file_equal, file_below)
    print_summary(f"LABEL SUMMARY label={label} files={len(files)}", stats, total,
                  above, equal, below)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/HEP/export/home/lwang223/jet-anomaly-data/ak8-v5"))
    parser.add_argument("--labels", "--request-labels", nargs="+", required=True,
                        choices=tuple(LABEL_FILES), help="Labels to scan independently")
    parser.add_argument("--step-size", default="100 MB", help="Uproot input chunk size")
    # Retain compatibility with earlier commands; summaries are now always used.
    parser.add_argument("--summary-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    files = discover_files(args.dataset_root.expanduser(), args.labels)
    print("# delta_pt = sum(Cpfcan_pt) + sum(Npfcan_pt) - jet_pt (GeV)")
    print("# Scalar pT sum; no cuts, truncation, padding, or normalization.")
    for label, paths in files.items():
        scan_label(label, paths, args)


if __name__ == "__main__":
    main()
