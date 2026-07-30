#!/usr/bin/env python3
"""Split pre-shuffled CMS ROOT shards into smaller contiguous shards.

This is intentionally a lightweight companion to
``shuffle_cms_root_events_uproot5_multiprocess_resume_v9.py``.

It does not reshuffle events. Each input TTree is read in contiguous event
ranges and copied branch-for-branch into output files containing at most
``--events-per-file`` events. The robust Uproot 5 schema/writer implementation
is imported from the v9 shuffle script, including scalar/fixed/jagged branches,
shared count-branch regeneration, compression, and atomic ``.partial`` writes.

Place this script in the same directory as the v9 shuffle script.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

from shuffle_cms_root_events_uproot5_multiprocess_resume_v9 import (
    DEFAULT_TREE_PATH,
    FALLBACK_TREE_PATH,
    AtomicRootWriter,
    ShuffleError,
    _compression_object,
    _import_modern_io,
    _inspect_reference_schema,
    _metadata_signature,
    _reference_metadata_signature,
    _resolve_tree,
    _tree_arrays_range,
    _tree_branch_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split pre-shuffled CMS ROOT files into smaller contiguous shards "
            "without changing event order."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--file-glob", default="hbb_*.root")
    parser.add_argument("--output-prefix", default="hbb")
    parser.add_argument("--events-per-file", type=int, default=50_000)
    parser.add_argument("--tree-path", default=DEFAULT_TREE_PATH)
    parser.add_argument("--fallback-tree-path", default=FALLBACK_TREE_PATH)
    parser.add_argument(
        "--output-tree-path",
        default=None,
        help="Default: preserve the tree path resolved from the reference file.",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "zlib", "lzma", "lz4"),
        default="zlib",
    )
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip output event-count and branch-schema verification.",
    )
    parser.set_defaults(verify=True)
    return parser


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path]:
    source_dir = args.source_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise ShuffleError(f"Source directory does not exist: {source_dir}")
    if source_dir == target_dir:
        raise ShuffleError(
            "SOURCE_DIR and TARGET_DIR must differ. Write to a separate "
            "directory, verify the outputs, then replace the old directory."
        )
    if int(args.events_per_file) <= 0:
        raise ShuffleError("--events-per-file must be positive.")
    if not str(args.output_prefix).strip():
        raise ShuffleError("--output-prefix must not be empty.")

    return source_dir, target_dir


def verify_output(
    path: Path,
    *,
    expected_entries: int,
    output_tree_path: str,
    schema,
    uproot,
) -> None:
    with uproot.open(str(path), num_workers=1) as handle:
        if output_tree_path not in handle:
            raise ShuffleError(
                f"Verification failed: {path} has no tree {output_tree_path!r}."
            )
        tree = handle[output_tree_path]
        actual_entries = int(tree.num_entries)
        if actual_entries != int(expected_entries):
            raise ShuffleError(
                f"Verification failed for {path}: expected "
                f"{expected_entries:,} entries, found {actual_entries:,}."
            )

        expected_names = set(schema.all_output_names)
        actual_names = set(_tree_branch_names(tree))
        if actual_names != expected_names:
            raise ShuffleError(
                f"Verification failed for {path}: missing="
                f"{sorted(expected_names - actual_names)}, extra="
                f"{sorted(actual_names - expected_names)}."
            )

        actual_signature = _metadata_signature(tree, schema)
        expected_signature = _reference_metadata_signature(schema)
        if actual_signature != expected_signature:
            raise ShuffleError(
                f"Verification failed for {path}: branch metadata differs "
                "from the reference schema."
            )


def main() -> None:
    args = build_parser().parse_args()
    source_dir, target_dir = validate_args(args)
    uproot, awkward = _import_modern_io()

    source_files = sorted(source_dir.glob(args.file_glob))
    source_files = [
        path
        for path in source_files
        if path.is_file() and not path.name.endswith(".partial")
    ]
    if not source_files:
        raise ShuffleError(
            f"No files matching {args.file_glob!r} in {source_dir}."
        )

    schema, reference_path, checked = _inspect_reference_schema(
        source_files,
        args.tree_path,
        args.fallback_tree_path,
        uproot,
    )
    output_tree_path = (
        str(args.output_tree_path).strip("/")
        if args.output_tree_path is not None
        else str(schema.tree_path).strip("/")
    )
    if not output_tree_path:
        raise ShuffleError("Resolved output TTree path is empty.")

    reference_signature = _reference_metadata_signature(schema)

    # Metadata-only preflight: validate every source and build the complete
    # output plan before writing anything.
    source_info: List[Tuple[Path, str, int]] = []
    total_events = 0
    total_outputs = 0

    for path in source_files:
        with uproot.open(str(path), num_workers=1) as handle:
            tree, resolved_tree_path = _resolve_tree(
                handle,
                args.tree_path,
                args.fallback_tree_path,
            )
            entries = int(tree.num_entries)
            signature = _metadata_signature(tree, schema)
            if signature != reference_signature:
                raise ShuffleError(
                    f"Source schema differs from the reference file: {path}"
                )

        source_info.append((path, resolved_tree_path, entries))
        total_events += entries
        if entries > 0:
            total_outputs += int(math.ceil(entries / float(args.events_per_file)))

    if total_outputs == 0:
        raise ShuffleError("All matching source ROOT files are empty.")

    width = max(2, len(str(total_outputs)))
    target_dir.mkdir(parents=True, exist_ok=True)

    plan: List[Tuple[Path, str, int, int, Path]] = []
    output_index = 1
    for source_path, resolved_tree_path, entries in source_info:
        for start in range(0, entries, int(args.events_per_file)):
            stop = min(start + int(args.events_per_file), entries)
            output_path = target_dir / (
                f"{args.output_prefix}_{output_index:0{width}d}.root"
            )
            plan.append(
                (source_path, resolved_tree_path, start, stop, output_path)
            )
            output_index += 1

    collisions = []
    for _, _, _, _, output_path in plan:
        partial_path = output_path.with_name(output_path.name + ".partial")
        if output_path.exists() or partial_path.exists():
            collisions.append(output_path)
    if collisions and not args.overwrite:
        preview = "\n  ".join(str(path) for path in collisions[:20])
        raise ShuffleError(
            "Generated output paths already exist. Pass --overwrite to replace "
            f"them. First collisions:\n  {preview}"
        )

    compression = _compression_object(
        uproot,
        args.compression,
        int(args.compression_level),
    )

    print(f"Source directory: {source_dir}", flush=True)
    print(f"Target directory: {target_dir}", flush=True)
    print(
        f"Reference schema: {reference_path} "
        f"(found after checking {checked} file(s))",
        flush=True,
    )
    print(
        f"Branches copied: {len(schema.branches)} data + "
        f"{len(schema.count_names)} regenerated count branches",
        flush=True,
    )
    print(f"Input files: {len(source_files)}", flush=True)
    print(f"Input events: {total_events:,}", flush=True)
    print(
        f"Output files: {len(plan)}; at most "
        f"{args.events_per_file:,} events each",
        flush=True,
    )
    print(f"Output tree: {output_tree_path}", flush=True)

    started = time.time()
    written_events = 0
    current_source: Path | None = None
    source_handle = None
    source_tree = None

    try:
        for position, (
            source_path,
            resolved_tree_path,
            start,
            stop,
            output_path,
        ) in enumerate(plan, start=1):
            if current_source != source_path:
                if source_handle is not None:
                    source_handle.close()
                source_handle = uproot.open(str(source_path), num_workers=1)
                source_tree = source_handle[resolved_tree_path]
                current_source = source_path
                print(
                    f"[{position}/{len(plan)}] Reading {source_path.name}",
                    flush=True,
                )

            assert source_tree is not None
            expected = stop - start
            arrays = _tree_arrays_range(
                source_tree,
                schema.data_names,
                start,
                stop,
            )

            writer = AtomicRootWriter(
                output_path,
                schema,
                output_tree_path,
                compression,
                uproot,
                awkward,
                bool(args.overwrite),
            )
            try:
                writer.extend(arrays)
                if writer.events_written != expected:
                    raise ShuffleError(
                        f"Writer count mismatch for {output_path}: wrote "
                        f"{writer.events_written:,}, expected {expected:,}."
                    )
                writer.finalize()
            except BaseException:
                writer.abort()
                raise
            finally:
                del arrays

            if args.verify:
                verify_output(
                    output_path,
                    expected_entries=expected,
                    output_tree_path=output_tree_path,
                    schema=schema,
                    uproot=uproot,
                )

            written_events += expected
            print(
                f"  -> {output_path.name}: source entries "
                f"[{start:,}, {stop:,}) = {expected:,}",
                flush=True,
            )
    finally:
        if source_handle is not None:
            source_handle.close()

    if written_events != total_events:
        raise ShuffleError(
            f"Final event-count mismatch: wrote {written_events:,}, "
            f"expected {total_events:,}."
        )

    elapsed = max(time.time() - started, 1e-9)
    print(
        f"Done: {total_events:,} events -> {len(plan)} files in "
        f"{elapsed / 60.0:.2f} min "
        f"({total_events / elapsed:,.0f} events/s).",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
