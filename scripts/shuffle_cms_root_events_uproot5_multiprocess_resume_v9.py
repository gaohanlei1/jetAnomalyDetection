#!/usr/bin/env python3
"""Multiprocess external event-level shuffle for CMS DeepNTuplizer ROOT datasets.

This script uses modern Uproot 5 and Awkward 2 and targets
``uproot==5.6.6`` with ``awkward==2.7.4``. It preserves every supported TTree branch
and only changes the outer event order.

Dataset layout
--------------
The source directory is expected to contain one subdirectory per jet type::

    SOURCE/
      qcd/*.root
      hbb/*.root
      wjets/*.root
      zjets/*.root
      ttbar/*.root

The target directory receives the same subdirectory layout. Final files contain
at most ``--events-per-file`` entries and are named ``PREFIX_NN.root``.
Known default prefixes are qcd, hbb, wqq, zqq, and tbqq; all mappings can be
overridden with ``--prefix-map SUBDIR=PREFIX``. The output TTree defaults to the top-level name ``tree`` for compatibility with
the existing CMS loader, though modern Uproot can also create nested paths.

Algorithm
---------
For each jet type independently:

1. Open only the first non-empty source file to infer the writable TTree schema.
   In fixed-bucket mode, the remaining files are not pre-scanned.
2. Phase-1 worker processes receive disjoint source-file subsets. Each process
   independently opens, decodes, buckets, and writes its own part file for every
   logical bucket. No decoded Awkward arrays cross a process boundary.
3. Phase 2A workers sort each bounded bucket part by a deterministic 64-bit shuffle
   key. A logical bucket therefore consists of multiple independently sorted
   ROOT runs, one per Phase-1 process.
4. Phase-2 worker processes receive disjoint contiguous ranges of final output
   files. Each process independently merges the sorted runs for the logical
   buckets overlapping its range, materializes at most one final-file-sized event
   batch at a time, and compresses its own final ROOT files. No decoded event
   arrays cross a process boundary, and no two processes write the same file.

Each source event receives a deterministic, seed-dependent, unique 64-bit
SplitMix key. The key chooses its logical bucket and defines the complete order
inside that bucket. Results are independent of process scheduling, with no event
repeated or dropped. The internal key branch is removed from final output.

Supported branch shapes
-----------------------
* One scalar numeric value per event.
* Fixed-size numeric NumPy arrays per event.
* One-level jagged numeric arrays (ROOT leaf lists), including the usual
  Cpfcan_* and Npfcan_* branches.

Unsupported object/string branches and nested jagged arrays cause an explicit
error before data writing begins.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import tempfile
import traceback
import multiprocessing as mp
from contextlib import ExitStack
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

# Every Phase-1 worker is already a separate process. Prevent NumPy/BLAS from
# creating nested thread pools inside each process and oversubscribing the node.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np




DEFAULT_TREE_PATH = "deepntuplizerAK8/tree"
FALLBACK_TREE_PATH = "tree"
SHUFFLE_KEY_BRANCH = "__cms_external_shuffle_key"
PHASE2A_CHECKPOINT_NAME = "phase2a_checkpoint.json"
_TEMP_RUN_RE = re.compile(
    r"^bucket_(?P<bucket>\d+)_worker_(?P<worker>\d+)_run_(?P<run>\d+)"
    r"(?P<sorted>_sorted)?\.root$"
)
DEFAULT_PREFIX_BY_SUBDIR = {
    "qcd": "qcd",
    "hbb": "hbb",
    "wjets": "wjets",
    "wqq": "wqq",
    "zjets": "zjets",
    "zqq": "zqq",
    "ttbar": "ttbar",
    "tbqq": "tbqq",
}


@dataclass(frozen=True)
class BranchSpec:
    name: str
    kind: str  # "scalar", "fixed", or "jagged"
    dtype: str
    fixed_shape: Tuple[int, ...] = ()
    count_name: Optional[str] = None
    interpretation: str = ""


@dataclass
class TreeSchema:
    tree_path: str
    tree_title: str
    branches: List[BranchSpec]

    @property
    def data_names(self) -> List[str]:
        return [spec.name for spec in self.branches]

    @property
    def count_names(self) -> List[str]:
        return sorted(
            {spec.count_name for spec in self.branches if spec.count_name is not None}
        )

    @property
    def all_output_names(self) -> List[str]:
        return self.data_names + self.count_names

    @property
    def jagged_by_count(self) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = defaultdict(list)
        for spec in self.branches:
            if spec.count_name is not None:
                output[spec.count_name].append(spec.name)
        return dict(output)

    def signature(self) -> List[Tuple[Any, ...]]:
        return [
            (
                spec.name,
                spec.kind,
                spec.dtype,
                spec.fixed_shape,
                spec.count_name,
                spec.interpretation,
            )
            for spec in self.branches
        ]


@dataclass
class FileInfo:
    path: str
    entries: int
    tree_path: str


@dataclass
class JetResult:
    jet_type: str
    prefix: str
    source_events: Optional[int]
    source_files: int
    temporary_buckets: int
    temporary_bucket_counts: List[int]
    output_files: List[str]
    output_file_counts: List[int]
    elapsed_seconds: float


class ShuffleError(RuntimeError):
    pass



def _import_modern_io():
    try:
        import uproot  # type: ignore
    except ImportError as exc:
        raise ShuffleError(
            "This script requires modern Uproot. The target environment uses "
            "uproot==5.6.6."
        ) from exc

    try:
        import awkward as awkward  # type: ignore
    except ImportError as exc:
        raise ShuffleError(
            "This script requires Awkward Array 2. The target environment uses "
            "awkward==2.7.4."
        ) from exc

    uproot_version = str(getattr(uproot, "__version__", "unknown"))
    awkward_version = str(getattr(awkward, "__version__", "unknown"))
    if uproot_version != "5.6.6" or awkward_version != "2.7.4":
        print(
            "WARNING: targeted for uproot==5.6.6 and awkward==2.7.4; "
            "found uproot=={} and awkward=={}.".format(
                uproot_version, awkward_version
            ),
            file=sys.stderr,
        )
    return uproot, awkward


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    return value.split(";", 1)[0]


def _decode_title(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_tree(handle: Any, preferred: str, fallback: str) -> Tuple[Any, str]:
    for candidate in (preferred, fallback):
        if candidate and candidate in handle:
            return handle[candidate], candidate
    raise KeyError(
        "ROOT file contains neither {!r} nor {!r}.".format(preferred, fallback)
    )


def _tree_branch_names(tree: Any) -> List[str]:
    # DeepNTuplizer leaves are direct TBranches. Avoid recursively including
    # subbranches if an unrelated split object ever appears in the same tree.
    try:
        keys = tree.keys(recursive=False)
    except TypeError:
        keys = tree.keys()
    return [_decode_name(name) for name in keys]



def _branch_count_name(branch: Any) -> Optional[str]:
    count_branch = None
    try:
        count_branch = branch.count_branch
    except (AttributeError, TypeError, ValueError):
        # Compatibility fallback for unusual branch wrappers.
        try:
            count_branch = branch.countbranch
        except (AttributeError, TypeError, ValueError):
            return None
    if count_branch is None:
        return None
    return _decode_name(count_branch.name)



def _is_jagged_array(array: Any) -> bool:
    try:
        import awkward as awkward  # type: ignore
        if isinstance(array, awkward.Array):
            return bool(array.ndim > 1 and not array.layout.purelist_isregular)
    except Exception:
        pass
    return False


def _array_length(arrays: Mapping[str, Any]) -> int:
    if not arrays:
        return 0
    first = next(iter(arrays.values()))
    return int(len(first))


def _slice_arrays(arrays: Mapping[str, Any], indexer: Any) -> Dict[str, Any]:
    # The exact same outer event indexer is applied to every branch. Jagged
    # contents remain grouped inside each event and constituent ordering is not
    # changed.
    return {name: value[indexer] for name, value in arrays.items()}



def _concatenate_arrays(
    fragments: Sequence[Mapping[str, Any]],
    branch_names: Sequence[str],
    awkward: Any,
) -> Dict[str, Any]:
    if not fragments:
        raise ValueError("Cannot concatenate an empty fragment list.")
    if len(fragments) == 1:
        return dict(fragments[0])

    output: Dict[str, Any] = {}
    for name in branch_names:
        pieces = [fragment[name] for fragment in fragments]
        if isinstance(pieces[0], awkward.Array):
            output[name] = awkward.concatenate(pieces, axis=0)
        else:
            output[name] = np.concatenate(pieces, axis=0)
    return output



def _schema_with_shuffle_key(schema: TreeSchema) -> TreeSchema:
    if SHUFFLE_KEY_BRANCH in schema.all_output_names:
        raise ShuffleError(
            "Source tree already contains reserved branch {!r}.".format(
                SHUFFLE_KEY_BRANCH
            )
        )
    return TreeSchema(
        tree_path=schema.tree_path,
        tree_title=schema.tree_title,
        branches=list(schema.branches)
        + [
            BranchSpec(
                name=SHUFFLE_KEY_BRANCH,
                kind="scalar",
                dtype=np.dtype(np.int64).str,
                interpretation="internal deterministic uint64 shuffle key stored as int64",
            )
        ],
    )


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic bijective 64-bit mixer for schedule-independent keys."""
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(_stable_seed(seed, "splitmix64-offset"))
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x



def _value_memory_bytes(value: Any) -> int:
    """Approximate resident bytes owned by one branch array."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    try:
        return int(value.nbytes)
    except Exception:
        return 0


def _arrays_memory_bytes(arrays: Mapping[str, Any]) -> int:
    return sum(_value_memory_bytes(value) for value in arrays.values())


def _materialize_event_slice(
    arrays: Mapping[str, Any],
    start: int,
    stop: int,
) -> Dict[str, Any]:
    """Copy an event range so it no longer retains a full parent bucket."""
    if stop <= start:
        raise ValueError("Cannot materialize an empty event slice.")
    indices = np.arange(int(start), int(stop), dtype=np.int64)
    return _slice_arrays(arrays, indices)


def _permute_arrays_branchwise_in_place(
    arrays: MutableMapping[str, Any],
    permutation: np.ndarray,
    branch_names: Sequence[str],
) -> MutableMapping[str, Any]:
    """Apply one event permutation while avoiding a second full bucket copy.

    Each old branch is released immediately after its permuted replacement has
    been created. Peak memory is therefore approximately one bucket plus one
    branch, rather than two complete buckets.
    """
    for name in branch_names:
        old_value = arrays.pop(name)
        arrays[name] = old_value[permutation]
        del old_value
    return arrays



def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2 ** 32)


def _parse_step(value: str) -> Union[int, str]:
    stripped = value.strip()
    if re.fullmatch(r"[0-9]+", stripped):
        parsed = int(stripped)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("read step must be positive")
        return parsed
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?\s*[kKmMgGtT]?[bB]", stripped):
        raise argparse.ArgumentTypeError(
            "read step must be an integer entry count or a memory string such "
            "as '512 MB'"
        )
    return stripped


def _parse_prefix_map(items: Sequence[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_PREFIX_BY_SUBDIR)
    for item in items:
        if "=" not in item:
            raise ShuffleError(
                "Invalid --prefix-map {!r}; expected SUBDIR=PREFIX.".format(item)
            )
        subdir, prefix = item.split("=", 1)
        subdir = subdir.strip()
        prefix = prefix.strip()
        if not subdir or not prefix:
            raise ShuffleError(
                "Invalid --prefix-map {!r}; both sides must be non-empty.".format(item)
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
            raise ShuffleError(
                "Unsafe output prefix {!r}; use letters, digits, '_', '-', or '.'."
                .format(prefix)
            )
        mapping[subdir] = prefix
    return mapping



def _compression_object(uproot: Any, name: str, level: int) -> Any:
    if name == "none":
        return None
    if not 0 <= level <= 9:
        raise ShuffleError("--compression-level must be between 0 and 9.")
    constructors = {
        "zlib": getattr(uproot, "ZLIB", None),
        "lzma": getattr(uproot, "LZMA", None),
        "lz4": getattr(uproot, "LZ4", None),
    }
    constructor = constructors.get(name)
    if constructor is None:
        raise ShuffleError(
            "Compression {!r} is unavailable in this Uproot installation."
            .format(name)
        )
    return constructor(level)


def _ensure_safe_paths(source: Path, target: Path, temp: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    temp = temp.resolve()
    if not source.is_dir():
        raise ShuffleError("Source directory does not exist: {}".format(source))
    if target == source or source in target.parents:
        raise ShuffleError(
            "Target directory must not equal or be nested inside source directory."
        )
    if temp == source or source in temp.parents:
        raise ShuffleError(
            "Temporary directory must not equal or be nested inside source directory."
        )
    if temp == target:
        raise ShuffleError("Temporary directory must differ from target directory.")


def _prepare_clean_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise ShuffleError("Expected directory path, found: {}".format(path))
        if any(path.iterdir()):
            if not overwrite:
                raise ShuffleError(
                    "Directory is not empty: {}. Pass --overwrite to replace it."
                    .format(path)
                )
            shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


def _discover_jet_dirs(source: Path, requested: Optional[Sequence[str]]) -> List[Path]:
    if requested:
        directories = [source / name for name in requested]
        missing = [str(path) for path in directories if not path.is_dir()]
        if missing:
            raise ShuffleError("Missing requested jet-type directories: {}".format(missing))
        return directories
    directories = sorted(path for path in source.iterdir() if path.is_dir())
    if not directories:
        raise ShuffleError("No jet-type subdirectories found under {}".format(source))
    return directories



def _read_sample(tree: Any, names: Sequence[str], stop: int) -> Dict[str, Any]:
    return tree.arrays(
        expressions=list(names),
        entry_start=0,
        entry_stop=int(stop),
        library="ak",
        how=dict,
    )



def _uproot_writable_dtype(dtype: np.dtype, *, jagged: bool) -> Optional[str]:
    """Return an explanatory error for primitive dtypes Uproot cannot write."""
    dtype = np.dtype(dtype)
    supported = {
        ("f", 4), ("f", 8),
        ("i", 1), ("i", 2), ("i", 4), ("i", 8),
        ("u", 1), ("u", 2), ("u", 4), ("u", 8),
        ("b", 1),
    }
    if (dtype.kind, dtype.itemsize) not in supported:
        return (
            "dtype {} is unsupported by the modern Uproot TTree writer; "
            "supported primitive output dtypes are bool, signed/unsigned "
            "integers up to 64-bit, and float32/64".format(dtype)
        )
    return None



def _build_schema(tree: Any, tree_path: str) -> TreeSchema:
    uproot, awkward = _import_modern_io()
    del uproot
    if int(tree.num_entries) <= 0:
        raise ShuffleError("Cannot infer writable branch dtypes from an empty reference tree.")

    all_names = _tree_branch_names(tree)
    count_names = set()
    count_by_data_name: Dict[str, Optional[str]] = {}
    interpretations: Dict[str, str] = {}
    for name in all_names:
        branch = tree[name]
        count_name = _branch_count_name(branch)
        count_by_data_name[name] = count_name
        if count_name is not None:
            count_names.add(count_name)
        interpretations[name] = repr(getattr(branch, "interpretation", None))

    # ROOT leaf-list count branches are regenerated by Uproot from Awkward lists.
    data_names = [name for name in all_names if name not in count_names]
    sample = _read_sample(tree, data_names, stop=min(16, int(tree.num_entries)))

    specs: List[BranchSpec] = []
    unsupported: List[str] = []
    for name in data_names:
        array = sample[name]
        count_name = count_by_data_name.get(name)
        interpretation = interpretations.get(name, "")

        if count_name is not None:
            if not isinstance(array, awkward.Array):
                unsupported.append(
                    "{}: counted branch {!r} did not decode as an Awkward Array ({})"
                    .format(name, count_name, type(array).__name__)
                )
                continue
            if array.ndim != 2:
                unsupported.append(
                    "{}: expected one-level jagged data, got ndim={} and type {}"
                    .format(name, array.ndim, array.type)
                )
                continue
            try:
                content_array = awkward.to_numpy(awkward.flatten(array, axis=1))
            except Exception as exc:
                unsupported.append(
                    "{}: could not obtain primitive jagged content: {}".format(name, exc)
                )
                continue
            if content_array.ndim != 1 or content_array.dtype.kind in ("O", "S", "U", "V"):
                unsupported.append(
                    "{}: unsupported jagged content shape/dtype {} {}".format(
                        name, content_array.shape, content_array.dtype
                    )
                )
                continue
            dtype_error = _uproot_writable_dtype(content_array.dtype, jagged=True)
            if dtype_error is not None:
                unsupported.append("{}: {}".format(name, dtype_error))
                continue
            specs.append(
                BranchSpec(
                    name=name,
                    kind="jagged",
                    dtype=content_array.dtype.str,
                    count_name=count_name,
                    interpretation=interpretation,
                )
            )
            continue

        try:
            dense = awkward.to_numpy(array) if isinstance(array, awkward.Array) else np.asarray(array)
        except Exception as exc:
            unsupported.append("{}: could not convert dense branch to NumPy: {}".format(name, exc))
            continue
        if dense.dtype.kind in ("O", "S", "U", "V"):
            unsupported.append(
                "{}: unsupported object/string/void dtype {}".format(name, dense.dtype)
            )
            continue
        dtype_error = _uproot_writable_dtype(dense.dtype, jagged=False)
        if dtype_error is not None:
            unsupported.append("{}: {}".format(name, dtype_error))
            continue
        if dense.ndim == 1:
            specs.append(
                BranchSpec(
                    name=name,
                    kind="scalar",
                    dtype=dense.dtype.str,
                    interpretation=interpretation,
                )
            )
        elif dense.ndim >= 2:
            specs.append(
                BranchSpec(
                    name=name,
                    kind="fixed",
                    dtype=dense.dtype.str,
                    fixed_shape=tuple(int(x) for x in dense.shape[1:]),
                    interpretation=interpretation,
                )
            )
        else:
            unsupported.append("{}: unsupported array rank {}".format(name, dense.ndim))

    if unsupported:
        raise ShuffleError(
            "The reference TTree contains branches that this safe copier cannot "
            "recreate with Uproot 5:\n  - " + "\n  - ".join(unsupported)
        )
    if not specs:
        raise ShuffleError("Reference TTree has no supported data branches.")

    return TreeSchema(
        tree_path=tree_path,
        tree_title=_decode_title(getattr(tree, "title", "")),
        branches=specs,
    )


def _metadata_signature(tree: Any, schema: TreeSchema) -> List[Tuple[str, Optional[str], str]]:
    output = []
    tree_names = set(_tree_branch_names(tree))
    expected_names = set(schema.all_output_names)
    if tree_names != expected_names:
        missing = sorted(expected_names - tree_names)
        extra = sorted(tree_names - expected_names)
        raise ShuffleError(
            "Branch-name mismatch. Missing={}, extra={}".format(missing, extra)
        )
    for spec in schema.branches:
        branch = tree[spec.name]
        output.append(
            (
                spec.name,
                _branch_count_name(branch),
                repr(getattr(branch, "interpretation", None)),
            )
        )
    return output


def _reference_metadata_signature(schema: TreeSchema) -> List[Tuple[str, Optional[str], str]]:
    return [
        (spec.name, spec.count_name, spec.interpretation) for spec in schema.branches
    ]


def _inspect_reference_schema(
    files: Sequence[Path],
    preferred_tree_path: str,
    fallback_tree_path: str,
    uproot: Any,
) -> Tuple[TreeSchema, Path, int]:
    """Infer the output schema from the first non-empty source ROOT file.

    This intentionally does not pre-open or count every source file. Remaining
    files are validated exactly once when Phase 1 actually processes them.
    """
    if not files:
        raise ShuffleError("No ROOT files supplied for schema inspection.")

    checked = 0
    for path in files:
        checked += 1
        try:
            with uproot.open(str(path), num_workers=1) as handle:
                tree, resolved = _resolve_tree(
                    handle, preferred_tree_path, fallback_tree_path
                )
                entries = int(tree.num_entries)
                if entries <= 0:
                    continue
                schema = _build_schema(tree, resolved)
                return schema, path.resolve(), checked
        except Exception as exc:
            raise ShuffleError(
                "Failed to inspect reference schema from {}: {}".format(path, exc)
            ) from exc

    raise ShuffleError("All source ROOT files inspected for a reference schema are empty.")


def _inspect_files(
    files: Sequence[Path],
    preferred_tree_path: str,
    fallback_tree_path: str,
    uproot: Any,
) -> Tuple[List[FileInfo], TreeSchema]:
    if not files:
        raise ShuffleError("No ROOT files supplied for inspection.")

    raw_infos: List[FileInfo] = []
    reference_path: Optional[Path] = None
    reference_tree_path: Optional[str] = None

    for path in files:
        try:
            with uproot.open(str(path), num_workers=1) as handle:
                tree, resolved = _resolve_tree(
                    handle, preferred_tree_path, fallback_tree_path
                )
                entries = int(tree.num_entries)
        except Exception as exc:
            raise ShuffleError("Failed to inspect {}: {}".format(path, exc)) from exc
        raw_infos.append(FileInfo(str(path.resolve()), entries, resolved))
        if entries > 0 and reference_path is None:
            reference_path = path
            reference_tree_path = resolved

    if reference_path is None or reference_tree_path is None:
        raise ShuffleError("All source ROOT files are empty.")

    with uproot.open(str(reference_path), num_workers=1) as handle:
        reference_tree, _ = _resolve_tree(
            handle, preferred_tree_path, fallback_tree_path
        )
        schema = _build_schema(reference_tree, reference_tree_path)

    reference_signature = _reference_metadata_signature(schema)
    for info in raw_infos:
        try:
            with uproot.open(info.path, num_workers=1) as handle:
                tree, _ = _resolve_tree(
                    handle, preferred_tree_path, fallback_tree_path
                )
                signature = _metadata_signature(tree, schema)
        except Exception as exc:
            raise ShuffleError(
                "Schema validation failed for {}: {}".format(info.path, exc)
            ) from exc
        if signature != reference_signature:
            mismatches = []
            for expected, actual in zip(reference_signature, signature):
                if expected != actual:
                    mismatches.append("expected={!r}, actual={!r}".format(expected, actual))
                    if len(mismatches) >= 10:
                        break
            raise ShuffleError(
                "TTree schema differs in {}:\n  {}".format(
                    info.path, "\n  ".join(mismatches)
                )
            )

    return raw_infos, schema



def _writer_layout(
    schema: TreeSchema,
) -> Tuple[OrderedDict, Dict[str, str], Dict[str, Tuple[str, ...]]]:
    """Build Uproot branch types while consolidating shared jagged counters.

    Uproot's writer represents several jagged fields sharing one leaf-count branch
    most safely as one synthetic ``var * record``. The synthetic record is not a
    physical output branch: ``field_name`` flattens its fields back to the original
    DeepNTuplizer branch names.
    """
    branch_types: OrderedDict = OrderedDict()
    counter_by_outer: Dict[str, str] = {}
    grouped_fields: Dict[str, Tuple[str, ...]] = {}
    jagged_groups = schema.jagged_by_count
    emitted_counts = set()
    spec_by_name = {spec.name: spec for spec in schema.branches}

    for spec in schema.branches:
        base_dtype = np.dtype(spec.dtype)
        if spec.kind == "scalar":
            branch_types[spec.name] = base_dtype
        elif spec.kind == "fixed":
            branch_types[spec.name] = np.dtype((base_dtype, tuple(spec.fixed_shape)))
        elif spec.kind == "jagged":
            assert spec.count_name is not None
            names = tuple(jagged_groups[spec.count_name])
            if spec.count_name in emitted_counts:
                continue
            emitted_counts.add(spec.count_name)
            if len(names) == 1:
                outer_name = names[0]
                branch_types[outer_name] = "var * {}".format(base_dtype.name)
            else:
                group_index = len(grouped_fields)
                outer_name = "__cms_jagged_group_{:04d}".format(group_index)
                while outer_name in schema.all_output_names or outer_name in branch_types:
                    group_index += 1
                    outer_name = "__cms_jagged_group_{:04d}".format(group_index)
                for name in names:
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        raise ShuffleError(
                            "Jagged branch {!r} cannot be represented as an Awkward "
                            "record field without renaming.".format(name)
                        )
                fields = ", ".join(
                    "{}: {}".format(name, np.dtype(spec_by_name[name].dtype).name)
                    for name in names
                )
                branch_types[outer_name] = "var * {{{}}}".format(fields)
                grouped_fields[outer_name] = names
            counter_by_outer[outer_name] = spec.count_name
        else:
            raise ShuffleError("Unknown branch kind {!r}.".format(spec.kind))
    return branch_types, counter_by_outer, grouped_fields



def _prepare_write_payload(
    arrays: Mapping[str, Any], schema: TreeSchema, awkward: Any
) -> Dict[str, Any]:
    n_events = _array_length(arrays)
    payload: Dict[str, Any] = {}
    _, _, grouped_fields = _writer_layout(schema)
    grouped_names = {name for names in grouped_fields.values() for name in names}

    # Validate every shared leaf-count group before zipping or writing it.
    for count_name, jagged_names in schema.jagged_by_count.items():
        reference_counts = np.asarray(
            awkward.to_numpy(awkward.num(arrays[jagged_names[0]], axis=1)),
            dtype=np.int32,
        )
        if len(reference_counts) != n_events:
            raise ShuffleError(
                "Jagged count length mismatch for {}.".format(jagged_names[0])
            )
        for name in jagged_names[1:]:
            counts = np.asarray(
                awkward.to_numpy(awkward.num(arrays[name], axis=1)),
                dtype=np.int32,
            )
            if not np.array_equal(reference_counts, counts):
                raise ShuffleError(
                    "Jagged branches sharing count branch {!r} are misaligned: "
                    "{} versus {}.".format(count_name, jagged_names[0], name)
                )

    for spec in schema.branches:
        if spec.name in grouped_names:
            continue
        value = arrays[spec.name]
        if spec.kind in ("scalar", "fixed") and isinstance(value, awkward.Array):
            value = awkward.to_numpy(value)
        payload[spec.name] = value

    for outer_name, names in grouped_fields.items():
        payload[outer_name] = awkward.zip(
            {name: arrays[name] for name in names},
            depth_limit=None,
        )

    lengths = {name: len(value) for name, value in payload.items()}
    bad = {name: length for name, length in lengths.items() if length != n_events}
    if bad:
        raise ShuffleError(
            "Output branch event counts are not aligned: expected {}, got {}"
            .format(n_events, bad)
        )
    return payload



def _make_writable_tree(
    writable_file: Any,
    tree_path: str,
    schema: TreeSchema,
    uproot: Any,
    awkward: Any,
) -> Any:
    del uproot
    tree_name = tree_path.strip("/")
    if not tree_name:
        raise ShuffleError("Output TTree path is empty.")
    branch_types, counter_by_outer, grouped_fields = _writer_layout(schema)

    def counter_name(counted: str) -> str:
        return counter_by_outer.get(counted, "n" + counted)

    def field_name(outer: str, inner: str) -> str:
        if outer in grouped_fields:
            return inner
        return inner if outer == "" else outer + "_" + inner

    return writable_file.mktree(
        tree_name,
        branch_types,
        title=schema.tree_title,
        counter_name=counter_name,
        field_name=field_name,
        initial_basket_capacity=16,
        resize_factor=10.0,
    )



class AtomicRootWriter:
    def __init__(
        self,
        final_path: Path,
        schema: TreeSchema,
        output_tree_path: str,
        compression: Any,
        uproot: Any,
        awkward: Any,
        overwrite: bool,
    ) -> None:
        self.final_path = final_path
        self.partial_path = final_path.with_name(final_path.name + ".partial")
        self.schema = schema
        self.output_tree_path = output_tree_path
        self.uproot = uproot
        self.awkward = awkward
        self.events_written = 0
        self._finalized = False

        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            if not overwrite:
                raise ShuffleError("Output file already exists: {}".format(final_path))
            final_path.unlink()
        if self.partial_path.exists():
            self.partial_path.unlink()

        self.file = uproot.recreate(str(self.partial_path), compression=compression)
        self.tree = _make_writable_tree(
            self.file, output_tree_path, schema, uproot, awkward
        )

    def extend(self, arrays: Mapping[str, Any]) -> None:
        n_events = _array_length(arrays)
        if n_events <= 0:
            return
        self.tree.extend(_prepare_write_payload(arrays, self.schema, self.awkward))
        self.events_written += n_events

    def finalize(self) -> None:
        if self._finalized:
            return
        self.file.close()
        os.replace(str(self.partial_path), str(self.final_path))
        self._finalized = True

    def abort(self) -> None:
        if self._finalized:
            return
        try:
            self.file.close()
        except Exception:
            pass
        try:
            if self.partial_path.exists():
                self.partial_path.unlink()
        except Exception:
            pass


def _run_modern_io_self_test(uproot: Any, awkward: Any) -> None:
    """Exercise the exact Uproot/Awkward writer layout used by this script.

    In particular, this verifies that several jagged branches can share one
    regenerated ROOT leaf-count branch through a synthetic ``var * record``
    layout, and that repeated ``extend`` calls remain readable after close.
    """
    schema = TreeSchema(
        tree_path="deepntuplizerAK8/tree",
        tree_title="cms-shuffle-modern-io-self-test",
        branches=[
            BranchSpec("event_id", "scalar", np.dtype(np.int64).str),
            BranchSpec(
                "fixed_pair",
                "fixed",
                np.dtype(np.float32).str,
                fixed_shape=(2,),
            ),
            BranchSpec(
                "shared_pt",
                "jagged",
                np.dtype(np.float32).str,
                count_name="nshared",
            ),
            BranchSpec(
                "shared_code",
                "jagged",
                np.dtype(np.int32).str,
                count_name="nshared",
            ),
            BranchSpec(
                "single_flag",
                "jagged",
                np.dtype(np.uint8).str,
                count_name="nsingle",
            ),
        ],
    )

    def make_batch(offset: int, rows: Sequence[Sequence[float]]) -> Dict[str, Any]:
        count = len(rows)
        shared_pt = awkward.values_astype(awkward.Array(rows), np.float32)
        shared_code = awkward.values_astype(
            awkward.Array(
                [
                    [int(round(value * 10.0)) + offset for value in row]
                    for row in rows
                ]
            ),
            np.int32,
        )
        single_flag = awkward.values_astype(
            awkward.Array(
                [
                    [int((offset + row_index + item_index) % 2) for item_index in range(len(row) + 1)]
                    for row_index, row in enumerate(rows)
                ]
            ),
            np.uint8,
        )
        return {
            "event_id": np.arange(offset, offset + count, dtype=np.int64),
            "fixed_pair": np.column_stack(
                (
                    np.arange(offset, offset + count, dtype=np.float32),
                    -np.arange(offset, offset + count, dtype=np.float32),
                )
            ),
            "shared_pt": shared_pt,
            "shared_code": shared_code,
            "single_flag": single_flag,
        }

    first = make_batch(0, [[1.0, 2.0], [], [3.0]])
    second = make_batch(3, [[4.0], [5.0, 6.0, 7.0]])
    expected = _concatenate_arrays([first, second], schema.data_names, awkward)

    try:
        with tempfile.TemporaryDirectory(prefix="cms_shuffle_uproot5_selftest_") as tmp:
            path = Path(tmp) / "roundtrip.root"
            writer = AtomicRootWriter(
                path,
                schema,
                schema.tree_path,
                uproot.ZLIB(1),
                uproot,
                awkward,
                overwrite=True,
            )
            try:
                writer.extend(first)
                writer.extend(second)
                writer.finalize()
            except BaseException:
                writer.abort()
                raise

            with uproot.open(str(path), num_workers=1) as handle:
                if schema.tree_path not in handle:
                    raise ShuffleError(
                        "modern I/O self-test did not create nested TTree {!r}"
                        .format(schema.tree_path)
                    )
                tree = handle[schema.tree_path]
                actual_names = set(_tree_branch_names(tree))
                expected_names = set(schema.all_output_names)
                if actual_names != expected_names:
                    raise ShuffleError(
                        "modern I/O self-test branch mismatch: missing={}, extra={}"
                        .format(
                            sorted(expected_names - actual_names),
                            sorted(actual_names - expected_names),
                        )
                    )
                if int(tree.num_entries) != 5:
                    raise ShuffleError(
                        "modern I/O self-test wrote {} entries, expected 5"
                        .format(int(tree.num_entries))
                    )
                decoded = tree.arrays(
                    expressions=schema.data_names + schema.count_names,
                    library="ak",
                    how=dict,
                )

            if not np.array_equal(
                awkward.to_numpy(decoded["event_id"]),
                np.asarray(expected["event_id"]),
            ):
                raise ShuffleError("modern I/O self-test scalar data mismatch")
            if not np.array_equal(
                awkward.to_numpy(decoded["fixed_pair"]),
                np.asarray(expected["fixed_pair"]),
            ):
                raise ShuffleError("modern I/O self-test fixed-array mismatch")
            for name in ("shared_pt", "shared_code", "single_flag"):
                if awkward.to_list(decoded[name]) != awkward.to_list(expected[name]):
                    raise ShuffleError(
                        "modern I/O self-test jagged data mismatch for {}"
                        .format(name)
                    )
            expected_shared_counts = awkward.to_numpy(
                awkward.num(expected["shared_pt"], axis=1)
            ).astype(np.int32, copy=False)
            expected_single_counts = awkward.to_numpy(
                awkward.num(expected["single_flag"], axis=1)
            ).astype(np.int32, copy=False)
            if not np.array_equal(
                awkward.to_numpy(decoded["nshared"]), expected_shared_counts
            ):
                raise ShuffleError(
                    "modern I/O self-test shared counter mismatch"
                )
            if not np.array_equal(
                awkward.to_numpy(decoded["nsingle"]), expected_single_counts
            ):
                raise ShuffleError(
                    "modern I/O self-test single counter mismatch"
                )
    except BaseException as exc:
        if isinstance(exc, ShuffleError):
            raise
        raise ShuffleError(
            "Modern Uproot/Awkward ROOT round-trip self-test failed: {}: {}"
            .format(type(exc).__name__, exc)
        ) from exc


def _check_open_file_budget(num_buckets: int, requested_limit: int) -> None:
    if requested_limit > 0:
        safe_limit = requested_limit
    else:
        safe_limit = 900
        try:
            import resource

            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft != resource.RLIM_INFINITY:
                safe_limit = max(1, int(soft) - 64)
        except Exception:
            pass
    if num_buckets > safe_limit:
        raise ShuffleError(
            "Phase 1 needs up to {} simultaneously open temporary bucket files, "
            "but the safe limit is {}. Raise 'ulimit -n', use fewer "
            "--num-temp-buckets, or explicitly adjust --max-open-temp-files "
            "after checking the system limit."
            .format(num_buckets, safe_limit)
        )


def _log_progress(label: str, done: int, total: int, started: float) -> None:
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed
    fraction = done / total if total else 1.0
    print(
        "{}: {:,}/{:,} events ({:.1%}), {:,.0f} events/s, {:.1f} min".format(
            label, done, total, fraction, rate, elapsed / 60.0
        ),
        flush=True,
    )


def _log_stream_progress(
    label: str,
    done: int,
    files_done: int,
    total_files: int,
    started: float,
) -> None:
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed
    print(
        "{}: {:,} events; files {}/{}; {:,.0f} events/s; {:.1f} min".format(
            label, done, files_done, total_files, rate, elapsed / 60.0
        ),
        flush=True,
    )



def _part_path(
    temp_jet_dir: Path,
    bucket_id: int,
    worker_id: int,
    num_buckets: int,
    num_workers: int,
    *,
    sorted_run: bool,
) -> Path:
    bucket_width = max(4, len(str(max(0, num_buckets - 1))))
    worker_width = max(2, len(str(max(0, num_workers - 1))))
    suffix = "_sorted" if sorted_run else ""
    return temp_jet_dir / (
        "bucket_{:0{bw}d}_part_{:0{ww}d}{}.root".format(
            bucket_id,
            worker_id,
            suffix,
            bw=bucket_width,
            ww=worker_width,
        )
    )


def _rolled_part_path(
    temp_jet_dir: Path,
    bucket_id: int,
    worker_id: int,
    run_index: int,
    num_buckets: int,
    num_workers: int,
) -> Path:
    bucket_width = max(4, len(str(max(0, num_buckets - 1))))
    worker_width = max(2, len(str(max(0, num_workers - 1))))
    run_width = max(4, len(str(max(0, run_index))))
    return temp_jet_dir / (
        "bucket_{:0{bw}d}_worker_{:0{ww}d}_run_{:0{rw}d}.root".format(
            bucket_id,
            worker_id,
            run_index,
            bw=bucket_width,
            ww=worker_width,
            rw=run_width,
        )
    )


def _sorted_run_path(path: Path) -> Path:
    return path.with_name(path.stem + "_sorted.root")


def _temp_run_identity(path: Path) -> Tuple[int, int, int, bool]:
    match = _TEMP_RUN_RE.fullmatch(path.name)
    if match is None:
        raise ShuffleError(
            "Unrecognized temporary ROOT filename: {}".format(path)
        )
    return (
        int(match.group("bucket")),
        int(match.group("worker")),
        int(match.group("run")),
        bool(match.group("sorted")),
    )


def _scan_resume_temp_directory(
    temp_jet_dir: Path,
    num_buckets: int,
    *,
    start_from_phase2a: bool,
    start_from_phase2b: bool,
) -> Tuple[List[List[Path]], List[List[Path]]]:
    """Discover cached runs by filename only; do not decode ROOT data.

    Phase-2A recovery keeps complete ``*_sorted.root`` files, removes only
    incomplete ``*_sorted.root.partial`` files, and schedules remaining plain
    ``*.root`` runs for sorting. A plain run whose exact sorted counterpart
    already exists is a stale duplicate left between atomic rename and unlink;
    it is removed after the complete sorted file has won the identity check.

    Phase-2B recovery is strict: every temporary file must be a complete
    ``*_sorted.root`` run.
    """
    if not temp_jet_dir.is_dir():
        raise ShuffleError(
            "Resume temporary directory does not exist: {}".format(temp_jet_dir)
        )

    partials = sorted(path for path in temp_jet_dir.iterdir() if path.name.endswith(".partial"))
    if start_from_phase2b and partials:
        raise ShuffleError(
            "--start-from-phase2b requires no partial files, found: {}".format(
                [path.name for path in partials[:20]]
            )
        )
    if start_from_phase2a:
        bad_partials = [
            path for path in partials
            if (
                not path.name.endswith("_sorted.root.partial")
                and path.name != PHASE2A_CHECKPOINT_NAME + ".partial"
            )
        ]
        if bad_partials:
            raise ShuffleError(
                "Phase 1 appears incomplete; found non-sorted partial files: {}"
                .format([path.name for path in bad_partials[:20]])
            )
        for path in partials:
            print(
                "  Resume Phase 2A: deleting incomplete sorted partial {}"
                .format(path.name),
                flush=True,
            )
            path.unlink()

    unsorted_by_identity: Dict[Tuple[int, int, int], Path] = {}
    sorted_by_identity: Dict[Tuple[int, int, int], Path] = {}
    checkpoint_path = temp_jet_dir / PHASE2A_CHECKPOINT_NAME
    allowed_non_root = {checkpoint_path.name}
    for path in sorted(temp_jet_dir.iterdir()):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        if path.suffix != ".root":
            if path.name in allowed_non_root:
                continue
            raise ShuffleError(
                "Unexpected file in temporary directory: {}".format(path)
            )
        bucket_id, worker_id, run_id, is_sorted = _temp_run_identity(path)
        if not 0 <= bucket_id < int(num_buckets):
            raise ShuffleError(
                "Temporary run {} has bucket {}, outside [0, {}).".format(
                    path.name, bucket_id, num_buckets
                )
            )
        identity = (bucket_id, worker_id, run_id)
        target = sorted_by_identity if is_sorted else unsorted_by_identity
        if identity in target:
            raise ShuffleError(
                "Duplicate temporary run identity {}: {} and {}".format(
                    identity, target[identity], path
                )
            )
        target[identity] = path

    if not sorted_by_identity and not unsorted_by_identity:
        raise ShuffleError(
            "No temporary ROOT runs found in {}.".format(temp_jet_dir)
        )

    if start_from_phase2b and unsorted_by_identity:
        raise ShuffleError(
            "--start-from-phase2b requires every run to be sorted; found {} "
            "plain ROOT runs, including {}.".format(
                len(unsorted_by_identity),
                [path.name for path in list(unsorted_by_identity.values())[:20]],
            )
        )

    if start_from_phase2a:
        duplicates = sorted(set(unsorted_by_identity) & set(sorted_by_identity))
        for identity in duplicates:
            stale = unsorted_by_identity.pop(identity)
            print(
                "  Resume Phase 2A: complete sorted counterpart exists; "
                "removing stale plain run {}".format(stale.name),
                flush=True,
            )
            stale.unlink()

    unsorted_by_bucket: List[List[Path]] = [[] for _ in range(num_buckets)]
    sorted_by_bucket: List[List[Path]] = [[] for _ in range(num_buckets)]
    for (bucket_id, _worker_id, _run_id), path in unsorted_by_identity.items():
        unsorted_by_bucket[bucket_id].append(path)
    for (bucket_id, _worker_id, _run_id), path in sorted_by_identity.items():
        sorted_by_bucket[bucket_id].append(path)
    for paths in unsorted_by_bucket:
        paths.sort()
    for paths in sorted_by_bucket:
        paths.sort()

    print(
        "Resume scan: {:,} completed sorted runs; {:,} plain runs remaining."
        .format(
            sum(len(paths) for paths in sorted_by_bucket),
            sum(len(paths) for paths in unsorted_by_bucket),
        ),
        flush=True,
    )
    return unsorted_by_bucket, sorted_by_bucket


def _write_phase2a_checkpoint(
    temp_jet_dir: Path,
    sorted_by_bucket: Sequence[Sequence[Path]],
    bucket_counts: Sequence[int],
    num_buckets: int,
    seed: int,
    output_tree_path: str,
) -> None:
    payload = {
        "version": 1,
        "num_temp_buckets": int(num_buckets),
        "seed": int(seed),
        "output_tree_path": str(output_tree_path),
        "bucket_counts": [int(value) for value in bucket_counts],
        "sorted_parts": [
            [path.name for path in paths] for paths in sorted_by_bucket
        ],
    }
    final_path = temp_jet_dir / PHASE2A_CHECKPOINT_NAME
    partial_path = final_path.with_name(final_path.name + ".partial")
    with partial_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(partial_path), str(final_path))


def _load_phase2a_checkpoint(
    temp_jet_dir: Path,
    num_buckets: int,
    seed: int,
    output_tree_path: str,
) -> Optional[Tuple[List[List[Path]], List[int]]]:
    path = temp_jet_dir / PHASE2A_CHECKPOINT_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(
            "  WARNING: ignoring unreadable Phase-2A checkpoint {}: {}"
            .format(path, error),
            file=sys.stderr,
            flush=True,
        )
        return None
    if (
        int(payload.get("num_temp_buckets", -1)) != int(num_buckets)
        or int(payload.get("seed", -1)) != int(seed)
        or str(payload.get("output_tree_path", "")) != str(output_tree_path)
    ):
        print(
            "  WARNING: Phase-2A checkpoint parameters do not match; "
            "falling back to metadata counting.",
            file=sys.stderr,
            flush=True,
        )
        return None
    raw_parts = payload.get("sorted_parts")
    raw_counts = payload.get("bucket_counts")
    if not isinstance(raw_parts, list) or not isinstance(raw_counts, list):
        return None
    if len(raw_parts) != num_buckets or len(raw_counts) != num_buckets:
        return None
    parts: List[List[Path]] = []
    for names in raw_parts:
        bucket_paths = [temp_jet_dir / str(name) for name in names]
        if not all(path.is_file() for path in bucket_paths):
            return None
        parts.append(bucket_paths)
    return parts, [int(value) for value in raw_counts]


def _phase1_worker_main(config: Mapping[str, Any], result_queue: Any) -> None:
    """Read and bucket one disjoint source-file subset into bounded part files."""
    worker_id = int(config["worker_id"])
    prefix = "  [P1 worker {:02d}]".format(worker_id)
    writers: Dict[int, AtomicRootWriter] = {}
    writer_paths: Dict[int, Path] = {}
    try:
        uproot, awkward = _import_modern_io()
        compression = _compression_object(
            uproot,
            str(config["compression_name"]),
            int(config["compression_level"]),
        )
        schema: TreeSchema = config["schema"]
        temp_schema = _schema_with_shuffle_key(schema)
        reference_signature = _reference_metadata_signature(schema)
        assigned_files: Sequence[Tuple[int, str]] = config["assigned_files"]
        num_buckets = int(config["num_buckets"])
        num_workers = int(config["num_workers"])
        temp_jet_dir = Path(str(config["temp_jet_dir"]))
        output_tree_path = str(config["output_tree_path"])
        flush_events = int(config["flush_events"])
        temp_part_events = int(config["temp_part_events"])
        max_pending_events = int(config["max_pending_events"])
        max_pending_bytes = int(config["max_pending_bytes"])
        read_step = config["read_step"]
        seed = int(config["seed"])
        jet_type = str(config["jet_type"])
        progress_every = int(config["progress_every"])
        overwrite = bool(config["overwrite"])

        pending: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        pending_counts = np.zeros(num_buckets, dtype=np.int64)
        pending_bytes = np.zeros(num_buckets, dtype=np.int64)
        bucket_counts = np.zeros(num_buckets, dtype=np.int64)
        next_run_index = np.zeros(num_buckets, dtype=np.int64)
        finalized_paths: List[List[str]] = [[] for _ in range(num_buckets)]
        pending_total_events = 0
        pending_total_bytes = 0
        processed = 0
        nonempty_files = 0
        source_array_bytes = 0
        next_progress = progress_every
        started = time.time()
        validated_first_part = False

        def open_writer(bucket_id: int) -> AtomicRootWriter:
            writer = writers.get(bucket_id)
            if writer is not None:
                return writer
            run_index = int(next_run_index[bucket_id])
            next_run_index[bucket_id] += 1
            path = _rolled_part_path(
                temp_jet_dir,
                bucket_id,
                worker_id,
                run_index,
                num_buckets,
                num_workers,
            )
            writer = AtomicRootWriter(
                path,
                temp_schema,
                output_tree_path,
                compression,
                uproot,
                awkward,
                overwrite,
            )
            writers[bucket_id] = writer
            writer_paths[bucket_id] = path
            return writer

        def close_writer(bucket_id: int) -> None:
            nonlocal validated_first_part
            writer = writers.pop(bucket_id, None)
            path = writer_paths.pop(bucket_id, None)
            if writer is None:
                return
            expected_entries = int(writer.events_written)
            writer.finalize()
            if path is None:
                raise ShuffleError("Missing path for finalized temporary writer.")

            # Decode one real CMS temporary part per Phase-1 process. This catches
            # any source-schema-specific writer incompatibility before Phase 2,
            # without rereading every bounded part.
            if not validated_first_part:
                with uproot.open(str(path), num_workers=1) as check_handle:
                    if output_tree_path not in check_handle:
                        raise ShuffleError(
                            "Temporary part {} is missing TTree {!r}."
                            .format(path, output_tree_path)
                        )
                    check_tree = check_handle[output_tree_path]
                    if int(check_tree.num_entries) != expected_entries:
                        raise ShuffleError(
                            "Temporary part {} has {:,} entries, expected {:,}."
                            .format(
                                path,
                                int(check_tree.num_entries),
                                expected_entries,
                            )
                        )
                    actual_names = set(_tree_branch_names(check_tree))
                    expected_names = set(temp_schema.all_output_names)
                    if actual_names != expected_names:
                        raise ShuffleError(
                            "Temporary part {} branch mismatch. Missing={}, extra={}."
                            .format(
                                path,
                                sorted(expected_names - actual_names),
                                sorted(actual_names - expected_names),
                            )
                        )
                    if expected_entries > 0:
                        check_tree.arrays(
                            expressions=temp_schema.data_names,
                            entry_start=0,
                            entry_stop=min(expected_entries, 2),
                            library="ak",
                            how=dict,
                        )
                validated_first_part = True
                print(
                    "{} validated first modern-Uproot temporary part: {}"
                    .format(prefix, path.name),
                    flush=True,
                )

            finalized_paths[bucket_id].append(str(path))

        def write_arrays_to_parts(bucket_id: int, arrays: Mapping[str, Any]) -> None:
            total = _array_length(arrays)
            cursor = 0
            while cursor < total:
                writer = open_writer(bucket_id)
                capacity = temp_part_events - int(writer.events_written)
                if capacity <= 0:
                    close_writer(bucket_id)
                    continue
                take = min(capacity, total - cursor)
                segment = _slice_arrays(arrays, slice(cursor, cursor + take))
                writer.extend(segment)
                cursor += take
                if int(writer.events_written) >= temp_part_events:
                    close_writer(bucket_id)

        def flush_bucket(bucket_id: int) -> None:
            nonlocal pending_total_events, pending_total_bytes
            fragments = pending.get(bucket_id)
            if not fragments:
                return
            arrays = _concatenate_arrays(
                fragments,
                temp_schema.data_names,
                awkward,
            )
            expected = int(pending_counts[bucket_id])
            actual = _array_length(arrays)
            if actual != expected:
                raise ShuffleError(
                    "Worker {} bucket {} pending mismatch: {} versus {}."
                    .format(worker_id, bucket_id, expected, actual)
                )
            write_arrays_to_parts(bucket_id, arrays)
            bucket_counts[bucket_id] += expected
            pending_total_events -= expected
            pending_total_bytes -= int(pending_bytes[bucket_id])
            pending[bucket_id] = []
            pending_counts[bucket_id] = 0
            pending_bytes[bucket_id] = 0
            del arrays

        def flush_largest_until_safe() -> None:
            nonlocal pending_total_events, pending_total_bytes
            while (
                (max_pending_events > 0 and pending_total_events > max_pending_events)
                or (max_pending_bytes > 0 and pending_total_bytes > max_pending_bytes)
            ):
                bucket_id = int(np.argmax(pending_bytes))
                if pending_counts[bucket_id] <= 0:
                    bucket_id = int(np.argmax(pending_counts))
                if pending_counts[bucket_id] <= 0:
                    break
                flush_bucket(bucket_id)

        for local_file_index, (file_position, path_string) in enumerate(
            assigned_files,
            start=1,
        ):
            path = Path(path_string)
            print(
                "{} source {}/{} (global position {}): opening {} ...".format(
                    prefix,
                    local_file_index,
                    len(assigned_files),
                    file_position,
                    path,
                ),
                flush=True,
            )
            with uproot.open(str(path), num_workers=1) as handle:
                tree, resolved_tree_path = _resolve_tree(
                    handle,
                    str(config["preferred_tree_path"]),
                    str(config["fallback_tree_path"]),
                )
                entries = int(tree.num_entries)
                signature = _metadata_signature(tree, schema)
                if signature != reference_signature:
                    raise ShuffleError(
                        "TTree schema differs in {} (resolved tree {!r}).".format(
                            path, resolved_tree_path
                        )
                    )
                print(
                    "{} {} -> {:,} events".format(prefix, path.name, entries),
                    flush=True,
                )
                if entries <= 0:
                    continue
                nonempty_files += 1
                current_entry = 0
                for chunk in tree.iterate(
                    expressions=schema.data_names,
                    step_size=read_step,
                    library="ak",
                    how=dict,
                ):
                    n_chunk = _array_length(chunk)
                    if n_chunk <= 0:
                        continue
                    local_entries = np.arange(
                        current_entry,
                        current_entry + n_chunk,
                        dtype=np.uint64,
                    )
                    source_ids = (
                        np.uint64(int(file_position) - 1) << np.uint64(32)
                    ) | local_entries
                    shuffle_keys = _splitmix64(
                        source_ids,
                        _stable_seed(seed, jet_type, "event-key"),
                    )
                    current_entry += n_chunk
                    chunk[SHUFFLE_KEY_BRANCH] = shuffle_keys.view(np.int64)
                    bucket_ids = np.remainder(
                        shuffle_keys,
                        np.uint64(num_buckets),
                    ).astype(np.int32, copy=False)
                    order = np.argsort(bucket_ids, kind="mergesort")
                    sorted_ids = bucket_ids[order]
                    boundaries = np.flatnonzero(
                        np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True]
                    )
                    chunk_bytes = _arrays_memory_bytes(chunk)
                    for group_start, group_stop in zip(
                        boundaries[:-1], boundaries[1:]
                    ):
                        bucket_id = int(sorted_ids[group_start])
                        indices = order[group_start:group_stop]
                        fragment = _slice_arrays(chunk, indices)
                        count = int(group_stop - group_start)
                        estimated_bytes = int(
                            round(chunk_bytes * (count / float(n_chunk)))
                        )
                        pending[bucket_id].append(fragment)
                        pending_counts[bucket_id] += count
                        pending_bytes[bucket_id] += estimated_bytes
                        pending_total_events += count
                        pending_total_bytes += estimated_bytes
                        if pending_counts[bucket_id] >= flush_events:
                            flush_bucket(bucket_id)
                    flush_largest_until_safe()

                    processed += n_chunk
                    source_array_bytes += chunk_bytes
                    if progress_every > 0 and processed >= next_progress:
                        elapsed = max(time.time() - started, 1e-9)
                        print(
                            "{} processed {:,} events; {:,.0f} events/s; "
                            "pending {:,} events/{:.1f} GiB; {:.1f} min".format(
                                prefix,
                                processed,
                                processed / elapsed,
                                pending_total_events,
                                pending_total_bytes / float(1024 ** 3),
                                elapsed / 60.0,
                            ),
                            flush=True,
                        )
                        while next_progress <= processed:
                            next_progress += progress_every
                if current_entry != entries:
                    raise ShuffleError(
                        "Sequential iteration in {} produced {:,} entries, tree "
                        "reports {:,}.".format(path, current_entry, entries)
                    )

        print("{} flushing remaining bucket fragments...".format(prefix), flush=True)
        for bucket_id in range(num_buckets):
            flush_bucket(bucket_id)
        for bucket_id in list(writers):
            close_writer(bucket_id)

        if int(bucket_counts.sum()) != processed:
            raise ShuffleError(
                "Worker {} wrote {:,} events to parts, expected {:,}.".format(
                    worker_id, int(bucket_counts.sum()), processed
                )
            )

        elapsed = max(time.time() - started, 1e-9)
        print(
            "{} complete: {:,} events in {:.1f} min ({:,.0f} events/s); "
            "{} bounded part files.".format(
                prefix,
                processed,
                elapsed / 60.0,
                processed / elapsed,
                sum(len(paths) for paths in finalized_paths),
            ),
            flush=True,
        )
        result_queue.put(
            {
                "ok": True,
                "worker_id": worker_id,
                "processed": processed,
                "nonempty_files": nonempty_files,
                "source_array_bytes": source_array_bytes,
                "bucket_counts": [int(value) for value in bucket_counts],
                "part_paths": finalized_paths,
            }
        )
    except BaseException as exc:
        for writer in writers.values():
            try:
                writer.abort()
            except Exception:
                pass
        traceback.print_exc()
        result_queue.put(
            {
                "ok": False,
                "worker_id": worker_id,
                "error": "{}: {}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
        )

def _phase1_make_temp_bucket_parts(
    jet_type: str,
    files: Sequence[Path],
    schema: TreeSchema,
    preferred_tree_path: str,
    fallback_tree_path: str,
    output_tree_path: str,
    temp_jet_dir: Path,
    num_buckets: int,
    read_step: Union[int, str],
    flush_events: int,
    temp_part_events: int,
    max_pending_events: int,
    max_pending_bytes: int,
    phase1_processes: int,
    phase1_start_method: str,
    seed: int,
    compression_name: str,
    compression_level: int,
    overwrite: bool,
    progress_every: int,
) -> Tuple[List[List[Path]], List[int], int, int, float]:
    """Run independent Phase-1 reader/bucketer/writer processes."""
    file_rng = np.random.RandomState(_stable_seed(seed, jet_type, "file-order"))
    order = file_rng.permutation(len(files))
    ordered_files = [files[int(index)] for index in order]
    num_workers = max(1, min(int(phase1_processes), len(ordered_files)))
    assignments: List[List[Tuple[int, str]]] = [[] for _ in range(num_workers)]
    for global_position, path in enumerate(ordered_files, start=1):
        assignments[(global_position - 1) % num_workers].append(
            (global_position, str(path))
        )

    per_worker_events = max(
        int(flush_events),
        int(math.ceil(float(max_pending_events) / num_workers)),
    )
    per_worker_bytes = max(
        1,
        int(math.ceil(float(max_pending_bytes) / num_workers)),
    )
    print(
        "Phase 1 process layout: {} workers; per-worker pending caps {:,} "
        "events/{:.1f} GiB.".format(
            num_workers,
            per_worker_events,
            per_worker_bytes / float(1024 ** 3),
        ),
        flush=True,
    )

    ctx = mp.get_context(phase1_start_method)
    result_queue = ctx.Queue()
    processes = []
    for worker_id in range(num_workers):
        config = {
            "worker_id": worker_id,
            "num_workers": num_workers,
            "assigned_files": assignments[worker_id],
            "jet_type": jet_type,
            "schema": schema,
            "preferred_tree_path": preferred_tree_path,
            "fallback_tree_path": fallback_tree_path,
            "output_tree_path": output_tree_path,
            "temp_jet_dir": str(temp_jet_dir),
            "num_buckets": num_buckets,
            "read_step": read_step,
            "flush_events": flush_events,
            "temp_part_events": temp_part_events,
            "max_pending_events": per_worker_events,
            "max_pending_bytes": per_worker_bytes,
            "seed": seed,
            "compression_name": compression_name,
            "compression_level": compression_level,
            "overwrite": overwrite,
            "progress_every": progress_every,
        }
        process = ctx.Process(
            target=_phase1_worker_main,
            args=(config, result_queue),
            name="cms-shuffle-p1-{:02d}".format(worker_id),
        )
        process.start()
        processes.append(process)

    results: Dict[int, Mapping[str, Any]] = {}
    try:
        while len(results) < num_workers:
            try:
                result = result_queue.get(timeout=1.0)
            except Exception:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise ShuffleError(
                        "A Phase-1 worker exited without a successful result: {}"
                        .format(
                            [(process.name, process.exitcode) for process in failed]
                        )
                    )
                continue
            worker_id = int(result["worker_id"])
            results[worker_id] = result
            if not bool(result.get("ok", False)):
                raise ShuffleError(
                    "Phase-1 worker {} failed: {}\n{}".format(
                        worker_id,
                        result.get("error", "unknown error"),
                        result.get("traceback", ""),
                    )
                )
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    finally:
        for process in processes:
            process.join()

    for process in processes:
        if process.exitcode != 0:
            raise ShuffleError(
                "Phase-1 process {} exited with code {}.".format(
                    process.name, process.exitcode
                )
            )

    bucket_counts = np.zeros(num_buckets, dtype=np.int64)
    parts_by_bucket: List[List[Path]] = [[] for _ in range(num_buckets)]
    total_events = 0
    nonempty_files = 0
    source_array_bytes = 0
    for worker_id in range(num_workers):
        result = results[worker_id]
        worker_counts = np.asarray(result["bucket_counts"], dtype=np.int64)
        bucket_counts += worker_counts
        total_events += int(result["processed"])
        nonempty_files += int(result["nonempty_files"])
        source_array_bytes += int(result["source_array_bytes"])
        for bucket_id, path_strings in enumerate(result["part_paths"]):
            for path_string in path_strings:
                path = Path(str(path_string))
                if not path.is_file():
                    raise ShuffleError(
                        "Phase-1 part is missing: {}".format(path)
                    )
                parts_by_bucket[bucket_id].append(path)

    if int(bucket_counts.sum()) != total_events:
        raise ShuffleError(
            "Phase-1 aggregate bucket count {:,} differs from processed {:,}."
            .format(int(bucket_counts.sum()), total_events)
        )
    average_source_bytes = source_array_bytes / float(max(total_events, 1))
    return (
        parts_by_bucket,
        [int(value) for value in bucket_counts],
        total_events,
        nonempty_files,
        average_source_bytes,
    )



def _phase2_sort_worker_main(config: Mapping[str, Any], result_queue: Any) -> None:
    """Keep completed sorted runs and sort remaining bounded runs."""
    worker_id = int(config["worker_id"])
    prefix = "  [P2 sort worker {:02d}]".format(worker_id)
    try:
        uproot, awkward = _import_modern_io()
        schema: TreeSchema = config["schema"]
        temp_schema = _schema_with_shuffle_key(schema)
        output_tree_path = str(config["output_tree_path"])
        compression = _compression_object(
            uproot,
            str(config["compression_name"]),
            int(config["compression_level"]),
        )
        bucket_ids = [int(value) for value in config["bucket_ids"]]
        parts_by_bucket = {
            int(bucket_id): [Path(str(path)) for path in paths]
            for bucket_id, paths in config["parts_by_bucket"].items()
        }
        existing_sorted_by_bucket = {
            int(bucket_id): [Path(str(path)) for path in paths]
            for bucket_id, paths in config.get(
                "existing_sorted_by_bucket", {}
            ).items()
        }
        expected_counts = {
            int(bucket_id): int(value)
            for bucket_id, value in config.get("expected_counts", {}).items()
        }
        overwrite = bool(config["overwrite"])
        sorted_paths: Dict[int, List[str]] = {}
        sorted_counts: Dict[int, int] = {}
        started = time.time()

        for position, bucket_id in enumerate(bucket_ids, start=1):
            source_paths = parts_by_bucket.get(bucket_id, [])
            existing_paths = existing_sorted_by_bucket.get(bucket_id, [])
            expected_bucket = expected_counts.get(bucket_id)
            expected_text = (
                "unknown" if expected_bucket is None
                else "{:,}".format(expected_bucket)
            )
            print(
                "{} bucket {}/{} (logical {}): {} completed sorted runs, "
                "{} plain runs; expected events {}.".format(
                    prefix,
                    position,
                    len(bucket_ids),
                    bucket_id,
                    len(existing_paths),
                    len(source_paths),
                    expected_text,
                ),
                flush=True,
            )
            bucket_sorted_paths: List[str] = []
            bucket_total = 0

            # Metadata only: num_entries is required to establish bucket and
            # final-shard boundaries. No event branch is decoded here.
            for existing_part in existing_paths:
                with uproot.open(str(existing_part), num_workers=1) as handle:
                    if output_tree_path not in handle:
                        raise ShuffleError(
                            "Completed sorted run {} is missing TTree {!r}."
                            .format(existing_part, output_tree_path)
                        )
                    bucket_total += int(handle[output_tree_path].num_entries)
                bucket_sorted_paths.append(str(existing_part))

            for part_position, source_part in enumerate(source_paths, start=1):
                destination_part = _sorted_run_path(source_part)
                print(
                    "{}   sorting remaining part {}/{}: {}".format(
                        prefix,
                        part_position,
                        len(source_paths),
                        source_part.name,
                    ),
                    flush=True,
                )
                arrays = _read_entire_tree(
                    source_part,
                    output_tree_path,
                    temp_schema,
                    uproot,
                )
                part_count = _array_length(arrays)
                key_signed = np.asarray(
                    arrays[SHUFFLE_KEY_BRANCH], dtype=np.int64
                )
                permutation = np.argsort(
                    key_signed.view(np.uint64), kind="mergesort"
                )
                _permute_arrays_branchwise_in_place(
                    arrays,
                    permutation,
                    temp_schema.data_names,
                )
                writer = AtomicRootWriter(
                    destination_part,
                    temp_schema,
                    output_tree_path,
                    compression,
                    uproot,
                    awkward,
                    overwrite,
                )
                try:
                    writer.extend(arrays)
                    writer.finalize()
                except BaseException:
                    writer.abort()
                    raise
                source_part.unlink()
                bucket_sorted_paths.append(str(destination_part))
                bucket_total += part_count
                del arrays, key_signed, permutation

            bucket_sorted_paths.sort()
            if expected_bucket is not None and bucket_total != expected_bucket:
                raise ShuffleError(
                    "Sorted bucket {} contains {:,} events, expected {:,}."
                    .format(bucket_id, bucket_total, expected_bucket)
                )
            sorted_paths[bucket_id] = bucket_sorted_paths
            sorted_counts[bucket_id] = bucket_total

        elapsed = max(time.time() - started, 1e-9)
        print(
            "{} complete: {} logical buckets in {:.1f} min.".format(
                prefix, len(bucket_ids), elapsed / 60.0
            ),
            flush=True,
        )
        result_queue.put(
            {
                "ok": True,
                "worker_id": worker_id,
                "sorted_paths": sorted_paths,
                "sorted_counts": sorted_counts,
            }
        )
    except BaseException as error:
        traceback.print_exc()
        result_queue.put(
            {
                "ok": False,
                "worker_id": worker_id,
                "error": "{}: {}".format(type(error).__name__, error),
                "traceback": traceback.format_exc(),
            }
        )


def _phase2_sort_bucket_parts_multiprocess(
    parts_by_bucket: Sequence[Sequence[Path]],
    bucket_counts: Optional[Sequence[int]],
    schema: TreeSchema,
    output_tree_path: str,
    compression_name: str,
    compression_level: int,
    overwrite: bool,
    phase2_processes: int,
    phase2_start_method: str,
    existing_sorted_by_bucket: Optional[Sequence[Sequence[Path]]] = None,
) -> Tuple[List[List[Path]], List[int]]:
    """Parallel Phase 2A, resumable at individual bounded-run granularity."""
    if existing_sorted_by_bucket is None:
        existing_sorted_by_bucket = [[] for _ in parts_by_bucket]
    if len(existing_sorted_by_bucket) != len(parts_by_bucket):
        raise ShuffleError(
            "Existing-sorted and plain-run bucket lists have different lengths."
        )
    if bucket_counts is not None and len(bucket_counts) != len(parts_by_bucket):
        raise ShuffleError("Bucket-count and run-list lengths differ.")

    bucket_ids = [
        bucket_id
        for bucket_id in range(len(parts_by_bucket))
        if (
            len(parts_by_bucket[bucket_id]) > 0
            or len(existing_sorted_by_bucket[bucket_id]) > 0
            or (bucket_counts is not None and int(bucket_counts[bucket_id]) > 0)
        )
    ]
    if not bucket_ids:
        return ([[] for _ in parts_by_bucket], [0 for _ in parts_by_bucket])

    num_workers = max(1, min(int(phase2_processes), len(bucket_ids)))
    assignments: List[List[int]] = [[] for _ in range(num_workers)]
    base = len(bucket_ids) // num_workers
    remainder = len(bucket_ids) % num_workers
    cursor = 0
    for worker_id in range(num_workers):
        take = base + int(worker_id < remainder)
        assignments[worker_id] = bucket_ids[cursor:cursor + take]
        cursor += take

    remaining_parts = sum(len(paths) for paths in parts_by_bucket)
    completed_parts = sum(len(paths) for paths in existing_sorted_by_bucket)
    print(
        "Phase 2A process layout: {} workers, {} logical buckets; "
        "{:,} completed sorted runs retained and {:,} plain runs scheduled."
        .format(
            num_workers, len(bucket_ids), completed_parts, remaining_parts
        ),
        flush=True,
    )
    ctx = mp.get_context(phase2_start_method)
    result_queue = ctx.Queue()
    processes = []
    for worker_id, assigned_bucket_ids in enumerate(assignments):
        expected = {}
        if bucket_counts is not None:
            expected = {
                int(bucket_id): int(bucket_counts[bucket_id])
                for bucket_id in assigned_bucket_ids
            }
        config = {
            "worker_id": worker_id,
            "bucket_ids": assigned_bucket_ids,
            "parts_by_bucket": {
                int(bucket_id): [str(path) for path in parts_by_bucket[bucket_id]]
                for bucket_id in assigned_bucket_ids
            },
            "existing_sorted_by_bucket": {
                int(bucket_id): [
                    str(path) for path in existing_sorted_by_bucket[bucket_id]
                ]
                for bucket_id in assigned_bucket_ids
            },
            "expected_counts": expected,
            "schema": schema,
            "output_tree_path": output_tree_path,
            "compression_name": compression_name,
            "compression_level": int(compression_level),
            "overwrite": bool(overwrite),
        }
        process = ctx.Process(
            target=_phase2_sort_worker_main,
            args=(config, result_queue),
            name="cms-shuffle-p2sort-{:02d}".format(worker_id),
        )
        process.start()
        processes.append(process)

    results: Dict[int, Mapping[str, Any]] = {}
    try:
        while len(results) < num_workers:
            try:
                result = result_queue.get(timeout=1.0)
            except Exception:
                failed = [
                    process for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise ShuffleError(
                        "A Phase-2 sorting worker exited without a result: {}"
                        .format([(p.name, p.exitcode) for p in failed])
                    )
                continue
            worker_id = int(result["worker_id"])
            results[worker_id] = result
            if not bool(result.get("ok", False)):
                raise ShuffleError(
                    "Phase-2 sorting worker {} failed: {}\n{}".format(
                        worker_id,
                        result.get("error", "unknown error"),
                        result.get("traceback", ""),
                    )
                )
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    finally:
        for process in processes:
            process.join()

    for process in processes:
        if process.exitcode != 0:
            raise ShuffleError(
                "Phase-2 sorting process {} exited with code {}.".format(
                    process.name, process.exitcode
                )
            )

    sorted_by_bucket: List[List[Path]] = [[] for _ in parts_by_bucket]
    sorted_counts: List[int] = [0 for _ in parts_by_bucket]
    for worker_id in range(num_workers):
        result = results[worker_id]
        for bucket_id_raw, path_strings in result["sorted_paths"].items():
            bucket_id = int(bucket_id_raw)
            sorted_by_bucket[bucket_id] = [
                Path(str(path)) for path in path_strings
            ]
            sorted_counts[bucket_id] = int(
                result["sorted_counts"][bucket_id_raw]
            )
            if (
                bucket_counts is not None
                and sorted_counts[bucket_id] != int(bucket_counts[bucket_id])
            ):
                raise ShuffleError(
                    "Phase-2 sorted count mismatch for bucket {}.".format(
                        bucket_id
                    )
                )
    return sorted_by_bucket, sorted_counts



def _read_entire_tree(
    path: Path,
    tree_path: str,
    schema: TreeSchema,
    uproot: Any,
    attempts: int = 3,
) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            with uproot.open(str(path), num_workers=1) as handle:
                tree = handle[tree_path]
                return tree.arrays(
                    expressions=schema.data_names,
                    entry_start=0,
                    entry_stop=int(tree.num_entries),
                    library="ak",
                    how=dict,
                )
        except BaseException as error:
            last_error = error
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    raise ShuffleError(
        "Failed to read temporary ROOT file {} after {} attempts: {}: {}"
        .format(path, attempts, type(last_error).__name__, last_error)
    ) from last_error



def _tree_arrays_range(
    tree: Any,
    branch_names: Sequence[str],
    start: int,
    stop: int,
) -> Dict[str, Any]:
    if stop <= start:
        raise ValueError("Cannot read an empty TTree range.")
    return tree.arrays(
        expressions=list(branch_names),
        entry_start=int(start),
        entry_stop=int(stop),
        library="ak",
        how=dict,
    )


def _materialize_merged_range(
    trees: Sequence[Any],
    ordered_part_ids: np.ndarray,
    ordered_local_indices: np.ndarray,
    start: int,
    stop: int,
    schema: TreeSchema,
    awkward: Any,
) -> Dict[str, Any]:
    """Read one exact globally merged range using contiguous per-part reads."""
    part_ids = np.asarray(ordered_part_ids[start:stop], dtype=np.int32)
    local_indices = np.asarray(ordered_local_indices[start:stop], dtype=np.int64)
    if len(part_ids) == 0:
        raise ValueError("Cannot materialize an empty merged range.")
    unique_parts = np.unique(part_ids)
    fragments: List[Mapping[str, Any]] = []
    base_offsets = np.full(len(trees), -1, dtype=np.int64)
    local_starts = np.full(len(trees), -1, dtype=np.int64)
    concatenated_count = 0
    for part_id in unique_parts:
        mask = part_ids == part_id
        selected = local_indices[mask]
        range_start = int(selected[0])
        range_stop = int(selected[-1]) + 1
        if len(selected) != range_stop - range_start or (
            len(selected) > 1 and not np.all(np.diff(selected) == 1)
        ):
            raise ShuffleError(
                "Merged slice selected a non-contiguous range from sorted part {}."
                .format(int(part_id))
            )
        arrays = _tree_arrays_range(
            trees[int(part_id)],
            schema.data_names,
            range_start,
            range_stop,
        )
        base_offsets[int(part_id)] = concatenated_count
        local_starts[int(part_id)] = range_start
        concatenated_count += len(selected)
        fragments.append(arrays)

    concatenated = _concatenate_arrays(
        fragments,
        schema.data_names,
        awkward,
    )
    gather = (
        base_offsets[part_ids]
        + local_indices
        - local_starts[part_ids]
    ).astype(np.int64, copy=False)
    output = _slice_arrays(concatenated, gather)
    del concatenated, gather
    return output


def _prepare_logical_bucket_merge(
    part_paths: Sequence[Path],
    expected_count: int,
    output_tree_path: str,
    uproot: Any,
) -> Dict[str, Any]:
    """Open one logical bucket and build its exact sorted-run merge plan."""
    if expected_count <= 0:
        raise ValueError("Cannot prepare an empty logical bucket.")
    if not part_paths:
        raise ShuffleError("A non-empty logical bucket has no sorted part files.")

    stack = ExitStack()
    try:
        handles = [
            stack.enter_context(uproot.open(str(path), num_workers=1))
            for path in part_paths
        ]
        trees = [handle[output_tree_path] for handle in handles]
        part_counts = [int(tree.num_entries) for tree in trees]
        if sum(part_counts) != int(expected_count):
            raise ShuffleError(
                "Logical bucket part counts sum to {:,}, expected {:,}.".format(
                    sum(part_counts), int(expected_count)
                )
            )

        key_arrays: List[np.ndarray] = []
        part_id_arrays: List[np.ndarray] = []
        local_index_arrays: List[np.ndarray] = []
        for part_id, (tree, part_count, part_path) in enumerate(
            zip(trees, part_counts, part_paths)
        ):
            key_payload = _tree_arrays_range(
                tree,
                [SHUFFLE_KEY_BRANCH],
                0,
                part_count,
            )
            keys = np.asarray(
                key_payload[SHUFFLE_KEY_BRANCH],
                dtype=np.int64,
            ).view(np.uint64)
            if len(keys) > 1 and np.any(keys[1:] < keys[:-1]):
                raise ShuffleError(
                    "Sorted part {} is not monotonic by shuffle key.".format(
                        part_path
                    )
                )
            key_arrays.append(keys)
            part_id_arrays.append(
                np.full(part_count, part_id, dtype=np.int32)
            )
            local_index_arrays.append(
                np.arange(part_count, dtype=np.int64)
            )

        all_keys = np.concatenate(key_arrays)
        all_part_ids = np.concatenate(part_id_arrays)
        all_local_indices = np.concatenate(local_index_arrays)
        merge_order = np.argsort(all_keys, kind="mergesort")
        sorted_keys = all_keys[merge_order]
        if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ShuffleError(
                "Duplicate deterministic shuffle keys found in a logical bucket."
            )
        ordered_part_ids = all_part_ids[merge_order]
        ordered_local_indices = all_local_indices[merge_order]
        del (
            key_arrays,
            part_id_arrays,
            local_index_arrays,
            all_keys,
            all_part_ids,
            all_local_indices,
            merge_order,
            sorted_keys,
        )
        return {
            "stack": stack,
            "trees": trees,
            "ordered_part_ids": ordered_part_ids,
            "ordered_local_indices": ordered_local_indices,
            "count": int(expected_count),
        }
    except BaseException:
        stack.close()
        raise


def _close_logical_bucket_merge(state: Optional[Mapping[str, Any]]) -> None:
    if state is None:
        return
    stack = state.get("stack")
    if stack is not None:
        stack.close()


def _scan_existing_final_shards(
    output_dir: Path,
    prefix: str,
    expected_files: int,
    file_width: int,
    events_per_file: int,
    total_events: int,
) -> Dict[int, Tuple[Path, int]]:
    """Trust atomically finalized ROOT shards and remove incomplete partials.

    AtomicRootWriter writes ``*.root.partial`` and only renames it to ``*.root``
    after a successful close. Therefore a matching final ``*.root`` can be
    reused without decoding it, while any matching partial is safe to discard.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_pattern = re.compile(
        r"^{}_(\d+)\.root$".format(re.escape(prefix))
    )
    partial_pattern = re.compile(
        r"^{}_(\d+)\.root\.partial$".format(re.escape(prefix))
    )

    removed_partials: List[str] = []
    existing: Dict[int, Tuple[Path, int]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        partial_match = partial_pattern.match(path.name)
        if partial_match is not None:
            path.unlink()
            removed_partials.append(path.name)
            continue

        final_match = final_pattern.match(path.name)
        if final_match is None:
            continue
        one_based = int(final_match.group(1))
        file_index = one_based - 1
        if file_index < 0 or file_index >= expected_files:
            raise ShuffleError(
                "Existing final shard {} is outside the expected range 1-{}."
                .format(path, expected_files)
            )
        expected_name = "{}_{:0{width}d}.root".format(
            prefix,
            one_based,
            width=file_width,
        )
        if path.name != expected_name:
            raise ShuffleError(
                "Existing final shard has an unexpected zero-padding/name: {} "
                "(expected {}).".format(path.name, expected_name)
            )
        expected_count = min(
            events_per_file,
            total_events - file_index * events_per_file,
        )
        existing[file_index] = (path, int(expected_count))

    if removed_partials:
        print(
            "  Phase 2B resume: removed {} incomplete final partial file{}: {}"
            .format(
                len(removed_partials),
                "" if len(removed_partials) == 1 else "s",
                ", ".join(removed_partials[:8])
                + (" ..." if len(removed_partials) > 8 else ""),
            ),
            flush=True,
        )
    return existing


def _phase2_worker_main(
    config: Mapping[str, Any],
    result_queue: Any,
) -> None:
    """Materialize and write an explicit set of missing final shard indices."""
    worker_id = int(config["worker_id"])
    prefix_log = "  [P2 worker {:02d}]".format(worker_id)
    active_state: Optional[Dict[str, Any]] = None
    try:
        uproot, awkward = _import_modern_io()
        schema: TreeSchema = config["schema"]
        output_tree_path = str(config["output_tree_path"])
        output_dir = Path(str(config["output_dir"]))
        prefix = str(config["prefix"])
        events_per_file = int(config["events_per_file"])
        total_events = int(config["total_events"])
        file_indices = sorted(int(value) for value in config["file_indices"])
        file_width = int(config["file_width"])
        progress_every = int(config["progress_every"])
        bucket_counts = np.asarray(config["bucket_counts"], dtype=np.int64)
        bucket_offsets = np.asarray(config["bucket_offsets"], dtype=np.int64)
        parts_by_bucket = [
            [Path(str(path)) for path in paths]
            for paths in config["parts_by_bucket"]
        ]
        compression = _compression_object(
            uproot,
            str(config["compression_name"]),
            int(config["compression_level"]),
        )
        if not file_indices:
            raise ShuffleError(
                "Phase-2 worker {} received no missing output files.".format(
                    worker_id
                )
            )

        worker_total = sum(
            min(events_per_file, total_events - index * events_per_file)
            for index in file_indices
        )
        if len(file_indices) == 1:
            file_description = str(file_indices[0] + 1)
        else:
            file_description = "{}, ... , {}".format(
                file_indices[0] + 1,
                file_indices[-1] + 1,
            )
        print(
            "{} rebuilding {} missing final file{} ({}).".format(
                prefix_log,
                len(file_indices),
                "" if len(file_indices) == 1 else "s",
                file_description,
            ),
            flush=True,
        )

        completed = 0
        next_progress = progress_every
        started = time.time()
        records: List[Tuple[int, str, int]] = []

        for file_index in file_indices:
            file_global_start = file_index * events_per_file
            file_global_stop = min(
                (file_index + 1) * events_per_file,
                total_events,
            )
            current_global = file_global_start
            current_bucket = int(
                np.searchsorted(
                    bucket_offsets,
                    file_global_start,
                    side="right",
                ) - 1
            )
            active_bucket_id: Optional[int] = None
            fragments: List[Mapping[str, Any]] = []
            file_remaining = file_global_stop - file_global_start

            try:
                while file_remaining > 0:
                    while (
                        current_bucket < len(bucket_counts)
                        and int(bucket_counts[current_bucket]) == 0
                    ):
                        current_bucket += 1
                    if current_bucket >= len(bucket_counts):
                        raise ShuffleError(
                            "Phase-2 worker {} exhausted logical buckets while "
                            "rebuilding file {}.".format(
                                worker_id,
                                file_index + 1,
                            )
                        )

                    if active_bucket_id != current_bucket:
                        _close_logical_bucket_merge(active_state)
                        active_state = _prepare_logical_bucket_merge(
                            parts_by_bucket[current_bucket],
                            int(bucket_counts[current_bucket]),
                            output_tree_path,
                            uproot,
                        )
                        active_bucket_id = current_bucket

                    local_start = (
                        current_global - int(bucket_offsets[current_bucket])
                    )
                    if (
                        local_start < 0
                        or local_start >= int(bucket_counts[current_bucket])
                    ):
                        raise ShuffleError(
                            "Invalid local cursor {} for bucket {} while "
                            "rebuilding final file {}.".format(
                                local_start,
                                current_bucket,
                                file_index + 1,
                            )
                        )
                    bucket_remaining = (
                        int(bucket_counts[current_bucket]) - local_start
                    )
                    take = min(file_remaining, bucket_remaining)
                    fragments.append(
                        _materialize_merged_range(
                            active_state["trees"],
                            active_state["ordered_part_ids"],
                            active_state["ordered_local_indices"],
                            local_start,
                            local_start + take,
                            schema,
                            awkward,
                        )
                    )
                    current_global += take
                    file_remaining -= take

                    if take == bucket_remaining:
                        _close_logical_bucket_merge(active_state)
                        active_state = None
                        active_bucket_id = None
                        current_bucket += 1

                if len(fragments) == 1:
                    final_arrays = fragments[0]
                else:
                    final_arrays = _concatenate_arrays(
                        fragments,
                        schema.data_names,
                        awkward,
                    )
                expected_count = file_global_stop - file_global_start
                actual_count = _array_length(final_arrays)
                if actual_count != expected_count:
                    raise ShuffleError(
                        "Phase-2 worker {} materialized {:,} events for file {}, "
                        "expected {:,}.".format(
                            worker_id,
                            actual_count,
                            file_index + 1,
                            expected_count,
                        )
                    )

                path = output_dir / (
                    "{}_{:0{width}d}.root".format(
                        prefix,
                        file_index + 1,
                        width=file_width,
                    )
                )
                writer = AtomicRootWriter(
                    path,
                    schema,
                    output_tree_path,
                    compression,
                    uproot,
                    awkward,
                    False,
                )
                try:
                    writer.extend(final_arrays)
                    writer.finalize()
                except BaseException:
                    writer.abort()
                    raise
                records.append((file_index, str(path), expected_count))
                completed += expected_count
                del fragments, final_arrays
            finally:
                _close_logical_bucket_merge(active_state)
                active_state = None

            if progress_every > 0 and completed >= next_progress:
                _log_progress(
                    "{} written".format(prefix_log),
                    completed,
                    worker_total,
                    started,
                )
                while next_progress <= completed:
                    next_progress += progress_every

        _log_progress(
            "{} complete".format(prefix_log),
            completed,
            worker_total,
            started,
        )
        result_queue.put(
            {
                "ok": True,
                "worker_id": worker_id,
                "records": records,
                "events": completed,
            }
        )
    except BaseException as error:
        _close_logical_bucket_merge(active_state)
        result_queue.put(
            {
                "ok": False,
                "worker_id": worker_id,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


def _phase2_write_final_files_multiprocess(
    prefix: str,
    parts_by_bucket: Sequence[Sequence[Path]],
    bucket_counts: Sequence[int],
    schema: TreeSchema,
    output_tree_path: str,
    output_dir: Path,
    total_events: int,
    events_per_file: int,
    compression_name: str,
    compression_level: int,
    overwrite: bool,
    progress_every: int,
    phase2_processes: int,
    phase2_start_method: str,
    resume_existing: bool = False,
) -> Tuple[List[Path], List[int]]:
    """Write missing exact shuffled shards and retain finalized existing ones."""
    if total_events <= 0:
        raise ShuffleError("Cannot run Phase 2 with zero source events.")
    expected_files = int(math.ceil(float(total_events) / events_per_file))
    file_width = max(2, len(str(expected_files)))
    bucket_offsets = np.zeros(len(bucket_counts) + 1, dtype=np.int64)
    bucket_offsets[1:] = np.cumsum(
        np.asarray(bucket_counts, dtype=np.int64),
        dtype=np.int64,
    )
    if int(bucket_offsets[-1]) != int(total_events):
        raise ShuffleError(
            "Phase-2 bucket total {:,} differs from source {:,}.".format(
                int(bucket_offsets[-1]),
                int(total_events),
            )
        )

    if resume_existing:
        existing = _scan_existing_final_shards(
            output_dir,
            prefix,
            expected_files,
            file_width,
            events_per_file,
            total_events,
        )
    else:
        existing = {}

    missing_indices = [
        index for index in range(expected_files) if index not in existing
    ]
    if existing:
        print(
            "  Phase 2B resume: retaining {:,}/{:,} completed final shards; "
            "{:,} shards remain.".format(
                len(existing),
                expected_files,
                len(missing_indices),
            ),
            flush=True,
        )

    new_records: List[Tuple[int, Path, int]] = []
    newly_completed_events = 0
    if missing_indices:
        num_workers = max(
            1,
            min(int(phase2_processes), len(missing_indices)),
        )
        base_files = len(missing_indices) // num_workers
        remainder = len(missing_indices) % num_workers
        assignments: List[List[int]] = []
        cursor = 0
        for worker_id in range(num_workers):
            count = base_files + int(worker_id < remainder)
            assignments.append(missing_indices[cursor:cursor + count])
            cursor += count

        print(
            "Phase 2 process layout: {} workers rebuild {} missing final "
            "files; completed shards are not opened or rewritten.".format(
                num_workers,
                len(missing_indices),
            ),
            flush=True,
        )
        ctx = mp.get_context(phase2_start_method)
        result_queue = ctx.Queue()
        processes: List[Any] = []
        serialized_parts = [
            [str(path) for path in paths]
            for paths in parts_by_bucket
        ]
        for worker_id, file_indices in enumerate(assignments):
            config = {
                "worker_id": worker_id,
                "file_indices": file_indices,
                "file_width": file_width,
                "prefix": prefix,
                "parts_by_bucket": serialized_parts,
                "bucket_counts": [int(value) for value in bucket_counts],
                "bucket_offsets": [int(value) for value in bucket_offsets],
                "schema": schema,
                "output_tree_path": output_tree_path,
                "output_dir": str(output_dir),
                "total_events": int(total_events),
                "events_per_file": int(events_per_file),
                "compression_name": compression_name,
                "compression_level": int(compression_level),
                "progress_every": int(progress_every),
            }
            process = ctx.Process(
                target=_phase2_worker_main,
                args=(config, result_queue),
                name="cms-shuffle-p2-{:02d}".format(worker_id),
            )
            process.start()
            processes.append(process)

        results: Dict[int, Mapping[str, Any]] = {}
        try:
            while len(results) < num_workers:
                try:
                    result = result_queue.get(timeout=1.0)
                except Exception:
                    failed = [
                        process
                        for process in processes
                        if process.exitcode not in (None, 0)
                    ]
                    if failed:
                        raise ShuffleError(
                            "A Phase-2 worker exited without a successful result: {}"
                            .format(
                                [
                                    (process.name, process.exitcode)
                                    for process in failed
                                ]
                            )
                        )
                    continue
                worker_id = int(result["worker_id"])
                results[worker_id] = result
                if not bool(result.get("ok", False)):
                    raise ShuffleError(
                        "Phase-2 worker {} failed: {}\n{}".format(
                            worker_id,
                            result.get("error", "unknown error"),
                            result.get("traceback", ""),
                        )
                    )
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise
        finally:
            for process in processes:
                process.join()

        for process in processes:
            if process.exitcode != 0:
                raise ShuffleError(
                    "Phase-2 process {} exited with code {}.".format(
                        process.name,
                        process.exitcode,
                    )
                )

        for worker_id in range(num_workers):
            result = results[worker_id]
            newly_completed_events += int(result["events"])
            for file_index, path_string, count in result["records"]:
                new_records.append(
                    (int(file_index), Path(str(path_string)), int(count))
                )
    else:
        print(
            "  Phase 2B resume: all {:,} final shards are already complete; "
            "nothing to rewrite.".format(expected_files),
            flush=True,
        )

    records: List[Tuple[int, Path, int]] = [
        (index, path, count)
        for index, (path, count) in existing.items()
    ]
    records.extend(new_records)
    records.sort(key=lambda item: item[0])

    expected_indices = list(range(expected_files))
    actual_indices = [record[0] for record in records]
    if actual_indices != expected_indices:
        raise ShuffleError(
            "Phase-2 output file indices are incomplete or duplicated."
        )
    output_paths = [record[1] for record in records]
    output_counts = [record[2] for record in records]
    retained_events = sum(count for _, (_, count) in existing.items())
    if (
        retained_events + newly_completed_events != total_events
        or sum(output_counts) != total_events
    ):
        raise ShuffleError(
            "Phase-2 output count {:,}/{:,} differs from source {:,}.".format(
                retained_events + newly_completed_events,
                sum(output_counts),
                total_events,
            )
        )
    if output_counts[:-1] and any(
        count != events_per_file for count in output_counts[:-1]
    ):
        raise ShuffleError("A non-final output file is not filled to capacity.")
    if output_counts and output_counts[-1] > events_per_file:
        raise ShuffleError("The final output file exceeds the event limit.")
    print(
        "  Phase 2 complete: {:,} events -> {} final files; {} retained, {} "
        "written in this run.".format(
            total_events,
            len(output_paths),
            len(existing),
            len(new_records),
        ),
        flush=True,
    )
    return output_paths, output_counts

def _verify_final_files(
    paths: Sequence[Path],
    counts: Sequence[int],
    schema: TreeSchema,
    tree_path: str,
    events_per_file: int,
    uproot: Any,
) -> None:
    expected_names = set(schema.all_output_names)
    verified_total = 0
    for path, expected_count in zip(paths, counts):
        with uproot.open(str(path), num_workers=1) as handle:
            if tree_path not in handle:
                raise ShuffleError(
                    "Verification: missing TTree {!r} in {}.".format(tree_path, path)
                )
            tree = handle[tree_path]
            actual_count = int(tree.num_entries)
            actual_names = set(_tree_branch_names(tree))
            # Force one small decode so branch metadata-only success cannot hide a
            # corrupt first basket.
            if actual_count > 0:
                tree.arrays(
                    expressions=schema.data_names,
                    entry_start=0,
                    entry_stop=min(actual_count, 2),
                    library="ak",
                    how=dict,
                )
        if actual_count != int(expected_count):
            raise ShuffleError(
                "Verification: {} has {:,} events, expected {:,}.".format(
                    path, actual_count, int(expected_count)
                )
            )
        if actual_count > events_per_file:
            raise ShuffleError(
                "Verification: {} exceeds {:,} events.".format(path, events_per_file)
            )
        if actual_names != expected_names:
            raise ShuffleError(
                "Verification: branch mismatch in {}. Missing={}, extra={}.".format(
                    path,
                    sorted(expected_names - actual_names),
                    sorted(actual_names - expected_names),
                )
            )
        verified_total += actual_count
    if verified_total != sum(counts):
        raise ShuffleError("Verification total does not match recorded output total.")


def _schema_to_json(schema: TreeSchema) -> Dict[str, Any]:
    return {
        "tree_path": schema.tree_path,
        "tree_title": schema.tree_title,
        "branches": [asdict(spec) for spec in schema.branches],
        "regenerated_count_branches": schema.count_names,
    }


def process_jet_type(
    jet_dir: Path,
    target_root: Path,
    temp_root: Path,
    prefix: str,
    args: argparse.Namespace,
    uproot: Any,
    awkward: Any,
    temp_compression: Any,
    final_compression: Any,
) -> JetResult:
    jet_started = time.time()
    jet_type = jet_dir.name
    files = sorted(path for path in jet_dir.glob(args.file_glob) if path.is_file())
    if not files:
        raise ShuffleError(
            "No files matching {!r} in {}.".format(args.file_glob, jet_dir)
        )

    output_dir = target_root / jet_type
    temp_jet_dir = temp_root / jet_type
    recovery_mode = bool(
        args.start_from_phase2a or args.start_from_phase2b
    )
    if not args.dry_run:
        # Recovery must preserve both the expensive temporary cache and any
        # fully finalized output shards from a previous Phase-2B attempt.
        if recovery_mode:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not temp_jet_dir.is_dir() or not any(temp_jet_dir.iterdir()):
                raise ShuffleError(
                    "Recovery requested, but temporary directory is missing or "
                    "empty: {}".format(temp_jet_dir)
                )
        else:
            _prepare_clean_directory(output_dir, args.overwrite)
            _prepare_clean_directory(temp_jet_dir, args.overwrite)

    print("\n=== Jet type: {} ===".format(jet_type), flush=True)
    print("Discovered {:,} source ROOT files by glob.".format(len(files)), flush=True)
    print(
        "Opening only the first non-empty file to infer the TTree schema...",
        flush=True,
    )
    schema, reference_path, reference_files_checked = _inspect_reference_schema(
        files,
        args.tree_path,
        args.fallback_tree_path,
        uproot,
    )
    print(
        "Reference schema: {} (found after opening {} file{}).".format(
            reference_path,
            reference_files_checked,
            "" if reference_files_checked == 1 else "s",
        ),
        flush=True,
    )

    if args.num_temp_buckets > 0:
        num_buckets = int(args.num_temp_buckets)
        total_events_before_phase1: Optional[int] = None
        print(
            "Using {:,} fixed temporary buckets; source event totals and all "
            "remaining schemas will be checked during Phase 1.".format(num_buckets),
            flush=True,
        )
    else:
        print(
            "Automatic bucket mode requested; opening every source file once "
            "to count entries and validate schemas...",
            flush=True,
        )
        file_infos, fully_validated_schema = _inspect_files(
            files,
            args.tree_path,
            args.fallback_tree_path,
            uproot,
        )
        if fully_validated_schema.signature() != schema.signature():
            raise ShuffleError(
                "Reference schema changed during full automatic inspection."
            )
        total_events_before_phase1 = sum(info.entries for info in file_infos)
        num_buckets = max(
            1,
            int(
                math.ceil(
                    float(total_events_before_phase1) / args.temp_bucket_events
                )
            ),
        )
        print(
            "Source events: {:,}; automatic temporary buckets: {:,}; "
            "expected mean bucket size: {:,.1f}.".format(
                total_events_before_phase1,
                num_buckets,
                total_events_before_phase1 / float(num_buckets),
            ),
            flush=True,
        )

    output_tree_path = args.output_tree_path
    _check_open_file_budget(num_buckets, args.max_open_temp_files)

    mean_pending = args.max_pending_events / float(num_buckets)
    print(
        "Phase-1 global limits: {:,} pending events and {:.1f} GiB; "
        "nominally {:,.0f} events per bucket; per-bucket flush target {:,}."
        .format(
            args.max_pending_events,
            args.max_pending_gb,
            mean_pending,
            args.temp_flush_events,
        ),
        flush=True,
    )
    if mean_pending < 500:
        print(
            "  WARNING: --max-pending-events is small relative to the number "
            "of buckets; ROOT writes may be fragmented. Increase it or use fewer "
            "temporary buckets.",
            file=sys.stderr,
            flush=True,
        )

    print(
        "Branches copied: {} data + {} regenerated count branches; "
        "temporary files add one internal shuffle-key branch that is removed "
        "from final output; output tree: {}"
        .format(len(schema.branches), len(schema.count_names), output_tree_path),
        flush=True,
    )
    print(
        "Final ROOT files will contain at most {:,} events each.".format(
            args.events_per_file
        ),
        flush=True,
    )

    if args.dry_run:
        return JetResult(
            jet_type=jet_type,
            prefix=prefix,
            source_events=total_events_before_phase1,
            source_files=len(files),
            temporary_buckets=num_buckets,
            temporary_bucket_counts=[],
            output_files=[],
            output_file_counts=[],
            elapsed_seconds=time.time() - jet_started,
        )

    if recovery_mode:
        if args.num_temp_buckets <= 0:
            raise ShuffleError(
                "Recovery requires the original positive --num-temp-buckets value."
            )
        unsorted_by_bucket, existing_sorted_by_bucket = (
            _scan_resume_temp_directory(
                temp_jet_dir,
                num_buckets,
                start_from_phase2a=bool(args.start_from_phase2a),
                start_from_phase2b=bool(args.start_from_phase2b),
            )
        )
        average_source_bytes = 0.0
        nonempty_files = 0

        if args.start_from_phase2b:
            checkpoint = _load_phase2a_checkpoint(
                temp_jet_dir,
                num_buckets,
                args.seed,
                output_tree_path,
            )
            if checkpoint is not None:
                checkpoint_parts, checkpoint_counts = checkpoint
                scanned_names = {
                    path.name
                    for paths in existing_sorted_by_bucket
                    for path in paths
                }
                checkpoint_names = {
                    path.name
                    for paths in checkpoint_parts
                    for path in paths
                }
                if scanned_names == checkpoint_names:
                    parts_by_bucket = checkpoint_parts
                    temp_counts = checkpoint_counts
                    print(
                        "Resume Phase 2B: loaded bucket counts from {}; no "
                        "temporary ROOT metadata pass needed.".format(
                            temp_jet_dir / PHASE2A_CHECKPOINT_NAME
                        ),
                        flush=True,
                    )
                else:
                    checkpoint = None
            if checkpoint is None:
                print(
                    "Resume Phase 2B: no matching checkpoint; reading only "
                    "TTree num_entries metadata in parallel to establish bucket "
                    "and final-shard boundaries. Event branches are not decoded.",
                    flush=True,
                )
                parts_by_bucket, temp_counts = (
                    _phase2_sort_bucket_parts_multiprocess(
                        parts_by_bucket=unsorted_by_bucket,
                        bucket_counts=None,
                        schema=schema,
                        output_tree_path=output_tree_path,
                        compression_name=args.temp_compression,
                        compression_level=args.temp_compression_level,
                        overwrite=False,
                        phase2_processes=args.phase2_processes,
                        phase2_start_method=args.phase2_start_method,
                        existing_sorted_by_bucket=existing_sorted_by_bucket,
                    )
                )
                _write_phase2a_checkpoint(
                    temp_jet_dir,
                    parts_by_bucket,
                    temp_counts,
                    num_buckets,
                    args.seed,
                    output_tree_path,
                )
        else:
            print(
                "Resume Phase 2A: keeping completed sorted runs and sorting "
                "only the remaining plain ROOT runs.",
                flush=True,
            )
            parts_by_bucket, temp_counts = (
                _phase2_sort_bucket_parts_multiprocess(
                    parts_by_bucket=unsorted_by_bucket,
                    bucket_counts=None,
                    schema=schema,
                    output_tree_path=output_tree_path,
                    compression_name=args.temp_compression,
                    compression_level=args.temp_compression_level,
                    overwrite=args.overwrite,
                    phase2_processes=args.phase2_processes,
                    phase2_start_method=args.phase2_start_method,
                    existing_sorted_by_bucket=existing_sorted_by_bucket,
                )
            )
            _write_phase2a_checkpoint(
                temp_jet_dir,
                parts_by_bucket,
                temp_counts,
                num_buckets,
                args.seed,
                output_tree_path,
            )
        total_events = int(sum(temp_counts))
        print(
            "Recovery cache contains {:,} events across {:,} sorted runs. "
            "Starting Phase 2B merge.".format(
                total_events,
                sum(len(paths) for paths in parts_by_bucket),
            ),
            flush=True,
        )
    else:
        (
            parts_by_bucket,
            temp_counts,
            total_events,
            nonempty_files,
            average_source_bytes,
        ) = _phase1_make_temp_bucket_parts(
            jet_type=jet_type,
            files=files,
            schema=schema,
            preferred_tree_path=args.tree_path,
            fallback_tree_path=args.fallback_tree_path,
            output_tree_path=output_tree_path,
            temp_jet_dir=temp_jet_dir,
            num_buckets=num_buckets,
            read_step=args.read_step,
            flush_events=args.temp_flush_events,
            temp_part_events=args.temp_part_events,
            max_pending_events=args.max_pending_events,
            max_pending_bytes=int(args.max_pending_gb * (1024 ** 3)),
            phase1_processes=args.phase1_processes,
            phase1_start_method=args.phase1_start_method,
            seed=args.seed,
            compression_name=args.temp_compression,
            compression_level=args.temp_compression_level,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )

        if (
            total_events_before_phase1 is not None
            and total_events != total_events_before_phase1
        ):
            raise ShuffleError(
                "Phase 1 counted {:,} events, but automatic inspection counted {:,}."
                .format(total_events, total_events_before_phase1)
            )

        print(
            "Phase 1 counted {:,} events across {:,}/{:,} non-empty/source "
            "files; actual mean temporary bucket size: {:,.1f}.".format(
                total_events,
                nonempty_files,
                len(files),
                total_events / float(num_buckets),
            ),
            flush=True,
        )
        print(
            "Phase 1 produced {:,} bounded temporary part files. Starting "
            "multiprocess Phase 2A sorting...".format(
                sum(len(paths) for paths in parts_by_bucket)
            ),
            flush=True,
        )
        parts_by_bucket, temp_counts = (
            _phase2_sort_bucket_parts_multiprocess(
                parts_by_bucket=parts_by_bucket,
                bucket_counts=temp_counts,
                schema=schema,
                output_tree_path=output_tree_path,
                compression_name=args.temp_compression,
                compression_level=args.temp_compression_level,
                overwrite=args.overwrite,
                phase2_processes=args.phase2_processes,
                phase2_start_method=args.phase2_start_method,
                existing_sorted_by_bucket=None,
            )
        )
        _write_phase2a_checkpoint(
            temp_jet_dir,
            parts_by_bucket,
            temp_counts,
            num_buckets,
            args.seed,
            output_tree_path,
        )

    max_bucket_events = max(temp_counts) if temp_counts else 0
    expected_final_files = int(
        math.ceil(float(total_events) / args.events_per_file)
    )
    active_phase2_processes = min(
        int(args.phase2_processes),
        expected_final_files,
    )
    if average_source_bytes > 0:
        estimated_phase2_batch_gib = (
            min(args.events_per_file, total_events)
            * average_source_bytes
            / float(1024 ** 3)
        )
        estimated_phase2_aggregate_gib = (
            estimated_phase2_batch_gib * active_phase2_processes
        )
        print(
            "Largest temporary bucket: {:,} events. Phase 2 does not decode "
            "the whole bucket at once; each process materializes at most one "
            "{:,}-event final batch. Estimated decoded payload: {:.1f} "
            "GiB/process, {:.1f} GiB across {} active processes, before "
            "merge-key and writer overhead.".format(
                max_bucket_events,
                args.events_per_file,
                estimated_phase2_batch_gib,
                estimated_phase2_aggregate_gib,
                active_phase2_processes,
            ),
            flush=True,
        )
        if estimated_phase2_aggregate_gib > args.phase2_memory_warning_gb:
            print(
                "  WARNING: concurrent Phase-2 batches may approach the memory "
                "allocation. Reduce --phase2-processes or --events-per-file.",
                file=sys.stderr,
                flush=True,
            )
    else:
        print(
            "Largest temporary bucket: {:,} events. Recovery mode skips the "
            "Phase-1 decoded-byte estimate; each Phase-2 process still "
            "materializes at most one {:,}-event final batch.".format(
                max_bucket_events, args.events_per_file
            ),
            flush=True,
        )

    output_paths, output_counts = _phase2_write_final_files_multiprocess(
        prefix=prefix,
        parts_by_bucket=parts_by_bucket,
        bucket_counts=temp_counts,
        schema=schema,
        output_tree_path=output_tree_path,
        output_dir=output_dir,
        total_events=total_events,
        events_per_file=args.events_per_file,
        compression_name=args.final_compression,
        compression_level=args.final_compression_level,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        phase2_processes=args.phase2_processes,
        phase2_start_method=args.phase2_start_method,
        resume_existing=recovery_mode,
    )

    if args.verify:
        print("  Verifying final ROOT metadata and branch sets...", flush=True)
        _verify_final_files(
            output_paths,
            output_counts,
            schema,
            output_tree_path,
            args.events_per_file,
            uproot,
        )

    if not args.keep_temp:
        shutil.rmtree(str(temp_jet_dir))

    elapsed = time.time() - jet_started
    print(
        "Completed {}: {:,} events -> {} files in {:.1f} min".format(
            jet_type, total_events, len(output_paths), elapsed / 60.0
        ),
        flush=True,
    )
    return JetResult(
        jet_type=jet_type,
        prefix=prefix,
        source_events=total_events,
        source_files=len(files),
        temporary_buckets=num_buckets,
        temporary_bucket_counts=temp_counts,
        output_files=[str(path) for path in output_paths],
        output_file_counts=output_counts,
        elapsed_seconds=elapsed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Event-level external shuffle for per-jet-type CMS ROOT datasets, "
            "targeting uproot==5.6.6 and awkward==2.7.4."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary bucket root; defaults to TARGET_DIR/.shuffle_tmp",
    )
    parser.add_argument(
        "--jet-types",
        nargs="+",
        default=None,
        help="Only process these source subdirectory names",
    )
    parser.add_argument("--file-glob", default="*.root")
    parser.add_argument("--tree-path", default=DEFAULT_TREE_PATH)
    parser.add_argument("--fallback-tree-path", default=FALLBACK_TREE_PATH)
    parser.add_argument(
        "--output-tree-path",
        default=FALLBACK_TREE_PATH,
        help=(
            "Output TTree path, such as 'tree' or 'deepntuplizerAK8/tree'"
        ),
    )
    parser.add_argument(
        "--prefix-map",
        action="append",
        default=[],
        metavar="SUBDIR=PREFIX",
        help="Override final filename prefix; may be supplied repeatedly",
    )
    parser.add_argument("--events-per-file", type=int, default=100_000)
    parser.add_argument(
        "--num-temp-buckets",
        type=int,
        default=32,
        help=(
            "Fixed number of temporary buckets. Positive values avoid an "
            "upfront all-file event-count scan. Set to 0 to restore automatic "
            "bucket sizing from --temp-bucket-events"
        ),
    )
    parser.add_argument(
        "--temp-bucket-events",
        type=int,
        default=50_000,
        help=(
            "Target mean events per temporary bucket; used only when "
            "--num-temp-buckets=0"
        ),
    )
    parser.add_argument(
        "--read-step",
        type=_parse_step,
        default="512 MB",
        help="Uproot 5 step_size value: entry count or memory string",
    )
    parser.add_argument(
        "--temp-flush-events",
        type=int,
        default=100_000,
        help="Flush a bucket's pending fragments after this many events",
    )
    parser.add_argument(
        "--temp-part-events",
        type=int,
        default=20_000,
        help=(
            "Maximum events in each process-local temporary part file. "
            "Bounded parts limit temporary-file size and are sorted "
            "independently in Phase 2"
        ),
    )
    parser.add_argument(
        "--max-pending-events",
        type=int,
        default=3_000_000,
        help="Global Phase-1 queued-plus-pending event limit",
    )
    parser.add_argument(
        "--max-open-temp-files",
        type=int,
        default=0,
        help="0 derives the safe limit from RLIMIT_NOFILE",
    )
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument(
        "--compression",
        choices=("zlib", "lzma", "lz4", "none"),
        default=None,
        help="Legacy shortcut that sets both temporary and final compression",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Legacy shortcut that sets both temporary and final levels",
    )
    parser.add_argument(
        "--temp-compression",
        choices=("zlib", "lzma", "lz4", "none"),
        default="zlib",
        help="Compression for Phase-1 temporary bucket ROOT files",
    )
    parser.add_argument("--temp-compression-level", type=int, default=1)
    parser.add_argument(
        "--final-compression",
        choices=("zlib", "lzma", "lz4", "none"),
        default="zlib",
        help="Compression for final shuffled ROOT files",
    )
    parser.add_argument("--final-compression-level", type=int, default=4)
    parser.add_argument(
        "--phase1-processes",
        type=int,
        default=16,
        help=(
            "Independent Phase-1 processes. Each process reads its own source "
            "files and writes separate bucket part files"
        ),
    )
    parser.add_argument(
        "--phase1-start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method for independent Phase-1 workers",
    )
    parser.add_argument(
        "--phase2-processes",
        "--final-write-processes",
        dest="phase2_processes",
        type=int,
        default=4,
        help=(
            "Independent Phase-2 processes. Each process merges the logical "
            "buckets overlapping its contiguous final-file range and writes "
            "those final ROOT files itself. --final-write-processes is kept "
            "as a compatibility alias"
        ),
    )
    parser.add_argument(
        "--phase2-start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method for independent Phase-2 workers",
    )
    parser.add_argument(
        "--max-pending-gb",
        type=float,
        default=48.0,
        help="Approximate byte cap for Phase-1 queued plus pending arrays",
    )
    parser.add_argument(
        "--phase2-memory-warning-gb",
        type=float,
        default=80.0,
        help="Warn when estimated aggregate Phase-2 decoded batches exceed this size",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Print progress after this many additional events; 0 disables",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON run manifest path. No metadata file is placed inside "
            "TARGET_DIR unless this is explicitly set there"
        ),
    )
    recovery_group = parser.add_mutually_exclusive_group()
    recovery_group.add_argument(
        "--start-from-phase2a",
        action="store_true",
        help=(
            "Resume from cached Phase-1 runs. Keep completed *_sorted.root "
            "files, delete only incomplete *_sorted.root.partial files, sort "
            "remaining plain runs, then continue with Phase 2B"
        ),
    )
    recovery_group.add_argument(
        "--start-from-phase2b",
        action="store_true",
        help=(
            "Resume directly from completed *_sorted.root runs. Any plain temp "
            "run or temp partial causes an error. Completed final ROOT shards "
            "are retained, final *.partial files are deleted, and only missing "
            "final shard numbers are regenerated"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Normal mode: replace existing per-jet target and temporary "
            "directories. Recovery mode: preserve temporary files and completed "
            "final ROOT shards; only missing shards are regenerated"
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary bucket ROOT files after successful final writing",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip final metadata/count/branch-set verification",
    )
    parser.set_defaults(verify=True)
    parser.add_argument(
        "--skip-io-self-test",
        action="store_true",
        help=(
            "Skip the small startup Uproot/Awkward ROOT round-trip test. "
            "Not recommended unless the same environment has already passed it"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover files and inspect one reference schema without writing. "
            "In fixed-bucket mode this intentionally does not count or validate "
            "every source file"
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "events_per_file",
        "temp_bucket_events",
        "temp_flush_events",
        "max_pending_events",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ShuffleError("--{} must be positive.".format(name.replace("_", "-")))
    if int(args.num_temp_buckets) < 0:
        raise ShuffleError("--num-temp-buckets must be non-negative.")
    if args.temp_part_events <= 0:
        raise ShuffleError("--temp-part-events must be positive.")
    if args.max_pending_events < args.temp_flush_events:
        raise ShuffleError(
            "--max-pending-events must be at least --temp-flush-events."
        )
    if args.progress_every < 0:
        raise ShuffleError("--progress-every must be non-negative.")
    if args.max_open_temp_files < 0:
        raise ShuffleError("--max-open-temp-files must be non-negative.")
    for name in (
        "phase1_processes",
        "phase2_processes",
    ):
        if int(getattr(args, name)) <= 0:
            raise ShuffleError("--{} must be positive.".format(name.replace("_", "-")))
    if float(args.max_pending_gb) <= 0:
        raise ShuffleError("--max-pending-gb must be positive.")
    if float(args.phase2_memory_warning_gb) <= 0:
        raise ShuffleError("--phase2-memory-warning-gb must be positive.")
    recovery_mode = bool(
        args.start_from_phase2a or args.start_from_phase2b
    )
    if recovery_mode and args.dry_run:
        raise ShuffleError("Recovery flags cannot be combined with --dry-run.")
    if recovery_mode and int(args.num_temp_buckets) <= 0:
        raise ShuffleError(
            "Recovery requires the original positive --num-temp-buckets value."
        )
    if recovery_mode and not args.jet_types:
        raise ShuffleError(
            "Recovery requires explicit --jet-types to avoid touching the wrong "
            "temporary directory."
        )
    output_tree_name = str(args.output_tree_path).strip("/")
    if not output_tree_name:
        raise ShuffleError("--output-tree-path must be non-empty.")
    args.output_tree_path = output_tree_name


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        uproot, awkward = _import_modern_io()
        source = args.source_dir.expanduser().resolve()
        target = args.target_dir.expanduser().resolve()
        temp = (
            args.temp_dir.expanduser().resolve()
            if args.temp_dir is not None
            else (target / ".shuffle_tmp").resolve()
        )
        _ensure_safe_paths(source, target, temp)
        prefix_map = _parse_prefix_map(args.prefix_map)
        if args.compression is not None:
            args.temp_compression = args.compression
            args.final_compression = args.compression
        if args.compression_level is not None:
            args.temp_compression_level = args.compression_level
            args.final_compression_level = args.compression_level
        temp_compression = _compression_object(
            uproot, args.temp_compression, args.temp_compression_level
        )
        final_compression = _compression_object(
            uproot, args.final_compression, args.final_compression_level
        )
        if not args.skip_io_self_test:
            print(
                "Running modern Uproot/Awkward ROOT round-trip self-test...",
                flush=True,
            )
            _run_modern_io_self_test(uproot, awkward)
            print("Modern ROOT I/O self-test passed.", flush=True)
        jet_dirs = _discover_jet_dirs(source, args.jet_types)

        if not args.dry_run:
            target.mkdir(parents=True, exist_ok=True)
            temp.mkdir(parents=True, exist_ok=True)

        print("Source: {}".format(source))
        print("Target: {}".format(target))
        print("Temporary: {}".format(temp))
        print("Jet types: {}".format(", ".join(path.name for path in jet_dirs)))
        print("Seed: {}".format(args.seed))
        if args.start_from_phase2a:
            print("Start mode: resume from Phase 2A")
        elif args.start_from_phase2b:
            print("Start mode: resume from Phase 2B")
        else:
            print("Start mode: full run from Phase 1")
        print(
            "Processes: Phase 1={} ({}), Phase 2={} ({})"
            .format(
                args.phase1_processes,
                args.phase1_start_method,
                args.phase2_processes,
                args.phase2_start_method,
            )
        )
        print(
            "Compression: temp={} level {}, final={} level {}".format(
                args.temp_compression,
                args.temp_compression_level,
                args.final_compression,
                args.final_compression_level,
            )
        )

        all_started = time.time()
        results: List[JetResult] = []
        for jet_dir in jet_dirs:
            prefix = prefix_map.get(jet_dir.name, jet_dir.name)
            results.append(
                process_jet_type(
                    jet_dir,
                    target,
                    temp,
                    prefix,
                    args,
                    uproot,
                    awkward,
                    temp_compression,
                    final_compression,
                )
            )

        root_manifest = {
            "script": Path(__file__).name,
            "uproot_version": str(getattr(uproot, "__version__", "unknown")),
            "awkward_version": str(getattr(awkward, "__version__", "unknown")),
            "source_dir": str(source),
            "target_dir": str(target),
            "temp_dir": str(temp),
            "seed": args.seed,
            "dry_run": args.dry_run,
            "implementation": "multiprocess-uproot5-phase1-phase2-resume-v8",
            "results": [asdict(result) for result in results],
            "elapsed_seconds": time.time() - all_started,
        }
        if args.manifest_path is not None:
            manifest_path = args.manifest_path.expanduser().resolve()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists() and not args.overwrite:
                raise ShuffleError(
                    "Manifest already exists: {}. Pass --overwrite to replace it."
                    .format(manifest_path)
                )
            with manifest_path.open("w", encoding="utf-8") as stream:
                json.dump(root_manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print("Manifest: {}".format(manifest_path), flush=True)

        if not args.keep_temp and not args.dry_run:
            try:
                temp.rmdir()
            except OSError:
                pass

        print(
            "\nAll requested jet types completed in {:.1f} min.".format(
                (time.time() - all_started) / 60.0
            ),
            flush=True,
        )
        return 0
    except (ShuffleError, OSError, KeyError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
