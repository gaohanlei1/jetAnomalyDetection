"""Shared CWoLa batching and truth-only diagnostic scoring for CMS/JetClass."""
import math
from time import perf_counter
from itertools import count

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from scripts.run_train_representation_flow import encode_finite, write_json


POOLED_BACKGROUND = "All backgrounds"
POOLED_SIGNAL = "All signals"


def validate_label_sets(summary):
    backgrounds = list(dict.fromkeys(summary.get("background_labels", [])))
    signals = list(dict.fromkeys(summary.get("signal_labels", [])))
    if not backgrounds or not signals:
        raise ValueError("CWoLa requires nonempty background_labels and signal_labels.")
    overlap = sorted(set(backgrounds) & set(signals))
    if overlap:
        raise ValueError(f"Background and signal label sets must be disjoint; overlap: {overlap}")
    return backgrounds, signals


def parse_label_override(value):
    """CLI comma-separated physical labels, preserving order and removing duplicates."""
    labels = [label.strip() for label in value.split(",")]
    if not labels or any(not label for label in labels):
        raise ValueError("Expected a nonempty comma-separated list of labels, e.g. label_QCD,label_Wqq.")
    return list(dict.fromkeys(labels))


def resolve_label_sets(summary, backgrounds=None, signals=None):
    effective = dict(summary)
    if backgrounds is not None:
        effective["background_labels"] = backgrounds
    if signals is not None:
        effective["signal_labels"] = signals
    # Check the effective downstream task, not the backbone's historical task.
    return validate_label_sets(effective)


def configure_cwola_labels(run, backgrounds, signals):
    """Change downstream data selection only, never checkpoint/model metadata."""
    backgrounds, signals = validate_label_sets(dict(background_labels=backgrounds, signal_labels=signals))
    run.backend.validate_requested_labels(backgrounds + signals)
    if run.backend.dataset_name == "cms":
        # Preserve the saved CMS split; never silently invent a new split for
        # categories not covered by the backbone's manifest.
        for split in ("train", "val", "test"):
            missing = sorted(set(backgrounds + signals) - set(run.backend.cms_splits[split]))
            if missing:
                raise ValueError(f"CMS {split} manifest has no split for {missing}; "
                                 "choose labels covered by the saved CMS split manifest.")
    run.backgrounds, run.signals = backgrounds, signals


def batch_sizes(batch_size, signal_fraction, *, for_roc=False):
    """Equal reference/mixture halves; round signal count to the nearest event."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch-size must be an even integer >= 2 (reference + mixture halves).")
    if not math.isfinite(signal_fraction) or not 0 <= signal_fraction <= 1:
        raise ValueError("signal-fraction must be finite and in [0, 1].")
    n_signal = int(math.floor(batch_size / 2 * signal_fraction + 0.5))
    # A zero-signal training mixture is valid, but a truth ROC needs positives.
    # This extra signal is diagnostic-only and never enters the training loss.
    if for_roc:
        n_signal = max(1, n_signal)
    return batch_size - n_signal, n_signal


def make_loaders(run, split, signal_fraction, *, steps, for_roc=False, training=False, num_workers=None):
    n_background, n_signal = batch_sizes(run.batch_size, signal_fraction, for_roc=for_roc)
    # Bounded validation repeats deterministically; test always visits events at
    # most once. steps=0 is full finite test evaluation (including partial tails).
    infinite = training or (split == "val" and steps > 0)
    drop_last = training or (split == "val" and steps > 0)
    if training:
        bg_limit = sg_limit = run.summary.get("max_train_events")
    elif split == "val":
        bg_limit = run.summary.get("max_val_events")
        sg_limit = run.summary.get("max_test_signal_events")
    else:
        bg_limit = run.summary.get("max_test_background_events")
        sg_limit = run.summary.get("max_test_signal_events")
    kwargs = dict(infinite=infinite, drop_last=drop_last,
                  persistent_workers=training,
                  prefetch_factor=int(run.summary.get("prefetch_factor", 2)) if training else 1)
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    background = run.loader(split, run.backgrounds, batch_size=n_background,
                            seed_offset=0 if training else 202, max_events=bg_limit,
                            active_shards=int(run.summary.get("shuffle_active_shards", 3)) if training else len(run.backgrounds),
                            **kwargs)
    signal = (run.loader(split, run.signals, batch_size=n_signal,
                         seed_offset=101 if training else 404, max_events=sg_limit,
                         active_shards=int(run.summary.get("shuffle_active_shards", 3)) if training else len(run.signals),
                         **kwargs)
              if n_signal else None)
    return background, signal


def combine_batches(background, signal):
    """Order: [reference B | mixture B | mixture S]. Never train on jet labels.

    Counts, truth labels and weak labels are derived from the actual rows.
    Finite diagnostic tails may contain only one population; weak labels on
    those tails are unused by test evaluation.
    """
    batches = [batch for batch in (background, signal) if batch is not None]
    if not batches:
        raise ValueError("Empty CWoLa batch.")
    n_bg = 0 if background is None else len(background["x_particles"])
    n_sg = 0 if signal is None else len(signal["x_particles"])
    batch = {key: torch.cat([item[key] for item in batches], dim=0)
             for key in ("x_particles", "padding_mask")}
    n = n_bg + n_sg
    weak = torch.ones(n, dtype=torch.float32)
    weak[:min(n_bg, n // 2)] = 0
    truth = torch.cat([torch.zeros(n_bg, dtype=torch.int64), torch.ones(n_sg, dtype=torch.int64)])
    return batch, weak, truth


def paired_batches(loaders, steps, *, require_full_budget=True):
    """Advance both datasets independently; keep finite test tails from either."""
    iterators = [iter(loader) if loader is not None else None for loader in loaders]
    for index in range(steps) if steps else count():
        parts = []
        for i, iterator in enumerate(iterators):
            part = next(iterator, None) if iterator is not None else None
            if part is None:
                iterators[i] = None
                if require_full_budget and loaders[i] is not None:
                    raise RuntimeError(f"CWoLa {'background' if i == 0 else 'signal'} stream ended at batch {index}; "
                                       "check event limits, batch sizes and dataset availability.")
            parts.append(part)
        if all(part is None for part in parts):
            break
        yield parts


def representation_batches(run, loaders, steps, *, per_signal, require_full_budget,
                           cache, description):
    """Cache only fixed CPU representations/labels, never scores or raw jets.

    Each cache belongs to one loader pair and budget for one frozen backbone.
    Commit only a complete pass, so a failed pass cannot leave a partial cache.
    """
    if cache:
        print(f"{description}: using {len(cache)} cached representation batches.", flush=True)
        yield from cache
        return
    pending = []
    started = perf_counter()
    print(f"{description}: starting loaders; waiting for initial ROOT shards...", flush=True)
    for index, (background, signal) in enumerate(
            paired_batches(loaders, steps, require_full_budget=require_full_budget)):
        if index == 0:
            print(f"{description}: first input batch ready after {perf_counter() - started:.1f}s.", flush=True)
        batch, weak, truth = combine_batches(background, signal)
        representation = encode_finite(run, batch)
        ids = None
        if per_signal:
            n_bg = 0 if background is None else len(background["x_particles"])
            ids = np.full(len(truth), -1, dtype=np.int64)
            if signal is not None:
                ids[n_bg:] = signal["y"].argmax(dim=-1).cpu().numpy()
        item = (representation.detach().cpu() if cache is not None else representation, weak, truth, ids)
        if cache is not None:
            pending.append(item)
        yield item
    if cache is not None:
        cache.extend(pending)
        size = sum(t.numel() * t.element_size() for item in cache for t in item[:3])
        print(f"{description}: cached {len(cache)} batches ({size / 2**20:.1f} MiB CPU tensors); "
              "later epochs skip ROOT loading and backbone inference.", flush=True)


@torch.no_grad()
def evaluate(model, run, loaders, steps, *, collect_scores=True, per_signal=False,
             require_full_budget=True, description="Validation", cache=None):
    model.eval()
    started = perf_counter()
    score_parts, truth_parts, signal_id_parts = [], [], []
    loss_sum, num_events = 0.0, 0
    batches = representation_batches(run, loaders, steps, per_signal=per_signal,
                                     require_full_budget=require_full_budget,
                                     cache=cache, description=description)
    for representation, weak, truth, ids in tqdm(batches, total=steps or None,
                                                 desc=description, unit="batch"):
        logits = model.forward_logits(representation.to(run.device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite CWoLa evaluation logits.")
        # Test tails can be unbalanced, so only validation uses this weak loss.
        loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(
            logits, weak.to(run.device), reduction="sum").item()
        num_events += len(logits)
        if collect_scores:
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
            truth_parts.append(truth.numpy())
            if per_signal:
                signal_id_parts.append(ids)
    if not num_events:
        raise RuntimeError(f"{description}: no events.")
    result = {"loss": loss_sum / num_events, "num_events": num_events}
    if collect_scores:
        scores, truth = np.concatenate(score_parts), np.concatenate(truth_parts)
        if len(np.unique(truth)) != 2:
            raise RuntimeError(f"{description}: truth ROC requires both background and signal events.")
        result.update(scores=scores, truth=truth, auc=float(roc_auc_score(truth, scores)))
        if per_signal:
            result["signal_ids"] = np.concatenate(signal_id_parts)
    print(f"{description}: {num_events:,} events in {perf_counter() - started:.1f}s.", flush=True)
    return result
