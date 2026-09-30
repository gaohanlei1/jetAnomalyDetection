"""Read-only restoration of a LeJEPA run for projected CLS inference and flow training."""
import json
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader

from scripts.diagnose_lejepa_latents import (
    DiagnosticDatasetBackend, build_model, read_state_dict, autocast_context,
    seed_everything,
)
from datasets import cms_streaming, jetclass_streaming


def add_backbone_arguments(parser):
    parser.add_argument("run_dir", type=Path, help="Backbone run containing summary.json.")
    parser.add_argument("--checkpoint", type=Path, help="Default: run_dir/best_model.pth.")
    parser.add_argument("--dataset-root", type=Path, help="Relocate the saved dataset root.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)


class LeJEPARun:
    def __init__(self, args):
        self.run_dir = args.run_dir.expanduser().resolve()
        with (self.run_dir / "summary.json").open() as handle:
            self.summary = json.load(handle)
        self.seed = int(self.summary.get("base_seed", self.summary.get("seed", 42)))
        seed_everything(self.seed)
        self.device = torch.device(args.device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"))
        self.backgrounds = list(dict.fromkeys(self.summary["background_labels"]))
        self.signals = list(dict.fromkeys(self.summary.get("signal_labels", [])))
        if set(self.backgrounds) & set(self.signals):
            raise ValueError("Background and signal labels must be disjoint.")
        # Older runs did not save this field. Recover the dataset-module convention.
        restored = dict(self.summary)
        if "batch_standardized_particle_features" not in restored:
            module = cms_streaming if restored.get("dataset", "jetclass") == "cms" else jetclass_streaming
            restored["batch_standardized_particle_features"] = [
                name for name in module.BATCH_NORMALIZED_FEATURES
                if name in restored["particle_features"]]
            warnings.warn("Legacy summary: using dataset-module batch-normalized features.")
        self.backend = DiagnosticDatasetBackend.from_summary(
            summary=restored, run_dir=self.run_dir, dataset_override=None,
            dataset_root_override=args.dataset_root, cms_manifest_override=None,
            max_num_particles_override=None)
        self.checkpoint = (args.checkpoint or self.run_dir / "best_model.pth").expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.model = build_model(restored, self.backend, self.device)
        state = read_state_dict(self.checkpoint, self.device)
        result = self.model.load_state_dict(state, strict=False)
        stats = ("_feature_running_mean", "_feature_running_var", "_feature_num_batches_tracked")
        missing = [key for key in result.missing_keys if not key.endswith(stats)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")
        missing_stats = any(key.endswith(stats) for key in result.missing_keys)
        self.model._use_frozen_feature_stats_in_eval = not missing_stats
        if missing_stats:
            warnings.warn("Legacy checkpoint without running statistics: use per-batch feature normalization.")
        self.model.eval().requires_grad_(False)
        self.batch_size = args.batch_size if args.batch_size is not None else int(
            self.summary.get("per_rank_batch_size", self.summary.get("batch_size", 128)))
        self.num_workers = args.num_workers if args.num_workers is not None else int(self.summary.get("num_workers", 0))
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch-size must be positive; num-workers must be nonnegative.")
        self.precision = str(self.summary.get("precision", "fp32"))

    def loader(self, split, labels, *, infinite=False, seed_offset=0, max_events=None):
        dataset = self.backend.make_dataset(split, labels, self.seed + seed_offset)
        dataset.infinite = infinite
        dataset.max_events = max_events
        # Finite iterators visit each event once; retain the final partial batch.
        # Both dataset backends preload the ENTIRE split when shuffle_files
        # is False. Use their active-pool streaming path even for finite eval.
        # The fixed dataset seed makes evaluation order repeatable; infinite
        # alone controls repetition, so finite evaluation still visits once.
        dataset.shuffle_files = True
        if not infinite:
            # One active shard per label/worker rather than inheriting the
            # training pool size for every single-label validation loader.
            dataset.shuffle_active_shards = len(labels)
        module = cms_streaming if self.backend.dataset_name == "cms" else jetclass_streaming
        collate = module.collate_cms_tensors if self.backend.dataset_name == "cms" else module.collate_jetclass_tensors
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                      pin_memory=self.device.type == "cuda", collate_fn=collate,
                      drop_last=False, persistent_workers=infinite and self.num_workers > 0)
        if self.num_workers:
            kwargs["prefetch_factor"] = (
                int(self.summary.get("prefetch_factor", 2)) if infinite else 1
            )
            # Evaluation workers start after CUDA/CPU thread pools are active.
            # Avoid inheriting their state via the Linux default fork method.
            kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(dataset, **kwargs)

    @torch.no_grad()
    def encode(self, batch):
        # Full-view CLS followed by representation_head; no pretraining path.
        with autocast_context(self.device, self.precision):
            cls = self.model.forward_representation(batch["x_particles"].to(self.device, non_blocking=True),
                             padding_mask=batch["padding_mask"].to(self.device, non_blocking=True))
        return cls.float()
