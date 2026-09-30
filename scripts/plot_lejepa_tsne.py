"""Plot all validation projected CLS states, colored by jet type, in the backbone run.

python scripts/plot_lejepa_tsne.py plots/run-lejepa-semi-sup-triplet-jetclass-ddp-fast-hbb-long
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from tsne_torch import TorchTSNE as TSNE
from tqdm import tqdm

from scripts.lejepa_run import LeJEPARun, add_backbone_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backbone_arguments(parser)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()
    if args.perplexity <= 0:
        parser.error("--perplexity must be positive")
    run = LeJEPARun(args)
    torch.manual_seed(run.seed)
    torch.cuda.manual_seed_all(run.seed)
    # Include every available jet type in the validation split, not just the
    # labels used to train the backbone. Forward all events; sample only the plot.
    if run.backend.dataset_name == "jetclass":
        labels = list(run.backend.label_axis)
    else:
        labels = list(run.backend.cms_splits["val"])
    # Uniform sampling without replacement via independent random priorities.
    # Forward every validation event, retaining only the 3000 plotting points
    # on CPU; never accumulate the full validation representation set on GPU.
    sample_limit = 3000
    generator = torch.Generator().manual_seed(run.seed)
    x = y = priorities = None
    total_events = 0
    for label in labels:
        started = perf_counter()
        label_events = 0
        print(f"Validation {label}: starting loader ({run.num_workers} workers; "
              "one active shard per worker).", flush=True)
        # Each worker loads only this type's shard before its first batch.
        loader = run.loader("val", [label])
        for batch in tqdm(loader, desc=f"Validation {label}", unit="batch"):
            representations = run.encode(batch).cpu()
            if not torch.isfinite(representations).all():
                raise ValueError(f"Non-finite backbone representations for {label}.")
            batch_targets = batch["y"].argmax(dim=-1).cpu()
            batch_priorities = torch.rand(len(representations), generator=generator,
                                          dtype=torch.float64)
            total_events += len(representations)
            label_events += len(representations)
            if x is None:
                x, y, priorities = representations, batch_targets, batch_priorities
            else:
                x = torch.cat((x, representations))
                y = torch.cat((y, batch_targets))
                priorities = torch.cat((priorities, batch_priorities))
            if len(x) > sample_limit:
                selected = priorities.topk(sample_limit).indices
                x, y, priorities = x[selected], y[selected], priorities[selected]
        print(f"Validation {label}: {label_events:,} events in "
              f"{perf_counter() - started:.1f}s.", flush=True)
    if x is None or len(x) < 2:
        raise ValueError("t-SNE requires at least two validation events.")
    print(f"Encoded all {total_events:,} validation events; retained {len(x):,} "
          f"points ({x.shape[1]} representation dimensions).", flush=True)
    x = x.to(run.device)

    if not torch.isfinite(x).all():
        raise ValueError("Non-finite backbone CLS states.")
    print(f"Running t-SNE on {len(x):,} validation events.", flush=True)
    # Import only in the parent, after data loading (workers use spawn).
    embedding = TSNE(n_components=2, perplexity=min(args.perplexity, len(x) - 1),
                    ).fit_transform(x)
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    y = y.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20")
    for index, label in enumerate(labels):
        selected = y == run.backend.label_axis.index(label)
        if selected.any():
            ax.scatter(embedding[selected, 0], embedding[selected, 1], s=3,
                       alpha=0.5, color=colors(index % 20), rasterized=True,
                       label=f"{label.removeprefix('label_')} (n={selected.sum():,})")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Validation projected CLS representations")
    ax.legend(markerscale=4, fontsize=8, loc="best")
    fig.tight_layout()
    output = run.run_dir / "validation_representation_tsne.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
