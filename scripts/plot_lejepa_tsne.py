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
    parser.add_argument("--perplexity", type=float, default=300.0)
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
    started = perf_counter()
    label_counts = torch.zeros(len(run.backend.label_axis), dtype=torch.long)
    print(f"Mixed validation: {len(labels)} jet types, {run.num_workers} workers; "
          "loading the initial active shards.", flush=True)
    # Keep all validation types in the same batch-normalization population.
    # Split/group only after inference, using the aligned dataset labels.
    loader = run.loader("val", labels, infinite=False, drop_last=False,
                        persistent_workers=False)
    for batch in tqdm(loader, desc="Mixed validation representations", unit="batch"):
        representations = run.encode(batch).cpu()
        if not torch.isfinite(representations).all():
            raise ValueError("Non-finite backbone representations in mixed validation.")
        batch_targets = batch["y"].argmax(dim=-1).cpu()
        batch_priorities = torch.rand(len(representations), generator=generator,
                                      dtype=torch.float64)
        total_events += len(representations)
        label_counts += torch.bincount(batch_targets, minlength=len(label_counts))
        if x is None:
            x, y, priorities = representations, batch_targets, batch_priorities
        else:
            x = torch.cat((x, representations))
            y = torch.cat((y, batch_targets))
            priorities = torch.cat((priorities, batch_priorities))
        if len(x) > sample_limit:
            selected = priorities.topk(sample_limit).indices
            x, y, priorities = x[selected], y[selected], priorities[selected]
    print(f"Mixed validation completed in {perf_counter() - started:.1f}s.", flush=True)
    for label in labels:
        count = int(label_counts[run.backend.label_axis.index(label)])
        print(f"  {label}: {count:,} events encoded.", flush=True)
    if x is None or len(x) < 2:
        raise ValueError("t-SNE requires at least two validation events.")
    print(f"Encoded all {total_events:,} validation events; retained {len(x):,} "
          f"points ({x.shape[1]} representation dimensions).", flush=True)
    x = x.to(run.device)

    if not torch.isfinite(x).all():
        raise ValueError("Non-finite backbone CLS states.")
    print(f"Running t-SNE on {len(x):,} validation events.", flush=True)
    # Fit once to the mixed sample; group by the saved labels only for plotting.
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
            ax.scatter(embedding[selected, 0], embedding[selected, 1], s=8,
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
