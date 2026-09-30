"""Plot all validation projected CLS states, colored by jet type, in the backbone run.

python scripts/plot_lejepa_tsne.py plots/run-lejepa-semi-sup-triplet-jetclass-ddp-fast-hbb-long
"""
import argparse
from pathlib import Path
import sys

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
    # labels used to train the backbone. No val_steps, max_events, or subsampling.
    if run.backend.dataset_name == "jetclass":
        labels = list(run.backend.label_axis)
    else:
        labels = list(run.backend.cms_splits["val"])
    latents, targets = [], []
    for batch in tqdm(run.loader("val", labels), desc="All validation projected CLS states"):
        latents.append(run.encode(batch)) # keep on GPU for cuda TSNE
        targets.append(batch["y"].argmax(dim=-1))
    if not latents or sum(len(x) for x in latents) < 2:
        raise ValueError("t-SNE requires at least two validation events.")
    x, y = torch.cat(latents), torch.cat(targets).to(x.device) # note: run.encode moves batch to gpu and returns latents on gpu, but y is still on cpu
    print(f"Computed latents for all {len(x):,} validation events ({x.shape[1]} representation dimensions).", flush=True)
    # select a 3000 sample subset if there are more than 3000 events
    if len(x) > 3000:
        indices = torch.randperm(len(x), device=x.device)[:3000]
        x = x[indices]
        y = y[indices]
        print(f"Subsampled to 3000 validation events.", flush=True)
    
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite backbone CLS states.")
    print(f"Running t-SNE on {len(x):,} validation events.", flush=True)
    embedding = TSNE(n_components=2, perplexity=min(args.perplexity, len(x) - 1),
                    ).fit_transform(x)
    # embedding = embedding.cpu() # move to cpu for plotting
    # TorchTSNE already returns a numpy array on CPU!
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
