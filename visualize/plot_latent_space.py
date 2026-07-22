"""
Latent space plotting utilities.
"""

import os
import sys
from typing import List, Tuple

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reduce_to_2d(bg_latents, sg_latents):
    all_latents = np.concatenate([bg_latents, sg_latents], axis=0)

    if all_latents.shape[1] == 2:
        return bg_latents, sg_latents, "Latent dim 1", "Latent dim 2"

    if all_latents.shape[1] < 2:
        raise ValueError(
            f"Latent dimension is {all_latents.shape[1]}, cannot plot 2D latent space."
        )

    pca = PCA(n_components=2)
    all_latents_2d = pca.fit_transform(all_latents)

    bg_count = len(bg_latents)
    bg_latents_2d = all_latents_2d[:bg_count]
    sg_latents_2d = all_latents_2d[bg_count:]

    x_label = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)"
    y_label = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)"

    return bg_latents_2d, sg_latents_2d, x_label, y_label


def plot_latent_space(
    bg_latents_2d,
    sg_latents_2d,
    background_label: str,
    signal_label: str,
    output_path: str,
    x_label: str = "Latent dim 1",
    y_label: str = "Latent dim 2",
):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        bg_latents_2d[:, 0],
        bg_latents_2d[:, 1],
        label=background_label,
        alpha=0.35,
        s=8,
    )

    ax.scatter(
        sg_latents_2d[:, 0],
        sg_latents_2d[:, 1],
        label=signal_label,
        alpha=0.35,
        s=8,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Latent Space")
    range_lim_low = min(bg_latents_2d.min(), sg_latents_2d.min()) - 0.1
    range_lim_high = max(bg_latents_2d.max(), sg_latents_2d.max()) + 0.1
    ax.set_xlim(range_lim_low, range_lim_high)
    ax.set_ylim(range_lim_low, range_lim_high)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)