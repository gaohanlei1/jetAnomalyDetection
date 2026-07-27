"""
Jet visualization utilities.

This module provides a helper to visualize one jet in deta-dphi space,
using token features from a (1, N, F) batch tensor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch


def plot_jet_eta_phi(
    token_features: torch.Tensor,
    output_path: os.PathLike | str,
    *,
    pt_frac_index: int = 5,
    deta_index: int = 6,
    dphi_index: int = 7,
    pdg_id_index: Optional[int] = None,
    particle_type_indices: Optional[Mapping[str, int]] = None,
    pt_frac_label: str = "pt_frac",
    cmap: str = "viridis",
    electron_marker: str = "o",
    hadron_marker: str = "^",
    muon_marker: str = "s",
    photon_marker: str = "D",
    marker_size: float = 18.0,
    alpha: float = 0.85,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.5, 5.8),
) -> None:
    """
    Plot a single jet in eta-phi space using token features.

    Parameters
    ----------
    token_features:
        Tensor with shape (1, N, F) or (N, F). Each token is a particle with F
        features. Padding tokens are ignored (tokens where all F features are 0).
    output_path:
        File path where the plot image will be written.
    pt_frac_index, deta_index, dphi_index:
        Feature indices inside the F dimension.
        For the CMS pipeline in this repo, defaults are:
        - pt_frac_index=5  -> "log_pt_fraction"
        - deta_index=6     -> "part_deta"
        - dphi_index=7     -> "part_dphi"
    pdg_id_index:
        If provided, the feature index of a per-token PDG id. Markers will be
        chosen by particle type:
        - electrons (abs(pdg)=11): circles
        - muons     (abs(pdg)=13): squares
        - photons   (abs(pdg)=22): diamonds
        - hadrons   (else): triangles
    particle_type_indices:
        Optional mapping from particle category to one-hot feature index.
        Supported keys: ``electron``, ``muon``, ``photon``, ``charged_hadron``,
        ``neutral_hadron``. Used when raw PDG ids are unavailable (e.g. CMS).
    pt_frac_label:
        Label used for the colorbar.
    cmap:
        Matplotlib colormap used to color points by pt_frac.
    marker_size, alpha:
        Scatter marker style.
    title:
        Optional plot title.
    figsize:
        Figure size.
    """

    if not isinstance(token_features, torch.Tensor):
        raise TypeError(f"token_features must be a torch.Tensor, got {type(token_features)}")

    if token_features.dim() == 3:
        if token_features.shape[0] != 1:
            raise ValueError(
                "Expected token_features with shape (1, N, F) for a single event; "
                f"got {tuple(token_features.shape)}"
            )
        feats = token_features[0]  # (N, F)
    elif token_features.dim() == 2:
        feats = token_features
    else:
        raise ValueError(
            "Expected token_features with shape (1, N, F) or (N, F); "
            f"got {tuple(token_features.shape)}"
        )

    if feats.numel() == 0:
        raise ValueError("token_features contains no elements.")

    # Ignore padding tokens: by convention they have all-zero features.
    # (Using all-zeros mask is more robust than "nonzero in one column".)
    padding_mask = torch.all(feats == 0, dim=-1)  # (N,)
    keep_mask = ~padding_mask

    if keep_mask.sum().item() == 0:
        # Nothing to plot; still write an empty plot for traceability.
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("deta")
        ax.set_ylabel("dphi")
        ax.set_title(title or "Jet (no valid tokens)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    feats = feats.detach().cpu()
    deta = feats[keep_mask, deta_index].to(torch.float64).numpy()
    dphi = feats[keep_mask, dphi_index].to(torch.float64).numpy()
    pt_frac = feats[keep_mask, pt_frac_index].to(torch.float64).numpy()

    # Use a stable normalization across particle categories.
    pt_min = float(np.nanmin(pt_frac))
    pt_max = float(np.nanmax(pt_frac))
    if not np.isfinite(pt_min) or not np.isfinite(pt_max):
        pt_min, pt_max = 0.0, 1.0
    if pt_min == pt_max:
        pt_max = pt_min + 1e-6
    norm = Normalize(vmin=pt_min, vmax=pt_max)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    scatter_handle = None
    if particle_type_indices is not None:
        valid_feats = feats[keep_mask]
        assigned = np.zeros(len(deta), dtype=bool)
        categories = [
            ("electrons", "electron", electron_marker),
            ("muons", "muon", muon_marker),
            ("photons", "photon", photon_marker),
            ("charged hadrons", "charged_hadron", hadron_marker),
            ("neutral hadrons", "neutral_hadron", hadron_marker),
        ]
        for label, key, marker in categories:
            if key not in particle_type_indices:
                continue
            flag = valid_feats[:, particle_type_indices[key]].to(torch.float64).numpy() > 0.5
            cat_mask = flag & ~assigned
            if not np.any(cat_mask):
                continue
            assigned |= cat_mask
            scatter_handle = ax.scatter(
                deta[cat_mask],
                dphi[cat_mask],
                c=pt_frac[cat_mask],
                cmap=cmap,
                norm=norm,
                s=marker_size,
                alpha=alpha,
                marker=marker,
                linewidths=0.0,
                label=label,
            )

        other_mask = ~assigned
        if np.any(other_mask):
            scatter_handle = ax.scatter(
                deta[other_mask],
                dphi[other_mask],
                c=pt_frac[other_mask],
                cmap=cmap,
                norm=norm,
                s=marker_size,
                alpha=alpha,
                marker=hadron_marker,
                linewidths=0.0,
                label="other",
            )
    elif pdg_id_index is None:
        scatter_handle = ax.scatter(
            deta,
            dphi,
            c=pt_frac,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            alpha=alpha,
            marker=electron_marker,
            linewidths=0.0,
        )
    else:
        pdg = feats[keep_mask, pdg_id_index].detach().cpu().numpy()
        pdg_int = np.rint(pdg).astype(np.int64)
        abs_pdg = np.abs(pdg_int)

        categories = [
            ("electrons", abs_pdg == 11, electron_marker),
            ("muons", abs_pdg == 13, muon_marker),
            ("photons", abs_pdg == 22, photon_marker),
            ("hadrons", (abs_pdg != 11) & (abs_pdg != 13) & (abs_pdg != 22), hadron_marker),
        ]

        for cat_label, cat_mask, marker in categories:
            if not np.any(cat_mask):
                continue
            scatter_handle = ax.scatter(
                deta[cat_mask],
                dphi[cat_mask],
                c=pt_frac[cat_mask],
                cmap=cmap,
                norm=norm,
                s=marker_size,
                alpha=alpha,
                marker=marker,
                linewidths=0.0,
                label=cat_label,
            )

    ax.set_xlabel("deta")
    ax.set_ylabel("dphi")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)
    if pdg_id_index is not None or particle_type_indices is not None:
        ax.legend(loc="best", fontsize=8, frameon=True)

    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, ax=ax)
        cbar.set_label(pt_frac_label)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

