"""
Multi-view PCA panel: four perspectives on how the decoder represents
each concept's geometry. Produces one figure per concept.

Views:
  1. PCA of raw delta_z (ground truth manifold structure)
  2. PCA of delta_z projected onto matched decoder subspace
  3. PCA of partial-code reconstruction (matched features only)
  4. Matched feature activations as direct coordinates

Usage:
    python scripts/plots/plot_multiview_pca.py \\
        --data $SCRATCH/synth/curved.h5 \\
        --model $SCRATCH/synth/model.pth --hid 128 \\
        --out figures/multiview/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
except Exception:
    pass
plt.rc("font", family="serif", size=13)

from data.synthetic_manifolds import load_benchmark
from ssae.model import DictLinearAE
from evals.common import load_embeddings
from evals.metrics.geometry import _greedy_r2


def _pca_2d(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.detach().float()
    centered = data - data.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return (centered @ Vh[:2].T).numpy()


def _scatter(ax, xy, colors, title):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="viridis",
               s=4, alpha=0.5, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    sns.despine(ax=ax)


def plot_concept(diffs, codes, W, selected, concept_name, d_ext, out_path):
    """Generate 4-panel figure for one concept."""
    n_cols = min(max(d_ext, 2), len(selected))
    sel = selected[:n_cols]

    raw_2d = _pca_2d(diffs)
    colors = raw_2d[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    _scatter(axes[0, 0], raw_2d, colors, r"Raw $\Delta\mathbf{z}$")

    W_sel = W[:, sel]
    Q, _ = torch.linalg.qr(W_sel)
    proj = (diffs.float() @ Q) @ Q.T
    _scatter(axes[0, 1], _pca_2d(proj), colors,
             f"Decoder subspace (top-{n_cols})")

    codes_sel = codes[:, sel]
    recon = codes_sel @ W[:, sel].T
    _scatter(axes[1, 0], _pca_2d(recon), colors,
             f"Partial reconstruction (top-{n_cols})")

    if codes_sel.shape[1] >= 2:
        feat_xy = codes_sel[:, :2].numpy()
    else:
        feat_xy = np.column_stack([codes_sel[:, 0].numpy(),
                                   np.zeros(len(codes_sel))])
    _scatter(axes[1, 1], feat_xy, colors, "Feature activations")
    axes[1, 1].set_xlabel("Feature 1")
    axes[1, 1].set_ylabel("Feature 2")

    fig.suptitle(f"{concept_name} ($d = {d_ext}$)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view PCA panel for manifold geometry"
    )
    parser.add_argument("--data", required=True, help="HDF5 benchmark file")
    parser.add_argument("--model", required=True, help="Model weights .pth")
    parser.add_argument("--hid", type=int, required=True)
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override columns per concept (default: auto)")
    args = parser.parse_args()

    gt = load_benchmark(args.data)
    test_emb = load_embeddings(Path(args.data), key="cfc_test")
    concept_labels = gt["test_concept_labels"]

    rep_dim = gt["config"]["ambient_dim"]
    model = DictLinearAE(rep_dim, args.hid, norm_type="ln", skip_input_norm=True)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    z = test_emb[0::2].float()
    zt = test_emb[1::2].float()
    diffs = zt - z

    with torch.inference_mode():
        _, codes = model(diffs)

    W = model.decoder.weight.detach().float()
    W_norm = F.normalize(W, dim=0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_concepts = gt["config"]["n_concepts"]
    for ci in range(n_concepts):
        spec = gt["manifold_specs"][ci]
        d_ext = spec["extrinsic_dim"]
        mask = concept_labels == ci

        concept_diffs = diffs[mask]
        concept_codes = codes[mask]

        if args.top_k:
            k = args.top_k
        else:
            k = max(d_ext * 2, 3)

        selected, _ = _greedy_r2(concept_diffs, W, max_atoms=k)

        if not selected:
            diff_norm = F.normalize(concept_diffs, dim=-1)
            cos = (diff_norm @ W_norm).abs().mean(dim=0)
            selected = cos.topk(k).indices.tolist()

        name = f"{spec['type']}_{ci}"
        plot_concept(
            concept_diffs, concept_codes, W, selected,
            name, d_ext,
            out_dir / f"multiview_{name}.png",
        )

    print(f"\nDone: {n_concepts} figures in {out_dir}/")


if __name__ == "__main__":
    main()
