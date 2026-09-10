"""
Synthetic manifold benchmark: generate data, train SAE, validate metrics.

Self-contained script for validating geometry and diagnostics metrics
against planted ground truth. Designed for cluster submission.

Usage:
    python scripts/test_manifold_metrics.py --preset minimal --epochs 300
    python scripts/test_manifold_metrics.py --data path/to/existing.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.synthetic_manifolds import (
    PRESETS, generate_benchmark, load_benchmark, save_benchmark,
)
from ssae.model import DictLinearAE, renorm_decoder_cols_
from ssae.data import SimpleCPUData
from evals.common import load_embeddings
from evals.metrics.geometry import compute_pca_alignment, compute_restricted_r2
from evals.metrics.diagnostics import (
    compute_diagnostics, compute_data_diagnostics, compute_ising_coupling,
    classify_capture_regime,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sparse_ae(
    h5_path: str,
    hid: int,
    epochs: int = 300,
    lr: float = 1e-3,
    l1_coeff: float = 0.05,
    batch_size: int = 256,
    seed: int = 42,
) -> DictLinearAE:
    """Train a DictLinearAE with MSE + L1 penalty on codes."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = SimpleCPUData(h5_path, key="cfc_train")
    rep_dim = dataset.data.shape[-1]

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DictLinearAE(rep_dim, hid, norm_type="ln", skip_input_norm=True).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for delta_z in loader:
            delta_z = delta_z.to(dev)
            x_hat, codes = model(delta_z)

            recon = F.mse_loss(x_hat, delta_z)
            sparsity = codes.abs().mean()
            loss = recon + l1_coeff * sparsity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                renorm_decoder_cols_(model.decoder.weight)

            total_loss += loss.item()
            n_batches += 1

        if ep % 50 == 0 or ep == epochs - 1:
            avg = total_loss / max(n_batches, 1)
            print(f"  epoch {ep:04d}  loss={avg:.4f}")

    return model.cpu()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data_diagnostics(test_emb, gt):
    """Check that data-level diagnostics match expected manifold structure."""
    results = compute_data_diagnostics(test_emb)

    expected_rank = sum(s["extrinsic_dim"] for s in gt["manifold_specs"])

    rank = results["diff_effective_rank"]
    rank_ok = rank >= expected_rank * 0.8

    print(f"\n  Data diagnostics:")
    print(f"    diff_effective_rank = {rank} "
          f"(expected >= {expected_rank}, {'PASS' if rank_ok else 'FAIL'})")
    print(f"    pca_subset_diversity = {results['pca_subset_diversity']:.3f}")
    print(f"    pca_unique_patterns = {results['pca_unique_patterns']}")

    return {"diff_rank": rank, "expected_rank": expected_rank, "pass": rank_ok}


def validate_model_diagnostics(model, test_emb, gt):
    """Check model-level diagnostics: code rank, support diversity."""
    results = compute_diagnostics(model, test_emb)

    n_concepts = gt["config"]["n_concepts"]
    supports = results.get("unique_supports", 0)
    supports_ok = supports >= n_concepts

    max_coact = results.get("max_coactivation", 1.0)
    coact_ok = max_coact < 0.95

    print(f"\n  Model diagnostics:")
    print(f"    code_effective_rank = {results['code_effective_rank']}")
    print(f"    unique_supports = {supports} "
          f"(expected >= {n_concepts}, {'PASS' if supports_ok else 'FAIL'})")
    print(f"    max_coactivation = {max_coact:.3f} "
          f"(expected < 0.95, {'PASS' if coact_ok else 'FAIL'})")

    return {"supports_ok": supports_ok, "coact_ok": coact_ok}


def validate_decoder_alignment(model, test_emb, gt):
    """Check decoder columns align with ground truth embedding directions.

    Uses absolute cosine because synthetic pairs are bidirectional:
    both (theta, theta') and (theta', theta) are equally likely, so
    signed mean cosine collapses to ~0 even for perfect recovery.
    """
    z = test_emb[0::2].float()
    zt = test_emb[1::2].float()
    diffs = F.normalize(zt - z, dim=-1)

    W = model.decoder.weight.detach().float()
    W_norm = F.normalize(W, dim=0)

    cos_all = (diffs @ W_norm).abs()
    per_col = cos_all.mean(dim=0)
    best_idx = per_col.argmax().item()
    best_cos = float(per_col[best_idx])
    cos_ok = best_cos > 0.5

    print(f"\n  Decoder-diff alignment (abs cosine):")
    print(f"    best_col_idx = {best_idx}")
    print(f"    best_abs_cos = {best_cos:.3f} "
          f"({'PASS' if cos_ok else 'FAIL'})")

    return {"cos_mean": best_cos, "pass": cos_ok}


def validate_per_concept_alignment(model, test_emb, concept_labels, gt):
    """Check per-concept decoder alignment against ground truth subspaces."""
    n_concepts = gt["config"]["n_concepts"]
    W = model.decoder.weight.detach().float()
    W_norm = F.normalize(W, dim=0)

    print(f"\n  Per-concept decoder alignment:")

    all_ok = True
    for ci in range(n_concepts):
        mask = concept_labels == ci
        if mask.sum() == 0:
            continue

        z = test_emb[0::2][mask].float()
        zt = test_emb[1::2][mask].float()
        diffs = zt - z
        diff_norm = F.normalize(diffs, dim=-1)

        cos_all = diff_norm @ W_norm
        per_col_mean = cos_all.abs().mean(dim=0)
        best_idx = per_col_mean.argmax().item()
        best_cos = float(per_col_mean[best_idx])

        V_gt = torch.from_numpy(gt["embedding_matrices"][ci]).float()
        V_gt_norm = F.normalize(V_gt, dim=0)
        subspace_cos = (W_norm.T @ V_gt_norm).abs().max(dim=1).values
        best_subspace = float(subspace_cos.max())

        spec = gt["manifold_specs"][ci]
        ok = best_cos > 0.3
        all_ok = all_ok and ok

        print(f"    concept {ci} ({spec['type']}, d={spec['extrinsic_dim']}): "
              f"best_decoder_cos={best_cos:.3f}, "
              f"subspace_alignment={best_subspace:.3f} "
              f"({'PASS' if ok else 'FAIL'})")

    return {"all_pass": all_ok}


def validate_pca(model, test_emb, gt):
    """Check PCA alignment between differences and decoder columns."""
    result = compute_pca_alignment(test_emb, model, n_components=10)
    mean_align = result["mean_alignment"]
    align_ok = mean_align > 0.5

    print(f"\n  PCA alignment:")
    print(f"    mean_alignment = {mean_align:.3f} "
          f"({'PASS' if align_ok else 'FAIL'})")
    top3 = result["max_alignment_per_component"][:3]
    print(f"    top-3 components: {[f'{x:.3f}' for x in top3]}")
    var3 = result["variance_explained"][:3]
    print(f"    variance explained: {[f'{x:.3f}' for x in var3]}")

    return {"mean_alignment": mean_align, "pass": align_ok}


def validate_restricted_r2(model, test_emb, concept_labels, gt):
    """Check that restricted R² saturates at the planted extrinsic dimension."""
    result = compute_restricted_r2(test_emb, model, concept_labels)

    print(f"\n  Restricted R²:")

    all_ok = True
    for ci, spec in enumerate(gt["manifold_specs"]):
        d_ext = spec["extrinsic_dim"]
        info = result["per_concept"][ci]
        curve = info["r2_curve"]

        r2_at_dim = curve[d_ext - 1] if d_ext <= len(curve) else (
            curve[-1] if curve else 0.0
        )
        sat = info["saturation_idx"]
        ok = r2_at_dim > 0.3
        all_ok = all_ok and ok

        curve_str = [f"{x:.2f}" for x in curve[:min(6, len(curve))]]
        print(f"    concept {ci} ({spec['type']}, d={d_ext}): "
              f"R²@d={r2_at_dim:.3f}, sat_k={sat}, "
              f"curve=[{', '.join(curve_str)}]"
              f"  {'PASS' if ok else 'FAIL'}")

    return {"all_pass": all_ok, "mean_saturation": result["mean_saturation_idx"]}


def validate_ising_communities(model, test_emb, concept_labels, gt):
    """Check that Ising coupling communities align with planted concepts.

    Maps features to ground truth concepts by mean activation, then
    compares this assignment to Ising-discovered communities via ARI.
    """
    from sklearn.metrics import adjusted_rand_score

    n_concepts = gt["config"]["n_concepts"]

    z = test_emb[0::2].float()
    zt = test_emb[1::2].float()
    delta_z = zt - z

    model.eval()
    with torch.inference_mode():
        _, codes = model(delta_z)

    ising = compute_ising_coupling(codes, n_communities=n_concepts)

    K = codes.shape[1]
    mean_act = torch.zeros(K, n_concepts)
    for ci in range(n_concepts):
        mask = torch.from_numpy(concept_labels == ci)
        if mask.sum() > 0:
            mean_act[:, ci] = codes[mask].abs().mean(dim=0)

    feature_concepts = mean_act.argmax(dim=1).numpy()

    active = (codes.abs() > 0.01).any(dim=0).numpy()
    feat_idx = ising["feature_indices"]
    active_in_ising = active[feat_idx]

    if active_in_ising.sum() > 1:
        ari = adjusted_rand_score(
            feature_concepts[feat_idx][active_in_ising],
            ising["community_labels"][active_in_ising],
        )
    else:
        ari = 0.0

    ari_ok = ari > 0.2
    block_ok = ising["block_score"] > 0.5

    # Signed within vs cross-concept coupling for diagnosis
    J = ising["coupling_matrix"]
    within_vals, cross_vals = [], []
    for i in range(len(feat_idx)):
        for j in range(i + 1, len(feat_idx)):
            fi, fj = feat_idx[i], feat_idx[j]
            if not (active[fi] and active[fj]):
                continue
            if feature_concepts[fi] == feature_concepts[fj]:
                within_vals.append(J[i, j])
            else:
                cross_vals.append(J[i, j])

    mean_within = float(np.mean(within_vals)) if within_vals else 0.0
    mean_cross = float(np.mean(cross_vals)) if cross_vals else 0.0

    print(f"\n  Ising coupling analysis:")
    print(f"    activation_rate = {ising['activation_rate']:.3f}")
    print(f"    n_dead / n_always_on / n_varying = "
          f"{ising['n_dead']} / {ising['n_always_on']} / {ising['n_varying']}")
    print(f"    n_communities = {ising['n_communities']}")
    print(f"    block_score = {ising['block_score']:.3f} "
          f"(expected > 0.5, {'PASS' if block_ok else 'FAIL'})")
    print(f"    ARI(ising, concept) = {ari:.3f} "
          f"(expected > 0.2, {'PASS' if ari_ok else 'FAIL'})")
    print(f"    signed coupling: within={mean_within:.4f}, "
          f"cross={mean_cross:.4f}")

    return {"ari": ari, "block_score": ising["block_score"],
            "ari_ok": ari_ok, "block_ok": block_ok,
            "coupling_matrix": ising["coupling_matrix"],
            "feature_indices": ising["feature_indices"],
            "feature_concepts": feature_concepts}


def validate_regime(model, test_emb, concept_labels, gt,
                    r2_result, ising_result):
    """Classify each concept into shattering / capture / dilution.

    Synthesises restricted R² saturation, Ising within-concept coupling,
    and L0 sparsity. Informational — not included in hard pass/fail.
    """
    n_concepts = gt["config"]["n_concepts"]

    z = test_emb[0::2].float()
    zt = test_emb[1::2].float()
    delta_z = zt - z

    model.eval()
    with torch.inference_mode():
        _, codes = model(delta_z)

    l0 = float((codes.abs() > 0.01).float().sum(dim=1).mean())

    J = ising_result["coupling_matrix"]
    feat_idx = ising_result["feature_indices"]
    fc = ising_result["feature_concepts"]

    print(f"\n  Regime classification (L0={l0:.1f}):")

    regimes = []
    for ci, spec in enumerate(gt["manifold_specs"]):
        d_ext = spec["extrinsic_dim"]
        sat = r2_result["per_concept"][ci]["saturation_idx"]

        within = []
        for i in range(len(feat_idx)):
            for j in range(i + 1, len(feat_idx)):
                fi, fj = feat_idx[i], feat_idx[j]
                if fc[fi] == ci and fc[fj] == ci:
                    within.append(J[i, j])

        mean_within = float(np.mean(within)) if within else 0.0

        result = classify_capture_regime(l0, sat, d_ext, mean_within)
        regimes.append(result["regime"])

        print(f"    concept {ci} ({spec['type']}, d={d_ext}): "
              f"{result['regime']}  "
              f"[sat={sat}, coupling={mean_within:+.3f}]")

    return {"regimes": regimes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic manifold metric validation")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="minimal")
    parser.add_argument("--data", type=str, help="Load existing HDF5 instead of generating")
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--hid", type=int, default=64,
                        help="Hidden dim (dictionary size)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-data", type=str,
                        help="Save generated data to this path")
    parser.add_argument("--save-model", type=str,
                        help="Save trained model weights to this path")
    args = parser.parse_args()

    # --- Data ---
    if args.data:
        print(f"Loading benchmark from {args.data}")
        gt = load_benchmark(args.data)
        h5_path = args.data
    else:
        manifolds = PRESETS[args.preset]
        print(f"Generating benchmark: preset={args.preset}, manifolds={manifolds}")

        gt = generate_benchmark(
            manifold_types=manifolds,
            ambient_dim=args.ambient_dim,
            n_pairs_per_concept=args.n_train,
            n_test_per_concept=args.n_test,
            seed=args.seed,
        )

        h5_path = args.save_data or "/tmp/synth_manifold_bench.h5"
        save_benchmark(gt, Path(h5_path))
        print(f"Saved to {h5_path}")

    n_concepts = gt["config"]["n_concepts"]
    ambient_dim = gt["config"]["ambient_dim"]
    print(f"\nBenchmark: {n_concepts} concepts in R^{ambient_dim}")
    for i, spec in enumerate(gt["manifold_specs"]):
        print(f"  concept {i}: {spec['type']} "
              f"(d_int={spec['intrinsic_dim']}, d_ext={spec['extrinsic_dim']})")

    # --- Train ---
    print(f"\nTraining DictLinearAE (hid={args.hid}, epochs={args.epochs})...")
    model = train_sparse_ae(
        h5_path, hid=args.hid, epochs=args.epochs,
        lr=args.lr, l1_coeff=args.l1, seed=args.seed,
    )

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    # --- Validate ---
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    test_emb = load_embeddings(Path(h5_path), key="cfc_test")
    concept_labels = gt["test_concept_labels"]

    results = {}
    results["data_diag"] = validate_data_diagnostics(test_emb, gt)
    results["model_diag"] = validate_model_diagnostics(model, test_emb, gt)
    results["decoder_align"] = validate_decoder_alignment(model, test_emb, gt)
    results["per_concept"] = validate_per_concept_alignment(
        model, test_emb, concept_labels, gt
    )
    results["pca"] = validate_pca(model, test_emb, gt)
    results["restricted_r2"] = validate_restricted_r2(
        model, test_emb, concept_labels, gt
    )
    results["ising"] = validate_ising_communities(
        model, test_emb, concept_labels, gt
    )
    results["regime"] = validate_regime(
        model, test_emb, concept_labels, gt,
        results["restricted_r2"], results["ising"],
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    all_pass = all([
        results["data_diag"]["pass"],
        results["model_diag"]["supports_ok"],
        results["model_diag"]["coact_ok"],
        results["decoder_align"]["pass"],
        results["per_concept"]["all_pass"],
        results["pca"]["pass"],
        results["restricted_r2"]["all_pass"],
        results["ising"]["ari_ok"],
        results["ising"]["block_ok"],
    ])
    status = "ALL PASSED" if all_pass else "SOME FAILED"
    print(f"  {status}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
