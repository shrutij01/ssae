"""
Identifiability metrics for SAE features.

Core metrics (via identifiability-guard):
    - MCC (Pearson/Spearman): Hungarian-matched correlation between
      learned codes and ground-truth factors.
    - DCI: Disentanglement, Completeness, Informativeness.
    - R²: Explicitness (linear predictability of factors from codes).

Additional metrics (in-house):
    - AUC: Hungarian-matched ROC-AUC.
    - Probe MCC/AUC: Linear-probe baselines on raw LLM embeddings.
    - Decoder projection MCC: Projecting embeddings onto decoder columns.
    - Support recovery: Precision/recall/F1 of activation patterns.
    - Pairwise MCC: Representation stability across model seeds.
    - Random baseline: MCC of random projection directions.
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from evals.concept_mapping import (
    ConceptActivations,
    map_concepts,
    prepare_from_activations,
)

# ===========================================================================
# Core identifiability metrics (via identifiability-guard)
# ===========================================================================

def compute_identifiability_metrics(Z, Z_hat, methods=None, *, strict=False):
    """Compute identifiability metrics via identifiability-guard.

    Takes the ground-truth factor matrix and learned codes directly —
    no per-concept slicing needed.

    Args:
        Z:       ``(N, C)`` ground-truth factor matrix (numpy, float64).
        Z_hat:   ``(N, K)`` learned codes / activations (numpy, float64).
        methods: Subset of ``["mcc", "mcc_s", "dci", "r2"]``.
                 Default: all four.
        strict: Re-raise metric failures instead of returning legacy zero
                sentinels. New routed evaluations use this so failed metrics
                cannot be mistaken for genuine zero scores.

    Returns:
        Dict mapping metric names to float values, e.g.
        ``{"mcc": .., "mcc_s": .., "dci_d": .., "dci_c": .., "dci_i": .., "r2": ..}``
    """
    if methods is None:
        methods = ["mcc", "mcc_s", "dci", "r2"]

    Z = np.asarray(Z, dtype=np.float64)
    Z_hat = np.asarray(Z_hat, dtype=np.float64)
    # Constant learned coordinates contain no information and can make
    # correlation implementations return NaN or a scalar-shaped result (common
    # for very wide TopK codes with thousands of dead columns). Removing them
    # leaves the MCC estimand unchanged while making the metric well-defined.
    active_columns = np.nanstd(Z_hat, axis=0) > 1e-12
    all_codes_constant = not active_columns.any()
    if active_columns.any():
        Z_hat = Z_hat[:, active_columns]
    results = {}

    if "mcc" in methods:
        if all_codes_constant:
            results["mcc"] = 0.0
        else:
            try:
                if Z.shape[1] == Z_hat.shape[1] == 1:
                    results["mcc"] = abs(float(np.corrcoef(Z[:, 0], Z_hat[:, 0])[0, 1]))
                else:
                    from identifiability_guard.metrics import MCC

                    r = MCC(method="pearson").compute(Z, Z_hat)
                    results["mcc"] = float(r.primary_score)
            except Exception as e:
                if strict:
                    raise
                print(f"    mcc failed: {e}")
                results["mcc"] = 0.0

    if "mcc_s" in methods:
        if all_codes_constant:
            results["mcc_s"] = 0.0
        else:
            try:
                if Z.shape[1] == Z_hat.shape[1] == 1:
                    from scipy.stats import spearmanr

                    results["mcc_s"] = abs(
                        float(spearmanr(Z[:, 0], Z_hat[:, 0]).statistic)
                    )
                else:
                    from identifiability_guard.metrics import MCC

                    r = MCC(method="spearman").compute(Z, Z_hat)
                    results["mcc_s"] = float(r.primary_score)
            except Exception as e:
                if strict:
                    raise
                print(f"    mcc_s failed: {e}")
                results["mcc_s"] = 0.0

    if "dci" in methods:
        try:
            r = _compute_id_guard_dci(Z, Z_hat, random_state=42)
            results["dci_d"] = float(r["disentanglement"])
            results["dci_c"] = float(r["completeness"])
            results["dci_i"] = float(r["informativeness_test"])
        except Exception as e:
            if strict:
                raise
            print(f"    dci failed: {e}")
            results["dci_d"] = 0.0
            results["dci_c"] = 0.0
            results["dci_i"] = 0.0

    if "r2" in methods:
        try:
            from identifiability_guard.metrics import R2

            r = R2().compute(Z, Z_hat)
            results["r2"] = float(r.primary_score)
        except Exception as e:
            if strict:
                raise
            print(f"    r2 failed: {e}")
            results["r2"] = 0.0

    return results


def _compute_id_guard_dci(Z, Z_hat, *, random_state):
    """Compute every DCI component through identifiability-guard.

    identifiability-guard 0.1.0's public ``DCI`` wrapper returns a vector from
    its one-factor disentanglement special case and then attempts to cast that
    vector to ``float``.  Until that dependency bug is fixed, use the same
    package's GBT/completeness primitives for the one-factor case.  Multi-factor
    evaluations use the public ``DCI`` metric directly.
    """
    from identifiability_guard.metrics import DCI

    if Z.shape[1] > 1:
        result = DCI(
            train_test_split=0.8,
            random_state=random_state,
        ).compute(Z, Z_hat)
        subscores = result.subscores or {}
        required = {
            "disentanglement",
            "completeness",
            "informativeness_test",
        }
        missing = required - set(subscores)
        if missing:
            raise ValueError(f"identifiability-guard DCI missing subscores: {missing}")
        return {name: float(subscores[name]) for name in required}

    from identifiability_guard.metrics.dci import (
        completeness,
        compute_importance_gbt,
    )

    n_samples = Z.shape[0]
    n_train = int(0.8 * n_samples)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    importance, _, test_score = compute_importance_gbt(
        Z_hat[train_idx].T,
        Z[train_idx].T,
        Z_hat[test_idx].T,
        Z[test_idx].T,
    )
    return {
        "disentanglement": 1.0,
        "completeness": float(completeness(importance)),
        "informativeness_test": float(np.clip(test_score, 0.0, 1.0)),
    }


def compute_id_guard_dci_oos(
    factors_train,
    codes_train,
    factors_test,
    codes_test,
    *,
    discrete_factors=None,
    random_state=42,
    n_jobs=1,
):
    """Compute id-guard DCI with an explicit probe train/test split.

    ``identifiability_guard.metrics.DCI.compute_oos`` currently inherits the
    base implementation, which ignores the supplied training split.  This
    adapter calls id-guard's DCI primitives directly so GBT importances are fit
    only on ``train`` and informativeness is evaluated only on ``test``.
    """
    from identifiability_guard.metrics.dci import completeness, disentanglement

    from evals.metrics.dci import compute_importance_gbt

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (factors_train, codes_train, factors_test, codes_test)
    )
    factors_train, codes_train, factors_test, codes_test = arrays
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("DCI factors and codes must all be two-dimensional")
    if factors_train.shape[0] != codes_train.shape[0]:
        raise ValueError("DCI train factors/codes sample counts differ")
    if factors_test.shape[0] != codes_test.shape[0]:
        raise ValueError("DCI test factors/codes sample counts differ")
    if factors_train.shape[1] != factors_test.shape[1]:
        raise ValueError("DCI train/test factor counts differ")
    if codes_train.shape[1] != codes_test.shape[1]:
        raise ValueError("DCI train/test code widths differ")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("DCI factors/codes contain NaN or Inf")

    state = np.random.get_state()
    try:
        np.random.seed(random_state)
        importance, train_score, test_score = compute_importance_gbt(
            codes_train.T,
            factors_train.T,
            codes_test.T,
            factors_test.T,
            discrete_factors=discrete_factors,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    finally:
        np.random.set_state(state)

    n_factors = importance.shape[1]
    dci_d = 1.0 if n_factors <= 1 else float(disentanglement(importance))
    return {
        "disentanglement": dci_d,
        "completeness": float(completeness(importance)),
        "informativeness_train": float(np.clip(train_score, 0.0, 1.0)),
        "informativeness_test": float(np.clip(test_score, 0.0, 1.0)),
    }


# ===========================================================================
# AUC (no id-guard equivalent — kept via concept_mapping)
# ===========================================================================

def compute_auc(all_acts, all_labels, concept_names=None):
    """SAE feature AUC with Hungarian matching."""
    concepts = prepare_from_activations(all_acts, all_labels, names=concept_names)
    result = map_concepts(concepts, method="auc", matching="hungarian")
    return {"auc": result.mean_score, "assignments": result.assignments}


def compute_probe_mcc(all_emb, all_labels, concept_names=None, seed=42):
    """Linear probe MCC: train probes on raw embeddings, score via direct assignment."""
    from scipy.stats import pearsonr
    from sklearn.linear_model import LogisticRegression

    C = len(all_emb)

    probes = []
    for c in range(C):
        X = all_emb[c].float().numpy()
        y = np.array(all_labels[c], dtype=np.float32)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, y)
        probes.append(clf)

    W = np.stack([p.coef_[0] for p in probes])  # (C, D)

    correlations = np.zeros(C)
    for c in range(C):
        emb_np = all_emb[c].float().numpy()
        proj_c = emb_np @ W[c]
        y = np.array(all_labels[c], dtype=np.float32)
        correlations[c] = abs(pearsonr(proj_c, y)[0])

    mcc = float(np.mean(correlations))
    return {"mcc": mcc, "per_concept_correlations": correlations}


def compute_probe_auc(all_emb, all_labels, concept_names=None, seed=42):
    """Linear probe AUC: same as probe MCC but scored by AUC with direct assignment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    C = len(all_emb)

    probes = []
    for c in range(C):
        X = all_emb[c].float().numpy()
        y = np.array(all_labels[c], dtype=np.float32)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, y)
        probes.append(clf)

    W = np.stack([p.coef_[0] for p in probes])

    aucs = np.zeros(C)
    for c in range(C):
        emb_np = all_emb[c].float().numpy()
        proj_c = emb_np @ W[c]
        y = np.array(all_labels[c], dtype=np.float32)
        aucs[c] = roc_auc_score(y, proj_c)

    auc = float(np.mean(aucs))
    return {"auc": auc, "per_concept_aucs": aucs}


def compute_decoder_projection_mcc(model, all_emb, all_labels, concept_names=None, max_cols=None):
    """Decoder projection MCC: project embeddings onto decoder columns, then score."""
    W = model.decoder.weight.detach().float()
    if max_cols is not None:
        W = W[:, :max_cols]
    C = len(all_emb)
    names = concept_names or [f"concept_{i}" for i in range(C)]

    dec_concepts = []
    for c in range(C):
        proj = (all_emb[c].float() @ W).numpy()
        dec_concepts.append(ConceptActivations(
            name=names[c],
            activations=proj.astype(np.float32),
            labels=np.array(all_labels[c], dtype=np.float32),
        ))

    result = map_concepts(dec_concepts, method="pearson", matching="hungarian")
    return {"mcc": result.mean_score}


# ===========================================================================
# Support recovery
# ===========================================================================

def compute_support_recovery(all_acts, all_labels, concept_names=None, threshold=0.1):
    """Measure how well SAE activations recover the true concept support."""
    concepts = prepare_from_activations(all_acts, all_labels, names=concept_names)
    result = map_concepts(concepts, method="pearson", matching="hungarian")

    C = len(all_acts)
    all_precision, all_recall, all_f1 = [], [], []
    all_gt_l0, all_pred_l0 = [], []

    for assignment in result.assignments:
        c_idx = assignment["concept_idx"]
        feat_idx = assignment["feature_idx"]

        acts_c = all_acts[c_idx]
        if isinstance(acts_c, torch.Tensor):
            acts_c = acts_c.float().numpy()
        labels_c = np.array(all_labels[c_idx], dtype=np.float32)

        gt_active = labels_c > 0.5
        pred_active = np.abs(acts_c[:, feat_idx]) > threshold

        tp = float((gt_active & pred_active).sum())
        pred_pos = float(pred_active.sum())
        true_pos = float(gt_active.sum())

        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / true_pos if true_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_gt_l0.append(true_pos / len(labels_c))
        all_pred_l0.append(pred_pos / len(labels_c))

    return {
        "precision": float(np.mean(all_precision)),
        "recall": float(np.mean(all_recall)),
        "f1": float(np.mean(all_f1)),
        "gt_active_frac": float(np.mean(all_gt_l0)),
        "pred_active_frac": float(np.mean(all_pred_l0)),
        "per_concept": [
            {"concept": result.assignments[i].get("concept_name", f"concept_{i}"),
             "precision": all_precision[i], "recall": all_recall[i], "f1": all_f1[i]}
            for i in range(C)
        ],
    }


def compute_pairwise_mcc(models):
    """Pairwise MCC between decoder weight matrices (representation stability)."""
    from evals.metrics.correlation import mean_corr_coef

    weights = []
    for m in models:
        W = m.decoder.weight.detach().float().numpy()
        weights.append(W)

    mccs = []
    for i, j in itertools.combinations(range(len(weights)), 2):
        mcc = float(mean_corr_coef(weights[i], weights[j], method="pearson"))
        mccs.append(mcc)
    return {"mean": float(np.mean(mccs)), "std": float(np.std(mccs)), "values": mccs}


def compute_random_baseline(embeddings, labels, n_random=100, seed=42):
    """Random direction MCC baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    _, D = z.shape
    labels_t = torch.tensor(labels[::2], dtype=torch.float32)

    diffs = z_tilde - z
    diff_norms = F.normalize(diffs, dim=-1)

    random_mccs = []
    random_cos = []

    for _ in range(n_random):
        rand_vec = F.normalize(torch.randn(D), dim=0)

        diff_proj = (z_tilde - z) @ rand_vec
        dc = diff_proj - diff_proj.mean()
        lc = labels_t - labels_t.mean()
        if dc.norm() > 0 and lc.norm() > 0:
            random_mccs.append(abs(float(dc @ lc / (dc.norm() * lc.norm()))))
        else:
            random_mccs.append(0.0)

        random_cos.append(float((diff_norms @ rand_vec).abs().mean()))

    return {
        "random_mcc_mean": float(np.mean(random_mccs)),
        "random_mcc_std": float(np.std(random_mccs)),
        "random_cos_mean": float(np.mean(random_cos)),
    }
