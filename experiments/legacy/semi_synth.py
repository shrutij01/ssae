#!/usr/bin/env python3
"""
Semi-synthetic experiment runner.

Runs identifiability, similarity, and diagnostics evaluations for
semi-synthetic datasets: eng-french, eng-german, masc-fem, 2-binary, corr-binary.

Usage::

    python -m experiments.legacy.semi_synth --embeddings data.h5 --model path/to/ssae_seed0 \\
        --aggregate-seeds --output results.jsonl
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

from evals.common import (
    EvalResult,
    load_embeddings,
    load_concept_labels,
    load_embeddings_split,
    load_concept_labels_split,
    slice_by_concept,
    get_sae_activations,
    get_diff_activations,
    diff_native_mcc,
    get_concept_acts,
    build_label_matrix,
    summarise_results,
    write_results,
)
from evals.metrics.classification import held_out_concept_classification
from evals.model_loading import load_model
from evals.metrics.identifiability import (
    compute_identifiability_metrics,
    compute_auc,
    compute_probe_mcc,
    compute_probe_auc,
    compute_decoder_projection_mcc,
    compute_support_recovery,
    compute_pairwise_mcc,
    compute_random_baseline,
)
from evals.metrics.geometry import (
    compute_decoder_diff_cosine,
    compute_pca_alignment,
    compute_dictionary_quality,
)
from evals.metrics.diagnostics import (
    compute_diagnostics,
    compute_data_diagnostics,
)
from evals.categorical_factors import (
    CATEGORICAL_CHANGE_NAMES,
    CATEGORICAL_VALUE_NAMES,
    load_aligned_categorical_factors,
    prepare_categorical_codes,
    prepare_categorical_model_inputs,
    route_categorical_model,
    routed_categorical_metric_names,
)
from experiments.legacy.runner import build_model_infos, add_common_args


def _factor_matrix_from_labels(concept_labels):
    """Build an ``(N, C)`` per-pair factor matrix from concept labels for the
    diff-native metric. Returns ``None`` if labels are absent or single-concept.

    Accepts per-pair labels as: dicts ``{concept: truthy}`` (one-hot of truthy
    keys), or scalars/strings (one-hot of unique values). Constant columns are
    dropped; needs >=2 informative factors to be meaningful.
    """
    if not concept_labels:
        return None
    labels = list(concept_labels)
    # Labels are stored in the .h5 as JSON strings (store_embeddings json.dumps
    # each). Parse dict/list labels back so multi-concept structure is recovered
    # (e.g. categorical's {color_changed, shape_changed, object_changed}).
    import json as _json
    parsed = []
    for v in labels:
        if isinstance(v, str) and v[:1] in "{[":
            try:
                parsed.append(_json.loads(v))
            except Exception:
                parsed.append(v)
        else:
            parsed.append(v)
    labels = parsed
    if isinstance(labels[0], (list, tuple)):
        M = np.array([[float(x) for x in row] for row in labels], dtype=np.float64)
        keep = M.std(axis=0) > 1e-8
        M = M[:, keep]
        return M if M.shape[1] >= 2 else None
    if isinstance(labels[0], dict):
        keys = sorted({k for d in labels for k in d})
        M = np.array(
            [[float(bool(d.get(k, 0))) for k in keys] for d in labels],
            dtype=np.float64,
        )
    else:
        uniq = sorted(set(labels), key=lambda v: str(v))
        if len(uniq) < 2:
            return None
        idx = {u: i for i, u in enumerate(uniq)}
        M = np.zeros((len(labels), len(uniq)), dtype=np.float64)
        for i, v in enumerate(labels):
            M[i, idx[v]] = 1.0
    keep = M.std(axis=0) > 1e-8
    M = M[:, keep]
    return M if M.shape[1] >= 2 else None


def _parse_label(v):
    """Parse a single stored per-pair label back into a dict/list/scalar.

    Labels are JSON-dumped at store time (``store_embeddings``); dict/list
    labels arrive as JSON strings and must be parsed to recover structure.
    """
    import json as _json
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return _json.loads(v)
        except Exception:
            return v
    return v


def _aligned_factor_matrices(labels_train, labels_test):
    """Build ``(N, C)`` binary factor matrices for two splits sharing the SAME
    column set/order, so per-concept held-out metrics line up across train/test.

    Columns constant across *both* splits are dropped (uninformative
    everywhere); a column constant in only one split is kept (the downstream
    metrics return NaN for it and it is excluded from the macro mean).

    Returns ``(M_train, M_test, names)`` or ``(None, None, None)`` if fewer
    than two informative concepts remain.
    """
    if not labels_train or not labels_test:
        return None, None, None
    tr = [_parse_label(v) for v in labels_train]
    te = [_parse_label(v) for v in labels_test]

    if isinstance(tr[0], dict):
        keys = sorted({k for d in (tr + te) if isinstance(d, dict) for k in d})
        if len(keys) < 2:
            return None, None, None

        def _mat(rows):
            return np.array(
                [[float(bool(d.get(k, 0))) for k in keys] for d in rows],
                dtype=np.float64,
            )

        M_tr, M_te, names = _mat(tr), _mat(te), list(keys)
    elif isinstance(tr[0], (list, tuple)):
        C = min(len(tr[0]), len(te[0]))
        M_tr = np.array([[float(x) for x in row[:C]] for row in tr], dtype=np.float64)
        M_te = np.array([[float(x) for x in row[:C]] for row in te], dtype=np.float64)
        names = [f"factor_{i}" for i in range(C)]
    else:
        uniq = sorted(set(map(str, tr)) | set(map(str, te)))
        if len(uniq) < 2:
            return None, None, None
        idx = {u: i for i, u in enumerate(uniq)}

        def _mat(rows):
            M = np.zeros((len(rows), len(uniq)), dtype=np.float64)
            for i, v in enumerate(rows):
                M[i, idx[str(v)]] = 1.0
            return M

        M_tr, M_te, names = _mat(tr), _mat(te), list(uniq)

    keep = (M_tr.std(axis=0) > 1e-8) | (M_te.std(axis=0) > 1e-8)
    M_tr, M_te = M_tr[:, keep], M_te[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    if M_tr.shape[1] < 2:
        return None, None, None
    return M_tr, M_te, names


def evaluate_heldout(model, model_info, heldout, args):
    """Evaluate held-out concept classification on diff-native codes.

    Codes use Δz = z̃ - z and a genuine train/test split. The evaluator emits
    per-concept rows plus a NaN-aware macro-average (``concept="all"``) row.
    """
    emb_tr, fac_tr, emb_te, fac_te, names = heldout
    Z_tr = get_diff_activations(model, emb_tr).float().cpu().numpy()
    Z_te = get_diff_activations(model, emb_te).float().cpu().numpy()

    ds = args.dataset_name
    results = []

    def _add(metric, value, concept):
        results.append(EvalResult(
            dataset=ds, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric=metric, value=float(value), concept=concept,
        ))

    # --- Held-out classification (probe AUC/acc, single-feature AUC, leak) ----
    cls = held_out_concept_classification(
        Z_tr, fac_tr, Z_te, fac_te, concept_names=names
    )
    cls_metrics = {
        "heldout_probe_auc": "probe_auc",
        "heldout_probe_acc": "probe_acc",
        "heldout_single_feat_auc": "single_feat_auc",
        "heldout_selectivity_leak": "selectivity_leak",
    }
    for r in cls:
        for out_metric, key in cls_metrics.items():
            _add(out_metric, r[key], r["concept"])
    for out_metric, key in cls_metrics.items():
        _add(out_metric, np.nanmean([r[key] for r in cls]), "all")

    return results


def evaluate_identifiability(
    model,
    model_info,
    embeddings,
    label_matrix,
    concept_data,
    args,
):
    """Run identifiability metrics for one model.

    MCC / DCI / R² are computed via identifiability-guard on the full
    ``(2N, C)`` label matrix and ``(2N, K)`` activation matrix — no
    per-concept slicing.  AUC / probe / support metrics still use the
    per-concept structure from ``concept_data``.
    """
    all_activations = get_sae_activations(model, embeddings)

    results = []
    dataset = args.dataset_name

    # --- MCC, MCC-S, DCI, R² via identifiability-guard ---
    Z_hat = all_activations.float().cpu().numpy()
    ig = compute_identifiability_metrics(label_matrix, Z_hat)
    for metric_name, value in ig.items():
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric=metric_name, value=value,
        ))

    # --- AUC (concept_mapping — id-guard has no AUC) ---
    if args.auc:
        all_acts = get_concept_acts(all_activations, concept_data)
        auc_result = compute_auc(all_acts, concept_data.labels)
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="auc", value=auc_result["auc"],
        ))

    if args.linear_probe:
        probe = compute_probe_mcc(concept_data.embeddings, concept_data.labels)
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="probe_mcc", value=probe["mcc"],
        ))

    if args.auc and args.linear_probe:
        probe_auc = compute_probe_auc(concept_data.embeddings, concept_data.labels)
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="probe_auc", value=probe_auc["auc"],
        ))

    dec = compute_decoder_projection_mcc(
        model, concept_data.embeddings, concept_data.labels
    )
    results.append(EvalResult(
        dataset=dataset, model_name=model_info.name,
        model_type=model_info.model_type, seed=model_info.seed,
        metric="decoder_proj_mcc", value=dec["mcc"],
    ))

    if args.support_recovery:
        all_acts = get_concept_acts(all_activations, concept_data)
        sr = compute_support_recovery(
            all_acts, concept_data.labels,
            concept_names=concept_data.names,
            threshold=args.activation_threshold,
        )
        for metric_name in ("precision", "recall", "f1"):
            results.append(EvalResult(
                dataset=dataset, model_name=model_info.name,
                model_type=model_info.model_type, seed=model_info.seed,
                metric=f"support_{metric_name}", value=sr[metric_name],
            ))

    return results


def evaluate_categorical_identifiability(
    model,
    model_info,
    embeddings,
    aligned_factors,
    *,
    dataset="categorical-contrastive",
):
    """Evaluate a categorical coder on the representation it was trained on.

    Raw SAEs receive every raw embedding and are compared with the 30
    per-example value factors. SSAE and ``*_DIFFS`` SAEs receive ``z_tilde-z``;
    their directionless categorical codes are converted to magnitudes and
    compared with the same three per-pair change factors.
    """
    route = route_categorical_model(model_info.model_type)
    output_names = routed_categorical_metric_names(model_info.model_type)
    model_inputs = prepare_categorical_model_inputs(
        model_info.model_type,
        embeddings,
    )
    codes = prepare_categorical_codes(
        model_info.model_type,
        get_sae_activations(model, model_inputs).float().cpu().numpy(),
    )
    if route == "diff":
        factors = aligned_factors.pair_change_factors
        factor_names = CATEGORICAL_CHANGE_NAMES
        representation = "abs(code(z_tilde-z))"
    else:
        factors = aligned_factors.raw_value_factors
        factor_names = CATEGORICAL_VALUE_NAMES
        representation = "code(raw_z)"

    if factors.shape[0] != codes.shape[0]:
        raise ValueError(
            f"{model_info.name}: routed categorical row mismatch: "
            f"{factors.shape[0]} factor rows vs {codes.shape[0]} code rows"
        )
    constant_factors = [
        factor_names[index]
        for index in np.flatnonzero(np.std(factors, axis=0) <= 1e-8)
    ]
    if constant_factors:
        raise ValueError(
            f"{model_info.name}: routed categorical factors contain constant "
            f"columns: {constant_factors}"
        )

    metrics = compute_identifiability_metrics(
        factors,
        codes,
        methods=("mcc", "mcc_s", "dci"),
        strict=True,
    )
    source_names = ("mcc", "mcc_s", "dci_d")
    extra = {
        "route": route,
        "representation": representation,
        "factor_granularity": "per_pair" if route == "diff" else "per_example",
        "num_factors": int(factors.shape[1]),
        "factor_names": list(factor_names),
        "split": aligned_factors.split_name,
        "alignment_provenance": aligned_factors.provenance,
    }
    return [
        EvalResult(
            dataset=dataset,
            model_name=model_info.name,
            model_type=model_info.model_type,
            seed=model_info.seed,
            metric=output_name,
            value=metrics[source_name],
            extra=extra,
        )
        for source_name, output_name in zip(source_names, output_names)
    ]


def evaluate_similarity(model, model_info, embeddings, args):
    """Run similarity metrics for one model."""
    results = []
    dataset = args.dataset_name

    dd = compute_decoder_diff_cosine(embeddings, model)
    results.append(EvalResult(
        dataset=dataset, model_name=model_info.name,
        model_type=model_info.model_type, seed=model_info.seed,
        metric="decoder_diff_cos", value=dd["cos_mean"],
        extra={"best_col": dd["best_col_idx"], "cos_std": dd["cos_std"]},
    ))

    if args.pca:
        pca = compute_pca_alignment(embeddings, model, n_components=args.pca_components)
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="pca_alignment", value=pca["mean_alignment"],
        ))

    if args.dict_quality:
        dq = compute_dictionary_quality(embeddings, model)
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="dict_angular_error_deg", value=dq["best_angular_error_deg"],
        ))
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="dict_frac_close_0.9", value=dq["frac_close_0.9"],
        ))
        results.append(EvalResult(
            dataset=dataset, model_name=model_info.name,
            model_type=model_info.model_type, seed=model_info.seed,
            metric="dict_norm_ratio", value=dq["best_norm_ratio"],
        ))

    return results


MODEL_DIAG_METRICS = [
    "diff_effective_rank", "code_effective_rank", "rank_ratio",
    "unique_supports", "support_diversity", "max_coactivation",
    "coact_pairs_above_0.9", "coact_pairs_above_0.8",
    "pca_subset_diversity", "pca_unique_patterns",
]


def evaluate_diagnostics(model, model_info, embeddings, args):
    """Run diagnostics for one model."""
    diag = compute_diagnostics(
        model, embeddings,
        max_samples=getattr(args, 'max_samples', 5000),
        batch_size=getattr(args, 'batch_size', 512),
    )

    results = []
    dataset = args.dataset_name

    for metric_name in MODEL_DIAG_METRICS:
        value = diag.get(metric_name)
        if value is not None and not isinstance(value, (list, dict)):
            results.append(EvalResult(
                dataset=dataset,
                model_name=model_info.name,
                model_type=model_info.model_type,
                seed=model_info.seed,
                metric=f"diag_{metric_name}",
                value=float(value),
            ))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run semi-synthetic evaluations (identifiability + similarity + diagnostics)"
    )
    add_common_args(parser)
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--auc", action="store_true")
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--support-recovery", action="store_true")
    parser.add_argument("--activation-threshold", type=float, default=0.1)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--dict-quality", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--heldout-eval", action="store_true",
                        help="Held-out concept classification on a real cfc_test split "
                             "(multi-concept + per-pair labels only)")
    parser.add_argument("--heldout-max-train", type=int, default=2000,
                        help="Cap on held-out probe TRAIN pairs (seeded subsample, shared "
                             "across all models for a fair comparison; test split stays full). "
                             "<=0 uses all train pairs.")
    parser.add_argument(
        "--diff-code-mode",
        choices=("auto", "signed", "magnitude"),
        default="auto",
        help=(
            "Representation used by mcc_diff. 'auto' uses code magnitudes for "
            "binary change-indicator factors and signed codes otherwise. Use "
            "'signed' to reproduce the legacy metric."
        ),
    )
    parser.add_argument(
        "--categorical-source",
        type=Path,
        default=Path("data/categorical/train.json"),
        help=(
            "Source prompt JSON used to recover per-example categorical values "
            "for legacy HDF5 files that do not store prompt text."
        ),
    )
    parser.add_argument(
        "--categorical-data-seed",
        type=int,
        default=42,
        help=(
            "Sampling seed used when categorical embeddings were created "
            "(the legacy YAML incorrectly records 21; the loader used 42)."
        ),
    )
    parser.add_argument(
        "--categorical-num-samples",
        type=int,
        default=None,
        help="Override categorical num_samples from the embedding YAML.",
    )
    parser.add_argument(
        "--categorical-split",
        type=float,
        default=None,
        help="Override the categorical train/test split from the embedding YAML.",
    )

    args = parser.parse_args()
    args.dataset_name = args.dataset or args.embeddings.stem.split("_")[0]

    embeddings = load_embeddings(args.embeddings)
    n_pairs = embeddings.shape[0] // 2
    labels = [0, 1] * n_pairs
    print(f"Loaded {n_pairs} pairs, shape {embeddings.shape}")

    concept_labels = load_concept_labels(args.embeddings)
    concept_data = slice_by_concept(embeddings, labels, concept_labels, args.dataset_name)
    print(f"{len(concept_data.names)} concept(s): {concept_data.names}")

    all_model_infos = build_model_infos(args.model, args.trained_sae, args.aggregate_seeds)

    categorical_factors = None
    if args.dataset_name == "categorical-contrastive":
        categorical_factors = load_aligned_categorical_factors(
            args.embeddings,
            source_path=args.categorical_source,
            data_seed=args.categorical_data_seed,
            num_samples=args.categorical_num_samples,
            split=args.categorical_split,
        )
        print(
            "Categorical routed eval enabled: "
            f"{categorical_factors.raw_value_factors.shape[1]} per-example "
            f"value factors / {categorical_factors.pair_change_factors.shape[1]} "
            f"per-pair change factors ({categorical_factors.split_name})"
        )

    # Label matrix for legacy raw-z identifiability on non-categorical data.
    label_matrix = build_label_matrix(concept_labels, n_pairs)

    all_results = []
    if categorical_factors is None:
        # The legacy random baseline uses pooled base/counterfactual labels and
        # is not comparable to categorical value-level MCC, so do not emit it
        # into routed categorical result files.
        baseline = compute_random_baseline(embeddings, labels)
        print(
            f"Random MCC: {baseline['random_mcc_mean']:.4f} "
            f"+/- {baseline['random_mcc_std']:.4f}"
        )
        all_results.append(EvalResult(
            dataset=args.dataset_name, model_name="random",
            model_type="RANDOM", seed=0,
            metric="mcc", value=baseline["random_mcc_mean"],
        ))

    # Build a per-pair factor matrix from concept labels (multi-concept only).
    diff_factor_matrix = (
        categorical_factors.pair_change_factors
        if categorical_factors is not None
        else _factor_matrix_from_labels(concept_labels)
    )
    if diff_factor_matrix is not None and categorical_factors is None:
        print(
            f"Diff-native eval enabled: {diff_factor_matrix.shape[1]} factors "
            f"(code mode: {args.diff_code_mode})"
        )

    # Held-out concept classification needs a GENUINE train/test
    # split and multi-concept per-pair labels (no fallback — a transductive
    # split would leak train into the "held-out" score). Inert unless
    # --heldout-eval is set.
    heldout = None
    if args.heldout_eval:
        emb_tr = load_embeddings_split(args.embeddings, "cfc_train")
        emb_te = load_embeddings_split(args.embeddings, "cfc_test")
        lab_tr = load_concept_labels_split(args.embeddings, "cfc_train_labels")
        lab_te = load_concept_labels_split(args.embeddings, "cfc_test_labels")
        if emb_tr is not None and emb_te is not None and lab_tr and lab_te:
            fac_tr, fac_te, names = _aligned_factor_matrices(lab_tr, lab_te)
            if fac_tr is not None:
                # Cap TRAIN pairs once (model-independent) to bound probe-fit cost
                # at eval scale. Fixed seed -> every model sees the same subsample,
                # so SSAE-vs-baseline held-out AUC stays a fair comparison.
                cap = args.heldout_max_train
                if cap and 0 < cap < fac_tr.shape[0]:
                    rng = np.random.RandomState(0)
                    pair_idx = np.sort(rng.choice(fac_tr.shape[0], cap, replace=False))
                    flat = np.empty(2 * cap, dtype=int)
                    flat[0::2] = 2 * pair_idx
                    flat[1::2] = 2 * pair_idx + 1
                    emb_tr, fac_tr = emb_tr[flat], fac_tr[pair_idx]
                heldout = (emb_tr, fac_tr, emb_te, fac_te, names)
                print(f"Held-out eval enabled: {fac_tr.shape[0]} train / "
                      f"{fac_te.shape[0]} test pairs, {fac_tr.shape[1]} concepts {names}")
            else:
                print("Held-out eval requested but labels are not multi-concept; skipping.")
        else:
            print("Held-out eval requested but no real cfc_test split/labels; skipping.")

    loaded_models = {}
    for mi in all_model_infos:
        print(f"  {mi.name}...", end=" ", flush=True)
        model = load_model(mi)
        loaded_models[mi.path] = model

        if categorical_factors is not None:
            results = evaluate_categorical_identifiability(
                model,
                mi,
                embeddings,
                categorical_factors,
                dataset=args.dataset_name,
            )
        else:
            results = evaluate_identifiability(
                model,
                mi,
                embeddings,
                label_matrix,
                concept_data,
                args,
            )
        if not args.skip_similarity:
            results += evaluate_similarity(model, mi, embeddings, args)
        if not args.skip_diagnostics:
            results += evaluate_diagnostics(model, mi, embeddings, args)

        # Diff-native identifiability (the metric a difference-trained coder
        # actually optimizes): codes on Δz = z̃ - z vs per-pair factors.
        # Multi-concept only; inert (skipped) when no per-pair concept labels.
        if diff_factor_matrix is not None and categorical_factors is None:
            try:
                dcodes = get_diff_activations(model, embeddings).float().cpu().numpy()
                dm = diff_native_mcc(
                    dcodes,
                    diff_factor_matrix,
                    code_mode=args.diff_code_mode,
                )
                results.append(EvalResult(
                    dataset=args.dataset_name, model_name=mi.name,
                    model_type=mi.model_type, seed=mi.seed,
                    metric="mcc_diff", value=dm["mcc"],
                    extra={"diff_code_mode": dm["code_mode"]},
                ))
            except Exception as e:
                print(f"(diff-native failed: {e})", end=" ")

        # Held-out classification (multi-concept + real test split only).
        if heldout is not None:
            try:
                results += evaluate_heldout(model, mi, heldout, args)
            except Exception as e:
                print(f"(held-out eval failed: {e})", end=" ")

        all_results.extend(results)

        mcc_results = [
            r for r in results if r.metric in ("mcc", "mcc_diff")
        ]
        if mcc_results:
            print(f"MCC={mcc_results[0].value:.4f}")
        else:
            print("done")

    if args.pairwise and len(loaded_models) > 1:
        models_list = [loaded_models[mi.path] for mi in all_model_infos]
        pw = compute_pairwise_mcc(models_list)
        print(f"Pairwise MCC: {pw['mean']:.4f} +/- {pw['std']:.4f}")
        all_results.append(EvalResult(
            dataset=args.dataset_name, model_name="pairwise",
            model_type="PAIRWISE", seed=0,
            metric="decoder_weight_mcc", value=pw["mean"],
        ))

    summarise_results(all_results)

    if args.output:
        write_results(all_results, args.output)


if __name__ == "__main__":
    main()
