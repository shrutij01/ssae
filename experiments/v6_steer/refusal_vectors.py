"""B.2 — refusal steering vectors at Arditi's selected (layer, position): label-free fits on shifts between random
pairs of the 50 / 50 pool (no labels), activation baselines, SAE latents, supervised DiffMean and probe.

    python -m experiments.v6_steer.refusal_vectors --model google/gemma-2-2b-it

* Shift fits (v4 code, same rules as v5): ``ssae_subspace`` (headline), PCA (r = v4 elbow rule), ICA (5 seeds),
  the original-formulation SSAE (``ssae_linear_l1``), on 20,000 train / 4,000 validation random ordered pairs.
* SAE latents: the v5 pooled activation SAEs (``*_p2``: the pool of the activation PCA / ICA) and every GemmaScope
  16k L0 variant at the matched layer; pair code = code(b) - code(a) with the v4 sign merge (candidates active on
  >= 1% of train pairs).
* Activation PCA / ICA (addendum #3) on the same v5 pool, r = the refusal shift PCA r, plus the elbow row.
* Matching (every label-free method, one rule): the 64 labelled calibration prompts (32 harmful, 32 harmless
  validation instructions), ``steer.vectors.match_latent``; +v = toward harmful / refusal. For GemmaScope the rule
  runs over the latents of all L0 variants together.
* Supervised: DiffMean = Arditi's selected candidate (mean over the filtered 128 / 128 train instructions); probe =
  logistic regression on the same filtered train activations.
* s = median norm of the train pool's cross-label shifts (one harmful and one harmless instruction).
"""

from __future__ import annotations

import argparse
import functools
import json
import time

import numpy as np
import pandas as pd

from steer.common import IT, MODEL_TAG, RECORDS, V5_LAYERS, assert_finite, read_json, scratch, write_json
from steer.vectors import SteerVec, all_pairs, get_vector, match_latent

ROOT = scratch() / "refusal"
N_TRAIN_PAIRS, N_VAL_PAIRS = 20_000, 4_000
SHIFT = ("ssae_subspace", "pca", "ica", "ica_s1", "ica_s2", "ica_s3", "ica_s4", "L1-v3")
SAE = ("sae_topk", "sae_relu", "sae_jumprelu", "gemmascope")
ACT = ("act_pca", "act_pca_elbow", "act_ica", "act_ica_s1", "act_ica_s2", "act_ica_s3", "act_ica_s4", "act_ica_elbow")
SUP = ("diffmean", "probe")
NULL = ("random0", "random1", "random2")
# wrapper name -> (matching rule, calibration budget). "refusal" is the preregistered primary (DEVIATIONS #19).
VARIANTS = {"refusal": ("r", 64), "refusal_cos": ("cos", 64), "refusal_r256": ("r", 256), "refusal_cos256": ("cos", 256)}
SSAE_SEEDS = ("ssae_subspace_s1", "ssae_subspace_s2", "ssae_subspace_s3", "ssae_subspace_s4")
# new fits (DEVIATIONS #22): full-width / elbow SSAE, all-component PCA, FastICA at every candidate width and seed,
# random basis, step-L0 original SSAE
NEW = ("ssae_bf", "ssae_full", "pca_full", "act_pca_full", "rand_basis", "L0-v3")          # elbow SSAE dropped (#25)


def ica_names(prefix: str, widths=None, seeds=range(5)) -> list[str]:
    from steer.newfits import ICA_WIDTHS
    return [f"{prefix}_w{w}_s{k}" for w in (widths or ICA_WIDTHS) for k in seeds]


def new_fit(model: str, method: str):
    """Loaded-like fit for a new-fit method name (cached under fits/refusal/<model>/new/)."""
    from steer import newfits as NF
    d = _fits_dir(model) / "new"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{method}.npz"
    L, _ = site()
    if not p.is_file():
        tr, va = splits(model)
        if method == "ssae_full":
            E, D, info = NF.ssae(tr.X, va.X, r=tr.X.shape[1])
        elif method in ("ssae_A", "ssae_B"):          # constrained brief step 4 (DEVIATIONS #30)
            from steer.constrained import chosen, fit as cfit
            v = method[-1]
            E, D, info = cfit(tr.X, v, chosen(v)["candidate"], seed=0)
            info["width"] = E.shape[0]
        elif method == "ssae_A2":                       # SSAE-A (v2): fitted only by fit_v2 (DEVIATIONS #43)
            raise FileNotFoundError(f"{p} missing: run experiments.v6_steer.fit_v2 B first")
        elif method in ("orig_l0_noln", "orig_l1_noln"):  # original SSAE without LayerNorm (DEVIATIONS #45)
            raise FileNotFoundError(f"{p} missing: run experiments.v6_steer.orig_noln B first")
        elif method == "ssae_bf":                       # SSAE (budget-free): Adam to convergence (DEVIATIONS #31)
            from steer.ssae_bf import solve
            out = solve(tr.X, seed=0)
            E, D = out["E"], out["D"]
            info = {"width": out["r"], "objective": out["objective"], "best_restart": out["best_restart"],
                    "best_is_pca_start": out["best_is_pca_start"], "restart_objectives": out["restart_objectives"],
                    "restart_stats": [{k: v for k, v in st.items() if k != "trace"} for st in out["restart_stats"]],
                    "seconds": out["seconds"], "optimizer": out["optimizer"]}
        elif method == "ssae_elbow":
            E, D, info = NF.ssae(tr.X, va.X, r=None)
        elif method == "pca_full":
            E, D, info = NF.pca_shifts(tr.X)
        elif method == "act_pca_full":
            from experiments.v6_steer import act_baselines as AB
            E, D, info = NF.pca_acts(AB.centred(model, L)[0])
        elif method == "rand_basis":
            E, D, info = NF.rand_basis(tr.X.shape[1])
        elif method.startswith("ica_w"):
            w, k = method[len("ica_w"):].split("_s")
            E, D, info = NF.ica_shifts(tr.X, int(w), int(k))
        elif method.startswith("act_ica_w"):
            from experiments.v6_steer import act_baselines as AB
            w, k = method[len("act_ica_w"):].split("_s")
            E, D, info = NF.ica_acts(AB.centred(model, L)[0], int(w), int(k))
        elif method == "L0-v3":
            W_enc, W_dec, info = NF.step_l0(tr)
            np.savez(p, W_enc=W_enc, W_dec=W_dec, rms=info["rms"])
            write_json(p.with_suffix(".json"), {k: v for k, v in info.items() if k != "trajectories"})
            if "trajectories" in info:
                write_json(p.with_name(p.stem + "_trajectories.json"), info["trajectories"])
        else:
            raise ValueError(method)
        if method != "L0-v3":
            np.savez(p, E=E, D=D)
            write_json(p.with_suffix(".json"), {k: v for k, v in info.items() if k != "trajectories"})
            if "trajectories" in info:
                write_json(p.with_name(p.stem + "_trajectories.json"), info["trajectories"])
    z = np.load(p)
    assert_finite(str(p), dict(z))
    if method == "L0-v3":
        from experiments.v5.models import encode_state

        class _V3:
            kind = "v3ln"
            atoms = z["W_dec"] / np.linalg.norm(z["W_dec"], axis=0, keepdims=True)
            state = {"W_enc": z["W_enc"], "rms": np.array(float(z["rms"])), "ln_weight": None, "D_raw": z["W_dec"]}

            def encode(self, split):
                return encode_state("v3ln", self.state, np.asarray(split.X, dtype=np.float32))
        return _V3()
    return _Lin(z["E"], z["D"])


@functools.lru_cache(maxsize=None)
def direction_meta() -> dict:
    return read_json(RECORDS / f"refusal_direction__{MODEL_TAG[IT]}.json")


def site() -> tuple[int, int]:
    m = direction_meta()
    return int(m["layer"]), int(m["pos"])


@functools.lru_cache(maxsize=8)
def acts(model: str, group: str) -> np.ndarray:
    """(n, d) activations of a group at the selected (layer, position)."""
    layer, pos = site()
    A = np.load(ROOT / f"acts__{MODEL_TAG[model]}__{group}.npy", mmap_mode="r")
    return np.asarray(A[:, V5_LAYERS.index(layer), pos], dtype=np.float32)


@functools.lru_cache(maxsize=2)
def splits(model: str):
    """Train / validation Split objects of random ordered pairs of the unlabelled pool."""
    from experiments.v4.data import Split

    sets = read_json(RECORDS / "refusal_sets.json")
    lab = np.array([p["harmful"] for p in sets["pool"]], dtype=float)
    spl = np.array([p["split"] for p in sets["pool"]])
    H = acts(model, "pool")
    out = {}
    for name, n, seed in (("train", N_TRAIN_PAIRS, 11), ("val", N_VAL_PAIRS, 12)):
        idx = np.flatnonzero(spl == name)
        rng = np.random.default_rng(seed)
        a, b = rng.choice(idx, n), rng.choice(idx, n)
        keep = a != b
        a, b = a[keep], b[keep]
        pairs = pd.DataFrame({"a": a, "b": b, "harm_delta": lab[b] - lab[a]})
        out[name] = Split(X=H[b] - H[a], Y=(lab[b] - lab[a])[:, None], pairs=pairs, factors=["harm"], ha=H[a],
                          hb=H[b], meta={"dataset": "refusal-v6", "split": name}, C=None)
    return out["train"], out["val"]


def _fits_dir(model: str):
    return scratch() / "fits" / "refusal" / MODEL_TAG[model]


def shift_r(model: str) -> int:
    from ssae.selection import select_r
    p = _fits_dir(model) / "r.json"
    if p.is_file():
        return read_json(p)["r"]
    tr, va = splits(model)
    r, curve = select_r(tr.X.astype(np.float64), va.X.astype(np.float64))
    write_json(p, {"r": int(r), "curve": curve.to_dict()})
    return int(r)


class _Lin:
    """Loaded-like linear fit: pair codes X @ E.T, unit atoms."""

    kind = "linear"

    def __init__(self, E, D):
        self.E, self.D = np.asarray(E, dtype=np.float64), np.asarray(D, dtype=np.float64)
        self.atoms = self.D / np.linalg.norm(self.D, axis=0, keepdims=True)
        self.state = {"D_raw": self.D}

    def encode(self, split):
        return np.asarray(split.X, dtype=np.float64) @ self.E.T


class _Wrap:
    """A v4 Fit (from experiments.v4.methods) as a Loaded-like object."""

    def __init__(self, fit, kind):
        self.fit, self.kind, self.atoms, self.state = fit, kind, fit.atoms, dict(fit.state)

    def encode(self, split):
        return np.asarray(self.fit.encode(split), dtype=np.float64)


def shift_fit(model: str, method: str):
    from experiments.v4 import methods as M
    d = _fits_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    tr, va = splits(model)
    r = shift_r(model)
    if method.startswith("ssae_subspace"):
        seed = int(method.split("_s")[-1]) if method != "ssae_subspace" else 0
        p = d / (f"ssae_subspace_s{seed}.npz" if seed else "ssae_subspace.npz")
        if not p.is_file():
            f, _ = M.fit_ssae_subspace(tr, va, seed=seed)
            np.savez(p, E=f.state["E"], D=f.state["D"])
            write_json(p.with_suffix(".json"), {k: v for k, v in f.info.items() if k in ("r", "tau", "objective",
                                                                                        "val_error")})
        z = assert_finite(str(p), dict(np.load(p)))
        return _Lin(z["E"], z["D"])
    if method == "pca":
        f = M.fit_pca(tr, r)
        return _Lin(f.state["U"].T, f.state["U"])
    if method.startswith("ica"):
        seed = int(method.split("_s")[1]) if "_s" in method else 0
        p = d / f"ica_s{seed}.npz"
        if not p.is_file():
            f = M.fit_ica(tr, r, seed)
            np.savez(p, E=f.state["E"], D=f.state["D"])
        z = assert_finite(str(p), dict(np.load(p)))
        return _Lin(z["E"], z["D"])
    if method == "L1-v3":
        p = d / "L1-v3.npz"
        if not p.is_file():
            f = M.fit_ssae_linear("ssae_linear_l1", tr, seed=0)
            np.savez(p, W_enc=f.state["W_enc"], W_dec=f.state["W_dec"], rms=f.info["rms"])
            write_json(p.with_suffix(".json"), {k: v for k, v in f.info.items() if k != "protocol"})
        z = assert_finite(str(p), dict(np.load(p)))
        from experiments.v5.models import encode_state

        class _V3:
            kind = "v3ln"
            atoms = z["W_dec"] / np.linalg.norm(z["W_dec"], axis=0, keepdims=True)
            state = {"W_enc": z["W_enc"], "rms": np.array(float(z["rms"])), "ln_weight": None, "D_raw": z["W_dec"]}

            def encode(self, split):
                return encode_state("v3ln", self.state, np.asarray(split.X, dtype=np.float32))
        return _V3()
    raise ValueError(method)


def sae_fits(model: str, method: str) -> list:
    """Activation SAE (v5 pooled *_p2) or every GemmaScope variant, as v4 fits on the train pairs."""
    from experiments.v4 import methods as M
    from experiments.v5.act_sae import path as sae_path
    tr, _ = splits(model)
    layer, _ = site()
    if method == "gemmascope":
        return [_Wrap(M.fit_gemmascope(tr, layer, l0), "gemmascope") for l0 in M.gemmascope_variants(layer)]
    from experiments.v4.sae_baselines import load
    tsae = load(sae_path(model, layer, method[len("sae_"):], 2304))
    W = tsae.decoder()
    z = tsae.codes(tr.hb) - tsae.codes(tr.ha)
    plus, minus = M.sign_merge(z, W)
    merged = M.apply_merge(z, plus, minus)
    keep = np.flatnonzero((merged != 0).mean(0) >= 0.01)
    pk, mk = plus[keep], minus[keep]

    class _S:
        kind = "sae_act"
        atoms = W[:, pk] / np.linalg.norm(W[:, pk], axis=0, keepdims=True)
        state = {"D_raw": W[:, pk]}

        def encode(self, split):
            return M.apply_merge(tsae.codes(split.hb) - tsae.codes(split.ha), pk, mk)
    return [_S()]


def calibration_set(model: str, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """64: Arditi's 32 + 32 validation instructions (preregistered). 256: 128 + 128 (DEVIATIONS #20; groups
    cal256_harmful / cal256_harmless from ``refusal_data.py --cal256``)."""
    gh, gl = ("cal_harmful", "cal_harmless") if n == 64 else ("cal256_harmful", "cal256_harmless")
    H = np.concatenate([acts(model, gh), acts(model, gl)])
    y = np.r_[np.ones(len(acts(model, gh))), np.zeros(len(acts(model, gl)))]
    return H, y


@functools.lru_cache(maxsize=None)
def scale(model: str) -> float:
    tr, _ = splits(model)
    cross = tr.pairs["harm_delta"].to_numpy() != 0
    return float(np.median(np.linalg.norm(tr.X[cross], axis=1)))


@functools.lru_cache(maxsize=None)
def diffmean(model: str) -> np.ndarray:
    Hh, Hl = acts(model, "harmful_train"), acts(model, "harmless_train")
    r = Hh.astype(np.float64).mean(0) - Hl.astype(np.float64).mean(0)
    return r / np.linalg.norm(r)


def refusal_vector(method: str, model: str, layer: int, wrapper: str = "refusal") -> SteerVec:
    rule, n_cal = VARIANTS[wrapper]
    L, pos = site()
    assert layer == L, (layer, L)
    s = scale(model)
    ref = diffmean(model)
    if method.startswith("random"):
        k = int(method[len("random"):])
        v = np.random.default_rng(2000 + k).standard_normal(ref.shape[0])
        return SteerVec(method, wrapper, model, L, None, v / np.linalg.norm(v), s, None, {"pos": pos})
    H, y = calibration_set(model, n_cal)
    a, b = all_pairs(len(H))
    if method in ("diffmean", "probe"):
        if method == "diffmean":
            v, info = ref.copy(), {}
        else:
            from steer.probe import fit_logistic
            Hh, Hl = acts(model, "harmful_train"), acts(model, "harmless_train")
            res = fit_logistic(np.concatenate([Hh, Hl]), np.r_[np.ones(len(Hh)), np.zeros(len(Hl))])
            v, info = res["v"], {k: x for k, x in res.items() if k != "v"}
        info.update({"pos": pos, "calib_r": float(np.corrcoef((H[b] - H[a]) @ v, y[b] - y[a])[0, 1]),
                     "cos_diffmean": float(v @ ref)})
        return SteerVec(method, wrapper, model, L, None, v, s, None, info)
    if method in NEW + ("ssae_A", "ssae_B", "ssae_A2", "orig_l0_noln", "orig_l1_noln") or method.startswith(("ica_w", "act_ica_w")):
        fits = [new_fit(model, method)]
    elif method.startswith("act_"):
        from experiments.v6_steer import act_baselines as AB
        kind = "pca" if method.startswith("act_pca") else "ica"
        r = AB.elbow_r(model, L) if method.endswith("_elbow") else shift_r(model)
        seed = int(method.rsplit("_s", 1)[1]) if "_s" in method else 0
        fits = [AB.fit(model, L, kind, r, seed)]
    elif method in SAE:
        fits = sae_fits(model, method)
    else:
        fits = [shift_fit(model, method)]
    # rule "r" (preregistered): largest |Pearson r| of the pair code with the label delta over all ordered pairs of
    # the 64 calibration prompts. Rule "cos" (DEVIATIONS #17): largest |cos| of the unit atom with the calibration
    # prompts' mean difference (same 64 labelled prompts), sign = sign(cos). Both picks are recorded.
    c = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
    c /= np.linalg.norm(c)
    best_r, best_c = None, None
    for fi, f in enumerate(fits):
        m = match_latent(f, H, y)
        if best_r is None or m["abs_r"] > best_r[1]["abs_r"]:
            best_r = (fi, m, f)
        A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
        cs = A.T @ c
        j = int(np.argmax(np.abs(cs)))
        if best_c is None or abs(cs[j]) > best_c[1]["abs_cos"]:
            best_c = (fi, {"latent": j, "abs_cos": float(abs(cs[j])), "sign": float(np.sign(cs[j]) or 1.0)}, f)
    fi_r, m_r, f_r = best_r
    v_r = m_r["sign"] * f_r.atoms[:, m_r["latent"]] / np.linalg.norm(f_r.atoms[:, m_r["latent"]])
    fi_c, m_c, f_c = best_c
    v_c = m_c["sign"] * f_c.atoms[:, m_c["latent"]] / np.linalg.norm(f_c.atoms[:, m_c["latent"]])
    # top-5 signed atoms of the chosen fit, in the rule's own ranking (for the top-k ablations, E2)
    fi, f = (fi_r, f_r) if rule == "r" else (fi_c, f_c)
    A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
    if rule == "r":
        from experiments.v4.data import Split
        import pandas as pd
        dlt = (y[b] - y[a]).astype(float)
        sp = Split(X=(H[b] - H[a]).astype(np.float32), Y=dlt[:, None], pairs=pd.DataFrame({"a": a, "b": b}),
                   factors=["c"], ha=H[a], hb=H[b], meta={}, C=None)
        from steer.vectors import _pearson_cols
        sc = _pearson_cols(np.asarray(f.encode(sp), dtype=np.float64), dlt)
    else:
        sc = A.T @ c
    order = np.argsort(-np.abs(sc))[:5]
    top5 = (A[:, order] * np.sign(sc[order])[None, :]).astype(np.float64)
    info = {"pos": pos, "rule": rule, "n_cal": n_cal, **{f"r_rule_{k}": x for k, x in m_r.items()},
            "r_rule_variant": fi_r, "r_rule_cos_diffmean": float(v_r @ ref), "cos_rule_latent": m_c["latent"],
            "cos_rule_abs_cos_calib": m_c["abs_cos"], "cos_rule_variant": fi_c, "cos_rule_cos_diffmean": float(v_c @ ref),
            "same_atom": bool(fi_r == fi_c and m_r["latent"] == m_c["latent"]), "top5": order.tolist(),
            "top5_cos_diffmean": (top5.T @ ref).tolist()}
    v = v_c if rule == "cos" else v_r
    Q, _ = np.linalg.qr(top5)
    info.update({"latent": m_c["latent"] if rule == "cos" else m_r["latent"], "variant": fi, "abs_r": m_r["abs_r"],
                 "cos_diffmean": float(v @ ref), "span2_diffmean": float(np.linalg.norm(Q[:, :2].T @ ref)),
                 "span5_diffmean": float(np.linalg.norm(Q.T @ ref))})
    return SteerVec(method, wrapper, model, L, None, v, s, top5, info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=IT)
    ap.add_argument("--wrapper", default="refusal", choices=sorted(VARIANTS))
    ap.add_argument("--new", action="store_true", help="the new fits only (DEVIATIONS #22)")
    ap.add_argument("--constrained", action="store_true", help="SSAE-A / SSAE-B only (DEVIATIONS #30)")
    ap.add_argument("--methods", nargs="+", default=None, help="explicit method list (e.g. ssae_A2, #43)")
    args = ap.parse_args()
    L, pos = site()
    rows = []
    methods = tuple(args.methods) if args.methods else ("ssae_A", "ssae_B") if args.constrained else \
        (NEW + tuple(ica_names("ica")) + tuple(ica_names("act_ica"))) if args.new else \
        SHIFT + SSAE_SEEDS + SAE + ACT + SUP + NULL
    for method in methods:
        t0 = time.time()
        try:
            v = get_vector(method, args.wrapper, args.model, L)
            rows.append({"model": args.model, "method": method, "layer": L, "pos": pos, "s": v.s,
                         **{k: v.info.get(k) for k in ("rule", "latent", "abs_r", "sign", "runner_up_abs_r", "n_latents",
                                                       "variant", "cos_diffmean", "calib_r", "r_rule_latent",
                                                       "r_rule_cos_diffmean", "cos_rule_latent",
                                                       "cos_rule_abs_cos_calib", "cos_rule_cos_diffmean", "same_atom",
                                                       "span2_diffmean", "span5_diffmean", "n_cal")},
                         "seconds": time.time() - t0})
        except Exception as e:
            rows.append({"model": args.model, "method": method, "error": f"{type(e).__name__}: {e}"[:300]})
        print(rows[-1], flush=True)
    suffix = ("" if args.wrapper == "refusal" else f"__{args.wrapper}") + ("__new" if args.new else "") + \
        ("__constrained" if args.constrained else "") + ("__" + "_".join(args.methods) if args.methods else "")
    pd.DataFrame(rows).to_csv(RECORDS / f"refusal_vectors__{MODEL_TAG[args.model]}{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
