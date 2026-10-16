"""The original (NeurIPS, Cooper) SSAE without LayerNorm, step-L0 and L1 (DEVIATIONS #45): fits for wrappers B, C, D
on the same data as every other new fit, written where the scorers read cached fits.

    python -m experiments.v6_steer.orig_noln A           # gender: bias-in-bios-full train pairs -> fits/bios/<model>/L<L>/
    python -m experiments.v6_steer.orig_noln B           # refusal train shifts -> fits/refusal/<model>/new/<name>.npz
    python -m experiments.v6_steer.orig_noln C 10        # AxBench pairs -> axbench/newfits_<name>_L10.npz
    python -m experiments.v6_steer.orig_noln D <li> ...  # SteeringSafety grid layers: fit + matching (both rules)
    python -m experiments.v6_steer.orig_noln D merge     # -> ss/dirs_new/<name>__<concept>[__cos].pt

Methods: orig_l0_noln (step-L0), orig_l1_noln (L1); steer.newfits.orig_noln (frozen v3 protocol, norm = none, width d).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from steer.common import IT, MODEL_TAG, RECORDS, scratch, write_json
from steer.newfits import ORIG_NOLN, orig_noln


def wrapper_b() -> None:
    from experiments.v6_steer.refusal_vectors import splits
    tr, _ = splits(IT)
    d = scratch() / "fits" / "refusal" / MODEL_TAG[IT] / "new"
    for m in ORIG_NOLN:
        if (d / f"{m}.npz").is_file():
            continue
        E, D, info = orig_noln(tr, m)
        np.savez(d / f"{m}.npz", E=E, D=D)
        write_json(d / f"{m}.json", info)
        print(m, {k: info.get(k) for k in ("updates", "final_violation", "dual_multiplier", "seconds")}, flush=True)


def wrapper_c(L: int) -> None:
    from experiments.v4.data import Split
    Hm = np.load(scratch() / "axbench" / f"pooled_L{L}.npy")
    rng = np.random.default_rng(0)                          # the same pairs as every other AxBench fit
    perm = rng.permutation(len(Hm))
    tr_i = perm[:int(0.8 * len(Hm))]
    a, b = rng.choice(tr_i, 30_000), rng.choice(tr_i, 30_000)
    k = a != b
    X = (Hm[b[k]] - Hm[a[k]]).astype(np.float32)
    tr = Split(X=X, Y=np.zeros((len(X), 1)), pairs=pd.DataFrame({"a": np.arange(len(X))}), factors=["none"],
               ha=None, hb=None, meta={"dataset": "axbench"}, C=None)
    for m in ORIG_NOLN:
        p = scratch() / "axbench" / f"newfits_{m}_L{L}.npz"
        if p.is_file():
            continue
        E, D, info = orig_noln(tr, m)
        np.savez(p, E=E, D=D)
        write_json(RECORDS / f"axbench_{m}__L{L}.json", info)
        print(m, L, {k: info.get(k) for k in ("updates", "final_violation", "dual_multiplier", "seconds")}, flush=True)


def wrapper_d(li: int) -> None:
    from experiments.v6_steer.ss_dirs import CONCEPTS, _Lin, _load, match_set, pooled_pairs
    from experiments.v6_steer.ss_dirs_new import OUT, val_auc
    from steer.vectors import match_latent
    L = _load(CONCEPTS[0])[2][li]
    part = OUT / "parts_orig_noln"
    part.mkdir(parents=True, exist_ok=True)
    if (part / f"L{L}.csv").is_file():
        return
    tr, _, _ = pooled_pairs(li)
    rows, dirs, infos = [], {}, {}
    for m in ORIG_NOLN:
        E, D, info = orig_noln(tr, m)
        infos[m] = info
        f = _Lin(E, D)
        for c in CONCEPTS:
            H, y, dim = match_set(c, li)
            cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
            cm /= np.linalg.norm(cm)
            mm = match_latent(f, H, y)
            v = mm["sign"] * f.atoms[:, mm["latent"]]
            v /= np.linalg.norm(v)
            cs = f.atoms.T @ cm
            jc = int(np.argmax(np.abs(cs)))
            vc = np.sign(cs[jc]) * f.atoms[:, jc]
            dirs[f"{m}__{c}__r"], dirs[f"{m}__{c}__cos"] = v.astype(np.float32), vc.astype(np.float32)
            rows.append({"layer": L, "concept": c, "method": m, "width": E.shape[0], "latent": mm["latent"],
                         "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "cos_rule_cos_dim": float(vc @ dim),
                         "val_auc": val_auc(c, li, v)})
    np.savez(part / f"L{L}.npz", **dirs)
    write_json(part / f"L{L}_info.json", infos)
    pd.DataFrame(rows).to_csv(part / f"L{L}.csv", index=False)
    print(pd.DataFrame(rows)[["concept", "method", "abs_r", "cos_dim"]].round(3).to_string(), flush=True)


def merge_d() -> None:
    from experiments.v6_steer.ss_dirs import CONCEPTS, _load
    from experiments.v6_steer.ss_dirs_new import OUT
    layers = _load(CONCEPTS[0])[2]
    part = OUT / "parts_orig_noln"
    zs = {L: dict(np.load(part / f"L{L}.npz")) for L in layers}
    for k in zs[layers[0]]:
        m, c, rule = k.rsplit("__", 2)
        torch.save({L: torch.tensor(zs[L][k]) for L in layers}, OUT / (f"{m}__{c}.pt" if rule == "r" else f"{m}__{c}__cos.pt"))
    pd.concat([pd.read_csv(part / f"L{L}.csv") for L in layers]).to_csv(RECORDS / "ss_dirs_orig_noln.csv", index=False)
    print("merged", len(zs[layers[0]]))


def wrapper_a() -> None:
    """Steering magnitude item 3 (2026-10-03, #63): the no-LayerNorm originals for the gender wrapper, on the same
    training data as the v5 L0-v3 / L1-v3 fits (bias-in-bios-full train pairs, pronoun read) at the common layer."""
    from experiments.v5.data import load_split
    from steer.common import BASE, read_json
    for model in (BASE, IT):
        L = read_json(RECORDS / "bios_common_layer.json")[model]["layer"]
        d = scratch() / "fits" / "bios" / MODEL_TAG[model] / f"L{L}"
        d.mkdir(parents=True, exist_ok=True)
        tr = None
        for m in ORIG_NOLN:
            if (d / f"{m}.npz").is_file():
                continue
            if tr is None:
                tr = load_split("bias-in-bios-full", "train", L, model)
            E, D, info = orig_noln(tr, m)
            np.savez(d / f"{m}.npz", E=E, D=D, rms=info["rms"])
            write_json(d / f"{m}.json", info)
            print(m, MODEL_TAG[model], L, {k: info.get(k) for k in ("updates", "final_violation", "dual_multiplier", "seconds")},
                  flush=True)


def main() -> None:
    w = sys.argv[1]
    if w == "A":
        wrapper_a()
    elif w == "B":
        wrapper_b()
    elif w == "C":
        for L in sys.argv[2:]:
            wrapper_c(int(L))
    elif w == "D":
        if sys.argv[2] == "merge":
            merge_d()
        else:
            for li in sys.argv[2:]:
                wrapper_d(int(li))


if __name__ == "__main__":
    main()
