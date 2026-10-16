"""SAE sanity checks for the manifest (user request 2026-10-05; CPU, no new fit or score).

    python -m experiments.v6_ravel.sae_sanity

For every activation SAE behind a RAVEL row: our paper-recipe SAEs (``sae.py``: Gemma-2-2B five types x L1 coefficient
{0.1, 0.3, 1.0}; Llama2-7B city) and GemmaScope 16k (canonical, the type's layer). Per SAE:
  dead fraction   latents never nonzero on the fit's own training pool (the user's validity rule (b): > 50% = invalid)
  FVU             mean squared reconstruction error / mean squared deviation from the pool mean (sae.py's val_fvu),
                  on the training pool and on the validation split
  L0              mean number of nonzero latents per activation (validation split)
Our SAEs: training pool = the 95% train split of the type's Wikipedia activations, exactly as ``sae.stage_train``
drew it (randperm seed 0, n_val = n // 20); the recorded val_fvu is recomputed as a check. GemmaScope: its own
training pool is not available, so its numbers are on the same Wikipedia activations and split. Writes
results/v6_ravel/audit/sae_sanity.{md,csv}.
"""

from __future__ import annotations

import csv
import time

import torch

from experiments.v6_ravel import sae as SA
from experiments.v6_ravel.common import ENTITY_SHORT, ENTITY_TYPES, RESULTS, layers_for, load_acts, model_dir, read_json

OUT = RESULTS / "audit"
CHUNK = 2048


def split(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, n // 20)
    return perm[n_val:], perm[:n_val]


def stats(P: dict, X: torch.Tensor) -> dict:
    m = P["W1"].shape[0]
    active = torch.zeros(m, dtype=torch.bool)
    se, nz = 0.0, 0.0
    var = float(((X - X.mean(0)) ** 2).mean())
    with torch.no_grad():
        for i in range(0, len(X), CHUNK):
            x = X[i:i + CHUNK]
            z = SA.encode(P, x)
            on = z > 0
            active |= on.any(0)
            nz += float(on.sum())
            se += float(((z @ P["W2"].T + P["b2"] - x) ** 2).sum())
    return {"dead_frac": 1.0 - float(active.float().mean()), "fvu": se / (X.numel() * var), "l0": nz / len(X),
            "n": int(len(X)), "m": int(m)}


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = []
    for model in ("gemma2_2b", "llama2_7b"):
        for e in ENTITY_TYPES:
            saes = [(f"sae_reg{r}", lambda r=r: SA.sae_params(model, e, r)) for r in SA.REGS
                    if SA.sae_path(model, e, r).exists()]
            if model == "gemma2_2b":
                saes.append(("gemmascope", lambda: SA.gemmascope_params(layers_for(model)[e])))
            if not saes:
                continue
            X = load_acts(SA.wiki_acts_path(model, e))["X"].float()
            tr, va = split(len(X))
            for name, get in saes:
                t0 = time.time()
                P = get()
                a, b = stats(P, X[tr]), stats(P, X[va])
                rec = None
                if name != "gemmascope":
                    rec = read_json(model_dir(model) / "fits" / e / f"{name}.json").get("val_fvu")
                rows.append({"model": model, "type": e, "sae": name, "kind": P["kind"], "layer": layers_for(model)[e],
                             "width": a["m"], "pool": "own training pool (95% train split of the type's Wikipedia "
                             "activations)" if name != "gemmascope" else "our Wikipedia activations, same split "
                             "(GemmaScope's own pool not available)",
                             "n_train_pool": a["n"], "n_val": b["n"], "dead_frac": a["dead_frac"],
                             "dead_check": dead_check(name, a["dead_frac"]),
                             "fvu_train": a["fvu"], "fvu_val": b["fvu"], "fvu_val_recorded": rec,
                             "fvu_val_reproduced": ("n/a" if rec is None else
                                                    "pass" if abs(rec - b["fvu"]) < 1e-3 else "FAIL"),
                             "l0_val": b["l0"], "seconds": time.time() - t0})
                print(f"[sae_sanity] {model} {e} {name} dead {a['dead_frac']:.3f} fvu {a['fvu']:.4f}/{b['fvu']:.4f} "
                      f"(rec {rec}) L0 {b['l0']:.1f} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sae_sanity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_md()


def dead_check(name: str, dead: float) -> str:
    """Validity rule (b) is defined on the fit's own training pool; GemmaScope's is not available (flag only)."""
    if name == "gemmascope":
        return ("below 50% on our pool" if dead <= 0.5 else
                "FLAG: > 50% never active on our pool (rule (b) is defined on the fit's own training pool, which is "
                "not available)")
    return "pass" if dead <= 0.5 else "FAIL (> 50% never active on its own training pool: invalid by rule (b))"


def write_md() -> None:
    """sae_sanity.md (and the csv's check column) from sae_sanity.csv, no recomputation."""
    from experiments.v6_ravel.status import mark
    rows = list(csv.DictReader(open(OUT / "sae_sanity.csv")))
    for r in rows:
        r["dead_check"] = dead_check(r["sae"], float(r["dead_frac"]))
    with open(OUT / "sae_sanity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    f = lambda x: None if x in ("", None) else float(x)
    L = ["# SAE sanity checks: dead latents, FVU, L0 (user request 2026-10-05; CPU, stored checkpoints)", "",
         "Dead = share of latents never nonzero on the pool (validity rule (b): more than half dead on the fit's own "
         "training pool = invalid). FVU = mean squared reconstruction error / mean squared deviation from the pool "
         "mean (as `sae.py`'s recorded val_fvu). Our SAEs: own training pool = the 95% train split of the type's "
         "Wikipedia activations, as `sae.stage_train` drew it. GemmaScope: the same activations and split; its own "
         "training pool is not available, so its dead share is a flag, not a rule-(b) verdict. GemmaScope is a "
         "JumpReLU SAE (pretrained; thresholds trained by its authors).", "",
         "| Model | Type | SAE | Width | Layer | Dead (pool) | Dead check | FVU train | FVU val | Val FVU recorded "
         "| Reproduced | L0 (val) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rec = f(r["fvu_val_recorded"])
        L.append(f"| {r['model']} | {ENTITY_SHORT[r['type']]} | {r['sae']} | {int(r['width']):,} | {r['layer']} | "
                 f"{100 * float(r['dead_frac']):.1f}% | {r['dead_check']} | {float(r['fvu_train']):.4f} | "
                 f"{float(r['fvu_val']):.4f} | {'–' if rec is None else f'{rec:.4f}'} | {r['fvu_val_reproduced']} | "
                 f"{float(r['l0_val']):.1f} |")
    (OUT / "sae_sanity.md").write_text("\n".join(L) + "\n")
    mark("audit: SAE dead latents and FVU (user request 2026-10-05)", "done",
         ["audit/sae_sanity.md", "audit/sae_sanity.csv"])
    print("sae sanity written", flush=True)


if __name__ == "__main__":
    import sys
    if "--md-only" not in sys.argv:
        main()
    else:
        write_md()
