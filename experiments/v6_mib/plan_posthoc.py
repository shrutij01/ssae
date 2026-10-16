"""Post hoc (DEVIATIONS #93, #95; PREREGISTRATION Addenda 22, 24): each method at its own best layer, matched size. CPU.

    python -m experiments.v6_mib.plan_posthoc --val      # queue the per-layer validation curves (queue_ph/)
    python -m experiments.v6_mib.plan_posthoc --select   # (layer, k) per method and variable -> records/posthoc_selection.json
    python -m experiments.v6_mib.plan_posthoc --test     # public test once at each selected point (Addendum 22 committed)
    python -m experiments.v6_mib.plan_posthoc --table    # tables/posthoc_best_layer.{md,csv}

Layers: the even layers 0-24 (ARC: 12-24) plus each variable's existing curve at the neuron mask's layer
(records/dbm_size_sites.json), which is reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
REPO = Path(__file__).resolve().parents[2]
QDIR = ROOT / "queue_ph"
TASKS = (("mcqa", "gemma"), ("arc", "gemma"))
VARS = ("answer_pointer", "answer")
BASES = ("ssae_L1noln", "ssae_L0noln", "pca_delta", "gemmascope")
NAMES = {"ssae_L1noln": "SSAE (l1) without LayerNorm: encoder coordinates",
         "ssae_L0noln": "SSAE (l0) without LayerNorm: encoder coordinates", "pca_delta": "PCA_delta",
         "gemmascope": "GemmaScope (MIB's SAE treatment)", "neurons": "Neurons"}
KS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
LABEL = "Post hoc: each method at its own best layer"


def sites():
    return json.loads((RESULTS / "records" / "dbm_size_sites.json").read_text())


def layers_for(t: str, var: str) -> list[int]:
    lo = 12 if t == "arc" else 0
    site = sites()[f"{t}/gemma/{var}"]["layer"]
    return [L for L in range(lo, 25, 2) if L != site]


def td(t, m) -> Path:
    return ROOT / "mib" / f"{t}__{m}"


def item(snap, t, m, var, b, layers, prio, test_k=0):
    args = [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "mib", "dbm_size_layers.py", "--task", t,
            "--model", m, "--variable", var, "--basis", b, "--layers", ",".join(map(str, layers))]
    if test_k:
        args += ["--test-k", str(test_k)]
        done = [str(td(t, m) / "runs" / "test" / "dbm_size_ph" / var / b / f"L{layers[0]}_k{test_k}.json")]
    else:
        done = [str(td(t, m) / "runs" / "val" / "dbm_size_ph" / var / b / f"L{L}.json") for L in layers]
    it = {"cmd": "module", "module": "cProfile", "args": args, "prio": prio, "done": done}
    if t == "arc":
        from experiments.v6_mib.plan_p1v14 import arc_tags
        it.update(arc_tags(min(layers)))
    return it


def write_spec(name: str, items: list) -> None:
    QDIR.mkdir(parents=True, exist_ok=True)
    f = QDIR / f"{name}.jsonl"
    have = {json.loads(line)["done"][0] for line in f.read_text().splitlines() if line.strip()} if f.exists() else set()
    new = [it for it in items if it["done"][0] not in have]
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text((f.read_text() if f.exists() else "") + "".join(json.dumps(it) + "\n" for it in new))
    os.replace(tmp, f)
    print(f"[plan_posthoc] {name}: {len(new)} new items", flush=True)


def plan_val() -> None:
    from experiments.v6_mib.qitems import snapshot
    snap = snapshot()
    items = []
    for t, m in TASKS:
        for vi, var in enumerate(VARS):
            Ls = layers_for(t, var)
            for bi, b in enumerate(BASES):
                if t == "mcqa":                          # fast: one item per (variable, basis)
                    items.append(item(snap, t, m, var, b, Ls, round(0.001 + 0.0001 * (2 * vi + bi), 5)))
                else:                                    # ARC: chunks of at most 3 layers
                    chunks = [Ls[i:i + 3] for i in range(0, len(Ls), 3)]
                    for ci, ch in enumerate(chunks):
                        items.append(item(snap, t, m, var, b, ch, round(0.01 + 0.0001 * (8 * ci + 2 * bi + vi), 5)))
    write_spec("x73_posthoc_val", items)


def curve_of(t, m, var, b, L):
    """Validation curve {k: iia} at layer L: the new per-layer run, or the existing size curve at the site."""
    f = td(t, m) / "runs" / "val" / "dbm_size_ph" / var / b / f"L{L}.json"
    if f.exists():
        return {int(k): v["val_iia"] for k, v in json.loads(f.read_text())["curve"].items()}, str(f)
    s = sites()[f"{t}/{m}/{var}"]
    if L == s["layer"]:
        g = td(t, m) / "runs" / "val" / "dbm_size" / var / f"{b}.json"
        rec = json.loads(g.read_text())
        return {int(k): v["val_iia"] for k, v in rec["topk"]["0.001"].items()}, str(g)
    return None, ""


def select() -> dict:
    out, missing = {}, []
    for t, m in TASKS:
        for var in VARS:
            site = sites()[f"{t}/{m}/{var}"]["layer"]
            for b in BASES:
                best = None
                for L in sorted(set(layers_for(t, var)) | {site}):
                    c, src = curve_of(t, m, var, b, L)
                    if c is None:
                        missing.append(f"{t}/{var}/{b}/L{L}")
                        continue
                    for k in KS:                         # max IIA; ties to the smaller k, then the lower layer
                        if k in c and (best is None or c[k] > best[0] + 1e-12 or
                                       (abs(c[k] - best[0]) <= 1e-12 and (k, L) < (best[1], best[2]))):
                            best = (c[k], k, L, src)
                c, _ = curve_of(t, m, var, b, best[2])      # Addendum 24: the existing rule at the selected layer
                k_rule = min(k for k in KS if k in c and c[k] >= max(c.values()) - 0.01 - 1e-12)
                out[f"{t}/{m}/{var}/{b}"] = {"task": t, "model": m, "variable": var, "basis": b, "val_iia": best[0],
                                             "k": best[1], "layer": best[2], "source": best[3], "site_layer": site,
                                             "k_rule": k_rule, "val_iia_rule": c[k_rule]}
    if missing:
        raise SystemExit(f"[plan_posthoc] curves missing: {missing[:8]} ... ({len(missing)})")
    (RESULTS / "records" / "posthoc_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])
    return out


def existing_test(t, m, var, b):
    bud = json.loads((RESULTS / "records" / "dbm_size_test_budgets.json").read_text())[f"{t}/{m}/{var}/{b}"]
    f = td(t, m) / "runs" / "test" / "dbm_size" / var / f"{b}_k{bud['k']}.json"
    return bud, (json.loads(f.read_text())["test_iia"] if f.exists() else None)


def plan_test() -> None:
    pre = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:results/v6_mib/PREREGISTRATION.md"],
                         capture_output=True, text=True).stdout
    assert LABEL in pre, "Addendum 22 is not committed"
    from experiments.v6_mib.qitems import snapshot
    snap = snapshot()
    sel = json.loads((RESULTS / "records" / "posthoc_selection.json").read_text())
    items = []
    for key, s in sel.items():
        bud, _ = existing_test(s["task"], s["model"], s["variable"], s["basis"])
        for k in sorted({s["k"], s["k_rule"]}):          # Addendum 24: the maximum and the existing rule's budget
            if s["layer"] == s["site_layer"] and k == bud["k"]:
                continue                                 # the existing test at that point is used (Addendum 22 item 5)
            if s["layer"] == s["site_layer"]:            # the site's logits live in dbm_size's state: copy them once
                src = td(s["task"], s["model"]) / "state" / "dbm_size" / s["variable"] / f"{s['basis']}_c0.001.npy"
                dst = td(s["task"], s["model"]) / "state" / "dbm_size_ph" / s["variable"] / s["basis"] / \
                    f"L{s['layer']}.npy"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
            items.append(item(snap, s["task"], s["model"], s["variable"], s["basis"], [s["layer"]], 0.0005, k))
    write_spec("x74_posthoc_test", items)


def posthoc_test(t, m, var, b, L, k):
    """Public-test IIA at (layer, k): the post hoc record, or the existing size-curve test at the site's budget."""
    f = td(t, m) / "runs" / "test" / "dbm_size_ph" / var / b / f"L{L}_k{k}.json"
    if f.exists():
        return json.loads(f.read_text())["test_iia"]
    bud, et = existing_test(t, m, var, b)
    if L == bud["layer"] and k == bud["k"]:
        return et
    g = td(t, m) / "runs" / "test" / "dbm_size_curve" / var / f"{b}.json"     # #98: the same mask, scored once there
    if L == bud["layer"] and g.exists():
        return json.loads(g.read_text())["masks"][str(k)]["test_iia"]
    return None


def table() -> None:
    sel = json.loads((RESULTS / "records" / "posthoc_selection.json").read_text())
    fmt = lambda x: "pending" if x is None else f"{100 * x:.1f}"      # noqa: E731
    rows = []
    md = ["# Post hoc: each method at its own best layer, matched size (MCQA and ARC-Easy, Gemma-2-2B)", "",
          "**Post hoc** (DEVIATIONS #93, #95; PREREGISTRATION Addenda 22, 24), added after every reported result; no "
          "reported number changes. DBM, mask size controlled, last token; IIA x100 at one unit, mean over the "
          "counterfactual datasets. Each cell: layer, k, validation IIA, public-test IIA.", "",
          "- **At the neuron mask's layer (existing):** the reported result at the best validation layer of MIB's DBM "
          "over neurons, with the existing size rule (Addendum 18: the smallest k within 1 point of the validation "
          "maximum).",
          "- **At the method's own best layer (new):** the layer holding the highest validation IIA over the even "
          "layers (ARC: 12-24) and the site layer, with two budgets:",
          "  - **rule:** the existing size rule at that layer; this column differs from the existing one only in the "
          "layer;",
          "  - **max:** Addendum 22's budget, the highest validation IIA (ties to the smaller k).",
          "- Neurons are reused: their layer is already the neuron mask's best validation layer.",
          "- Layers run: the even layers 0-24 (ARC 12-24) and the site layer; the other odd layers and ARC's layers "
          "below 12 were not run. Among these, every method's best validation layer is the neuron mask's layer "
          "(DEVIATIONS #98). The max budget's test values come from the test-curve run, which scores the same masks "
          "once.",
          "- **ARC, both columns:** the layer and the size are chosen on validation data that ARC's masks were trained "
          "on, because MIB trains ARC's DBM masks on train + validation. ARC's choices are therefore in-sample.",
          "- The public test is scored once per (layer, k). A cell at the site layer and the existing budget reuses "
          "the existing test.", "",
          "| task / variable | method | at the neuron mask's layer (existing rule) | own best layer, existing rule | "
          "own best layer, max |", "|---|---|---|---|---|"]
    for t, m in TASKS:
        for var in VARS:
            for b in BASES + ("neurons",):
                bud, et = existing_test(t, m, var, b)
                ex = (bud["layer"], bud["k"], bud["val_iia_at_k"], et)
                if b == "neurons":
                    nr = nm = ex
                else:
                    s = sel[f"{t}/{m}/{var}/{b}"]
                    nr = (s["layer"], s["k_rule"], s["val_iia_rule"],
                          posthoc_test(t, m, var, b, s["layer"], s["k_rule"]))
                    nm = (s["layer"], s["k"], s["val_iia"], posthoc_test(t, m, var, b, s["layer"], s["k"]))
                cell = lambda x: f"L{x[0]}, k = {x[1]}, {fmt(x[2])}, {fmt(x[3])}"   # noqa: E731
                md.append(f"| {t} / {var} | {NAMES[b]} | {cell(ex)} | {cell(nr)} | {cell(nm)}"
                          + (" (neurons reused)" if b == "neurons" else "") + " |")
                for col, x, lab in (("at the neuron mask's layer, existing rule", ex, "existing"),
                                    ("at the method's own best layer, existing rule", nr, "post hoc"),
                                    ("at the method's own best layer, max", nm, "post hoc")):
                    rows.append({"task": t, "variable": var, "method": b, "column": col, "layer": x[0], "k": x[1],
                                 "val_iia": x[2], "test_iia": x[3], "label": lab})
    (RESULTS / "tables" / "posthoc_best_layer.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "tables" / "posthoc_best_layer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for opt in ("--val", "--select", "--test", "--table"):
        g.add_argument(opt, action="store_true")
    a = ap.parse_args()
    if a.val:
        plan_val()
    elif a.select:
        select()
    elif a.test:
        plan_test()
    else:
        table()


if __name__ == "__main__":
    main()
