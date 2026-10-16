"""Synthetic study v2: one markdown table per step, predicted_vs_observed.md and the CSV of all cells
(PREREGISTRATION addendum S2).

    python -m experiments.v6_synth.report2 [T0 A B C D X ...]   # -> results/v6_synth_v2/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_synth import gen2 as G
from experiments.v6_synth.run2 import RUNS

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__import__("os").environ.get("SYNTH_REPORT_OUT", REPO / "results" / "v6_synth_v2"))
LABEL = {"SSAE": "SSAE (L1, no LayerNorm; best in Table 0, chosen post hoc)", "L1-v3": "SSAE l1 (L1-v3)", "L0-v3": "SSAE l0 (L0-v3)", "LAE": "Linear autoencoder (aff)",
         "PCA_delta": "PCA_delta", "ICA_delta": "ICA_delta", "PCA_act": "PCA (activations)", "ICA_act": "ICA (activations)",
         "L1-noLN": "[diagnostic] SSAE l1 without LayerNorm (L1-noLN)",
         "L0-v3 (oracle budget = max|S|)": "SSAE l0, oracle budget = max|S|",
         "E9 null encoder (on true delta_c)": "control: E9 null encoder (true delta_c)",
         "E3 entangled (on true delta_c)": "control: E3 entangled (true delta_c)"}
ORDER = list(LABEL)
TARGET = {"A_V3_S2": (0.999, 0.873), "A_V4_S3": (0.999, 0.835), "A_V10_S7": (0.993, 0.769)}
SYN = {"A_V3_S2": "SYNTH(3,2)", "A_V4_S3": "SYNTH(4,3)", "A_V10_S7": "SYNTH(10,7)"}


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(RUNS.glob("*__s*/eval*.json")):
        e = json.loads(p.read_text())
        items = list(e["methods"].items()) + list(e.get("controls", {}).items())
        for m, v in items:
            r = {"cell": e["cell"], "seed": e["seed"], "tag": e["tag"], "V": e["V"], "method": m,
                 **{k: x for k, x in v.items() if k not in ("cos_per_concept", "ood")}}
            for s, o in v.get("ood", {}).items():
                r.update({f"{s}_mcc_p": o["mcc_p"], f"{s}_mcc_s": o["mcc_s"]})
            sel = e["selection"].get(m, {})
            if isinstance(sel, dict) and "budget" in sel:
                r.update({"budget": sel["budget"], "target": sel.get("target"), "meets_budget": sel.get("meets_budget"),
                          "val_active": sel.get("val_active")})
            rows.append(r)
    return pd.DataFrame(rows)


def ms(x, rng: bool = False) -> str:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n/a"
    s = f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
    return s + (f" [{x.min():.3f}, {x.max():.3f}]" if rng else "")


def grid(df: pd.DataFrame, cells: dict, metric: str, methods=ORDER, rng: bool = False) -> pd.DataFrame:
    out = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty or metric not in sub:
            continue
        row = {"method": LABEL.get(m, m)}
        for c, lab in cells.items():
            v = sub[sub["cell"] == c][metric]
            row[lab] = ms(v, rng) if len(v) else ""
        if any(row[lab] not in ("", "n/a") for lab in cells.values()):
            out.append(row)
    return pd.DataFrame(out)


def write(name: str, parts: list[tuple[str, pd.DataFrame]], head: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# {head}", "", "Mean ± sample std over seeds 0-4 (test split); [min, max] over seeds where shown. Design: "
          "`PREREGISTRATION.md` (Addendum S2).", ""]
    for cap, t in parts:
        md += [f"**{cap}**", "", t.to_markdown(index=False), ""]
    notes = OUT / "notes" / f"{name}.md"                      # hand-written notes, kept across rebuilds
    if notes.is_file():
        md += [notes.read_text().rstrip(), ""]
    (OUT / f"{name}.md").write_text("\n".join(md))


# ------------------------------------------------------------------ tables
def table0(df: pd.DataFrame) -> list[str]:
    d = df[(df["tag"] == "eval_T0") & df["cell"].isin(G.TABLE0)]
    cells = {c: SYN[c] for c in G.TABLE0}
    t = grid(d, cells, "mcc_p", rng=True)
    tgt = pd.DataFrame([{"method": "paper target (SSAE / aff)", **{SYN[c]: f"{TARGET[c][0]:.3f} / {TARGET[c][1]:.3f}"
                                                                    for c in G.TABLE0}}])
    write("table0_reproduction", [("MCC-P (identifiability-guard)", pd.concat([t, tgt])),
                                  ("MCC-S", grid(d, cells, "mcc_s")), ("Direction recovery |cos|", grid(d, cells, "cos_mean")),
                                  ("DCI disentanglement", grid(d, cells, "dci_d")), ("DCI completeness", grid(d, cells, "dci_c")),
                                  ("DCI informativeness", grid(d, cells, "dci_i"))],
          "Table 0: reproduction of SYNTH(3,2), (4,3), (10,7) (App. B.8)")
    lines = []
    for c in G.TABLE0:
        for m, i in (("L1-v3", 0), ("LAE", 1)):
            v = d[(d["cell"] == c) & (d["method"] == m)]["mcc_p"]
            ok = abs(v.mean() - TARGET[c][i]) <= 0.02
            lines.append((c, m, v.mean(), v.min(), v.max(), TARGET[c][i], ok))
    return lines


def table_a(df: pd.DataFrame) -> None:
    d = df[(df["tag"] == "eval") & df["cell"].str.startswith("A_")]
    cells = {c: c.replace("A_V", "|V|=").replace("_S", ", max|S|=").replace("_full", ", full") for c in G.CELLS
             if c.startswith("A_")}
    write("tableA_identification", [("Direction recovery |cos| (Hungarian, columns of L)", grid(d, cells, "cos_mean")),
                                    ("MCC-P", grid(d, cells, "mcc_p")), ("MCC-S", grid(d, cells, "mcc_s")),
                                    ("DCI disentanglement", grid(d, cells, "dci_d")),
                                    ("DCI completeness", grid(d, cells, "dci_c")),
                                    ("DCI informativeness", grid(d, cells, "dci_i")),
                                    ("SSAE selected budget b (codes per shift; seed mean)", grid(d, cells, "budget",
                                                                                                ["L1-v3", "L0-v3"]))],
          "Table A: identification grid")


def table_b(df: pd.DataFrame) -> None:
    d = df[df["tag"] == "eval"]
    ci = {f"B_phi{p:g}": f"phi={p:g}" for p in (0, 0.5, 0.8, 0.95, 1)}
    cii = {"B_phi0.5": "rho_delta=0", "B_rhod0.5": "rho_delta=0.5", "B_rhod0.9": "rho_delta=0.9"}
    parts = [("(i) |cos| concept 1", grid(d, ci, "cos_c1")), ("(i) |cos| concept 2", grid(d, ci, "cos_c2")),
             ("(i) largest principal angle (deg), span(L_1, L_2) vs the two matched directions",
              grid(d, ci, "plane_angle_max_deg")),
             ("(i) mean |cos| concepts 3-10", grid(d, ci, "cos_others")), ("(i) mean |cos| all", grid(d, ci, "cos_mean")),
             ("(i) MCC-P", grid(d, ci, "mcc_p")),
             ("(ii) mean |cos| all", grid(d, cii, "cos_mean")), ("(ii) MCC-P", grid(d, cii, "mcc_p")),
             ("(ii) |cos| concepts 1-2 (mean of the two)", grid(d.assign(cos_12=(d["cos_c1"] + d["cos_c2"]) / 2), cii,
                                                                 "cos_12")),
             ("(ii) largest principal angle (deg)", grid(d, cii, "plane_angle_max_deg"))]
    write("tableB_covariation", parts, "Table B: co-variation (i) and change correlation (ii), |V| = 10, max|S| = 7")


def table_c(df: pd.DataFrame) -> None:
    d = df[df["tag"] == "eval"]
    dc = d[d["cell"].isin(["A_V10_S7", "C_rhoc0.5", "C_rhoc0.9"])].copy()
    base = d[d["cell"] == "A_V10_S7"]
    extra = []
    for c in ("C_rhoc0.5", "C_rhoc0.9"):                     # difference methods: identical data, copied from A_V10_S7
        b = base[~base["method"].isin(["PCA_act", "ICA_act"]) & ~base["method"].str.startswith(("E9", "E3"))].copy()
        b["cell"] = c
        extra.append(b)
    dc = pd.concat([dc[~(dc["cell"].str.startswith("C_") & ~dc["method"].isin(["PCA_act", "ICA_act"]) &
                         ~dc["method"].str.startswith(("E9", "E3")))]] + extra)
    cells = {"A_V10_S7": "rho_c=0", "C_rhoc0.5": "rho_c=0.5", "C_rhoc0.9": "rho_c=0.9"}
    write("tableC_correlated_values", [("Direction recovery |cos|", grid(dc, cells, "cos_mean")),
                                       ("MCC-P", grid(dc, cells, "mcc_p"))],
          "Table C: correlated concept values (|V| = 10, max|S| = 7). Rows of methods on differences are the "
          "rho_c = 0 fits: their training data are identical at every rho_c (by construction)")


def table_d(df: pd.DataFrame) -> None:
    d = df[(df["tag"] == "eval") & (df["cell"] == "D_heldout")].copy()
    rows = []
    for m in ORDER:
        s = d[d["method"] == m]
        if s.empty or "test_a_mcc_p" not in s or s["test_a_mcc_p"].isna().all():
            continue
        rows.append({"method": LABEL[m], "in-distribution": ms(s["mcc_p"], True),
                     "(a) held-out pairs together": ms(s["test_a_mcc_p"], True),
                     "(b) |S| up to 9": ms(s["test_b_mcc_p"], True),
                     "(c) Uniform(-3, 3)": ms(s["test_c_mcc_p"], True)})
    write("tableD_ood", [("MCC-P of the inferred shifts", pd.DataFrame(rows))],
          "Table D: out of distribution (train SYNTH(10,7) with 14 of 45 concept pairs never changing together)")


def table_x(df: pd.DataFrame) -> None:
    d = df[df["tag"] == "eval"]
    cells = {"X_laplace": "Laplace", "X_gauss": "Gaussian", "B_phi1": "Uniform", "X_rademacher": "Rademacher"}
    mx = d.assign(mix=(d.get("mix_angle_c1_deg", np.nan) + d.get("mix_angle_c2_deg", np.nan)) / 2)
    parts = [("|cos| concept 1", grid(d, cells, "cos_c1")), ("|cos| concept 2", grid(d, cells, "cos_c2")),
             ("largest principal angle (deg)", grid(d, cells, "plane_angle_max_deg")),
             ("mixing angle theta_1 (deg; 0 = aligned with L_1, 45 = equal mix)", grid(d, cells, "mix_angle_c1_deg",
                                                                                      rng=True)),
             ("mixing angle theta_2 (deg)", grid(d, cells, "mix_angle_c2_deg", rng=True)),
             ("mean |cos| concepts 3-10", grid(d, cells, "cos_others"))]
    write("appendix_shapes", parts, "Appendix: shape of changes at phi = 1 (|V| = 10, max|S| = 7)")
    o = df[df["tag"] == "eval_oracle"]
    if not o.empty:
        cells = {c: c for c in G.CELLS if c.startswith("A_")}
        write("appendix_oracle_l0", [("|cos|", grid(o, cells, "cos_mean")), ("MCC-P", grid(o, cells, "mcc_p"))],
              "Appendix: SSAE l0 with budget = max|S| (oracle; not a label-free choice)")


V64 = {"A_V64_S32": "max\\|S\\| = 32", "A_V64_S45": "max\\|S\\| = 45", "A_V64_S63": "max\\|S\\| = 63",
       "A_V64_full": "full (\\|S\\| = 64)"}                     # pipes escaped for markdown tables
GEMMA = {"G_V2048_S2000": "d_z = 2304, |V| = 2048, max|S| = 2000"}
S3_METHODS = ["SSAE", "LAE", "PCA_delta", "ICA_delta", "PCA_act", "ICA_act", "E9 null encoder (on true delta_c)",
              "E3 entangled (on true delta_c)"]


def table_v64(df: pd.DataFrame) -> None:
    d = df[(df["tag"] == "eval_s3") & df["cell"].isin(V64)]
    parts = [(lab, grid(d, V64, k, S3_METHODS, rng=True)) for k, lab in
             (("mcc_p", "MCC-P"), ("mcc_s", "MCC-S"), ("dci_d", "DCI disentanglement"), ("dci_c", "DCI completeness"),
              ("dci_i", "DCI informativeness (test R^2)"), ("cos_mean", "Direction recovery |cos| (columns of L)"))]
    parts.append(("SSAE selected budget b (codes per shift; elbow rule)", grid(d, V64, "budget", ["SSAE"], rng=True)))
    write("table_v64", parts, "|V| = 64 concepts in d_z = 128 (Table A's largest setting); SSAE = the L1 SSAE without "
          "LayerNorm, chosen after Table 0 (Addendum S3)")


def table_gemma(df: pd.DataFrame) -> None:
    d = df[(df["tag"] == "eval_s3") & df["cell"].isin(GEMMA)]
    rows = []
    for m in S3_METHODS:
        s = d[d["method"] == m]
        if s.empty:
            continue
        rows.append({"method": LABEL.get(m, m), "MCC-P": ms(s["mcc_p"], True), "MCC-S": ms(s["mcc_s"], True),
                     "\\|cos\\|": ms(s["cos_mean"], True) if s["cos_mean"].notna().any() else "n/a"})
    b = d[d["method"] == "SSAE"]["budget"]
    write("table_gemma_scale", [("MCC and direction recovery", pd.DataFrame(rows)),
                                ("SSAE selected budget b", pd.DataFrame([{"budget": ms(b, True)}]))],
          "Gemma scale: d_z = 2304 (Gemma-2-2B width), |V| = 2048 concepts, |S| ~ Uniform{1..2000}; SSAE = the L1 "
          "SSAE without LayerNorm, chosen after Table 0 (Addendum S3); MCC only (no DCI)")


def predictions_s3(df: pd.DataFrame) -> list[str]:
    e = df[(df["tag"] == "eval_s3") & df["cell"].isin(V64)]
    if e.empty:
        return []
    sm = lambda m, c, k: e[(e["method"] == m) & (e["cell"] == c)][k].mean()
    ident = lambda m, c: sm(m, c, "cos_mean") >= 0.95 and sm(m, c, "mcc_p") >= 0.95
    var, full = ["A_V64_S32", "A_V64_S45", "A_V64_S63"], ["A_V64_full"]
    out = ["", "## Addendum S3, |V| = 64 (SSAE = L1 without LayerNorm, chosen after Table 0)", ""]

    def line(key, text, ok, detail):
        out.append(f"- **{key}** {text}: **{'held' if ok else 'failed'}**. {detail}")
    line("A1 (|V| = 64)", "SSAE identified wherever supports vary, not with full supports",
         all(ident("SSAE", c) for c in var) and not ident("SSAE", full[0]),
         "; ".join(f"{V64[c]}: |cos| {sm('SSAE', c, 'cos_mean'):.3f}, MCC-P {sm('SSAE', c, 'mcc_p'):.3f}" for c in V64))
    for m in ("LAE", "PCA_delta"):
        line("A2 (|V| = 64)", f"{LABEL[m]} not identified", not any(ident(m, c) for c in V64),
             "; ".join(f"{V64[c]}: |cos| {sm(m, c, 'cos_mean'):.3f}, MCC-P {sm(m, c, 'mcc_p'):.3f}" for c in V64))
    line("A3 (|V| = 64)", "ICA_delta identified with full supports", ident("ICA_delta", full[0]),
         f"|cos| {sm('ICA_delta', full[0], 'cos_mean'):.3f}, MCC-P {sm('ICA_delta', full[0], 'mcc_p'):.3f}")
    line("A4 (|V| = 64)", "ICA (activations) identified", all(ident("ICA_act", c) for c in V64),
         "; ".join(f"{V64[c]}: |cos| {sm('ICA_act', c, 'cos_mean'):.3f}, MCC-P {sm('ICA_act', c, 'mcc_p'):.3f}" for c in V64))
    line("A4 (|V| = 64)", "PCA (activations) not identified", not any(ident("PCA_act", c) for c in V64),
         "; ".join(f"{V64[c]}: |cos| {sm('PCA_act', c, 'cos_mean'):.3f}" for c in V64))
    g = df[(df["tag"] == "eval_s3") & df["cell"].isin(GEMMA)]
    if not g.empty:
        s = g[g["method"] == "SSAE"]
        out.append(f"- **G (no prediction)** Gemma scale, SSAE: MCC-P {s['mcc_p'].mean():.3f}, |cos| {s['cos_mean'].mean():.3f}.")
    return out


# ------------------------------------------------------------------ predictions
def predictions(df: pd.DataFrame) -> list[str]:
    e = df[df["tag"] == "eval"]
    sm = lambda m, c, k: e[(e["method"] == m) & (e["cell"] == c)][k].mean()
    ident = lambda m, c: sm(m, c, "cos_mean") >= 0.95 and sm(m, c, "mcc_p") >= 0.95
    have = lambda cells: all(((e["cell"] == c) & (e["method"] == "L1-v3")).any() or
                             ((e["cell"] == c) & (e["method"] == "ICA_act")).any() for c in cells)
    out = []

    def line(key, text, ok, detail):
        out.append(f"- **{key}** {text}: **{'held' if ok else 'failed'}**. {detail}")
    A = [c for c in G.CELLS if c.startswith("A_")]
    var, full = [c for c in A if not c.endswith("full")], [c for c in A if c.endswith("full")]
    if have(A):
        ok = all(ident("L1-v3", c) for c in var) and not any(ident("L1-v3", c) for c in full)
        line("A1", "SSAE l1 identified wherever supports vary, not with full supports", ok,
             "; ".join(f"{c}: |cos| {sm('L1-v3', c, 'cos_mean'):.3f}, MCC-P {sm('L1-v3', c, 'mcc_p'):.3f}" for c in A))
        for m in ("LAE", "PCA_delta"):
            ok = not any(ident(m, c) for c in A)
            line("A2", f"{LABEL[m]} not identified anywhere", ok,
                 "identified in: " + (", ".join(c for c in A if ident(m, c)) or "none"))
        ok = all(ident("ICA_delta", c) for c in full)
        line("A3", "ICA_delta identified with full supports", ok,
             "; ".join(f"{c}: |cos| {sm('ICA_delta', c, 'cos_mean'):.3f}, MCC-P {sm('ICA_delta', c, 'mcc_p'):.3f}"
                       for c in full))
        ok = all(ident("ICA_act", c) for c in A)
        line("A4", "ICA (activations) identified (rho_c = 0)", ok,
             "not identified in: " + (", ".join(c for c in A if not ident("ICA_act", c)) or "none"))
        ok = not any(ident("PCA_act", c) for c in A)
        line("A4", "PCA (activations) not identified", ok,
             "identified in: " + (", ".join(c for c in A if ident("PCA_act", c)) or "none"))
        out.append("- **A5** SSAE l0: reported, no prediction.")
    B1 = [f"B_phi{p:g}" for p in (0, 0.5, 0.8, 0.95, 1)]
    if have(B1):
        mn = {c: min(sm("L1-v3", c, "cos_c1"), sm("L1-v3", c, "cos_c2")) for c in B1}
        ok = all(mn[c] >= 0.95 for c in B1[:3]) and mn["B_phi0.95"] >= 0.90
        line("B1", "SSAE l1 separates concepts 1 and 2 for phi < 1", ok,
             "min(|cos| c1, c2): " + "; ".join(f"{c}: {mn[c]:.3f}" for c in B1[:4]))
        ang = sm("L1-v3", "B_phi1", "plane_angle_max_deg")
        mx = max(sm("L1-v3", "B_phi1", "cos_c1"), sm("L1-v3", "B_phi1", "cos_c2"))
        line("B2", "at phi = 1 only their plane (angle <= 10 deg, both |cos| < 0.95)", ang <= 10 and mx < 0.95,
             f"angle {ang:.1f} deg; |cos| c1 {sm('L1-v3', 'B_phi1', 'cos_c1'):.3f}, c2 {sm('L1-v3', 'B_phi1', 'cos_c2'):.3f}")
        oth = [sm("L1-v3", c, "cos_others") for c in B1]
        line("B3", "other concepts unaffected (>= 0.95, spread <= 0.02)", min(oth) >= 0.95 and max(oth) - min(oth) <= 0.02,
             "concepts 3-10: " + ", ".join(f"{x:.3f}" for x in oth))
    B2 = ["B_phi0.5", "B_rhod0.5", "B_rhod0.9"]
    if have(B2):
        line("B4", "SSAE l1 identified at every rho_delta", all(ident("L1-v3", c) for c in B2),
             "; ".join(f"{c}: |cos| {sm('L1-v3', c, 'cos_mean'):.3f}, MCC-P {sm('L1-v3', c, 'mcc_p'):.3f}" for c in B2))
        v = [(sm("ICA_delta", c, "cos_c1") + sm("ICA_delta", c, "cos_c2")) / 2 for c in B2]
        ok = v[1] - v[0] <= 0.01 and v[2] - v[1] <= 0.01 and v[0] - v[2] >= 0.05
        line("B5", "ICA_delta degrades as rho_delta grows (concepts 1-2)", ok, ", ".join(f"{x:.3f}" for x in v))
    C = ["A_V10_S7", "C_rhoc0.5", "C_rhoc0.9"]
    if have(C):
        v = [sm("ICA_act", c, "cos_mean") for c in C]
        ok = ident("ICA_act", C[0]) and v[1] - v[0] <= 0.01 and v[2] - v[1] <= 0.01 and v[0] - v[2] >= 0.05
        line("C1", "ICA (activations) identified at rho_c = 0 and degrades", ok,
             "|cos| " + ", ".join(f"{x:.3f}" for x in v) + "; MCC-P " + ", ".join(f"{sm('ICA_act', c, 'mcc_p'):.3f}" for c in C))
        line("C2", "PCA (activations) fails at every rho_c", not any(ident("PCA_act", c) for c in C),
             "|cos| " + ", ".join(f"{sm('PCA_act', c, 'cos_mean'):.3f}" for c in C))
        out.append("- **C3** methods on differences unchanged: **held (by construction)**: the difference data do not "
                   "depend on c, so these fits are the rho_c = 0 fits.")
    d = e[(e["cell"] == "D_heldout") & (e["method"] == "L1-v3")]
    if len(d) and "test_a_mcc_p" in d:
        idm = d["mcc_p"].mean()
        diffs = {s: d[f"{s}_mcc_p"].mean() - idm for s in G.D_TESTS}
        line("D1", "SSAE l1 within 0.05 of its in-distribution MCC-P on (a), (b), (c)",
             all(abs(x) <= 0.05 for x in diffs.values()),
             f"ID {idm:.3f}; " + "; ".join(f"{s}: {x:+.3f}" for s, x in diffs.items()))
    X = {"X_laplace": "Laplace", "X_gauss": "Gaussian", "B_phi1": "Uniform", "X_rademacher": "Rademacher"}
    if have(list(X)):
        g = lambda c, k: e[(e["method"] == "L1-v3") & (e["cell"] == c)][k]
        mn = min(g("X_laplace", "cos_c1").mean(), g("X_laplace", "cos_c2").mean())
        line("X1", "Laplace separated", mn >= 0.95, f"min |cos| {mn:.3f}")
        plane = lambda c: g(c, "plane_angle_max_deg").mean() <= 10 and max(g(c, "cos_c1").mean(), g(c, "cos_c2").mean()) < 0.95
        sd = g("X_gauss", "mix_angle_c1_deg").std(ddof=1)
        line("X2", "Gaussian plane only, rotation varies by seed", plane("X_gauss") and sd >= 10,
             f"angle {g('X_gauss', 'plane_angle_max_deg').mean():.1f} deg, |cos| c1/c2 {g('X_gauss', 'cos_c1').mean():.3f}/"
             f"{g('X_gauss', 'cos_c2').mean():.3f}, theta_1 seed SD {sd:.1f} deg")
        for c in ("B_phi1", "X_rademacher"):
            t1, t2 = g(c, "mix_angle_c1_deg").mean(), g(c, "mix_angle_c2_deg").mean()
            line("X3", f"{X[c]} plane only, mixed near 45 deg", plane(c) and abs(t1 - 45) <= 10 and abs(t2 - 45) <= 10,
                 f"angle {g(c, 'plane_angle_max_deg').mean():.1f} deg, |cos| c1/c2 {g(c, 'cos_c1').mean():.3f}/"
                 f"{g(c, 'cos_c2').mean():.3f}, theta_1/theta_2 {t1:.1f}/{t2:.1f} deg")
    return out


def main() -> None:
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_cells.csv", index=False)
    steps = sys.argv[1:] or ["T0", "A", "B", "C", "D", "X", "V64", "G"]
    t0 = []
    if "T0" in steps:
        t0 = table0(df)
    for s, fn in (("A", table_a), ("B", table_b), ("C", table_c), ("D", table_d), ("X", table_x), ("V64", table_v64),
                  ("G", table_gemma)):
        if s in steps:
            fn(df)
    lines = ["# Predicted vs observed (Addendum S2)", "", "Seed means; test split. One line per prediction.", ""]
    if t0:
        for c, m, mean, lo, hi, tgt, ok in t0:
            lines.append(f"- **T0** {SYN[c]} {LABEL[m]}: MCC-P {mean:.3f} [{lo:.3f}, {hi:.3f}] vs target {tgt:.3f} "
                         f"(within 0.02): **{'held' if ok else 'failed'}**")
    lines += predictions(df)
    if "T0" in steps and not any(x in steps for x in ("A", "B", "C", "D", "X")):
        lines += ["", "**Gate failed: phases A, B, C, D and the appendix of Addendum S2 were not run (stopped as "
                      "instructed, 2026-10-05 20:21 UTC).**"]
    lines += predictions_s3(df)
    if (OUT / "notes" / "predicted_vs_observed.md").is_file():
        lines += [(OUT / "notes" / "predicted_vs_observed.md").read_text().rstrip()]
    (OUT / "predicted_vs_observed.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
