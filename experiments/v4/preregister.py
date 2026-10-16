"""Append the chosen configurations to PREREGISTRATION.md (Phase 3 → 4 boundary).

    python -m experiments.v4.preregister

Rebuilds ``tables/layer_sweep.csv`` and ``preregistered_configs.json`` from the
validation sweep, appends Section 4 (the configuration table) and the line
``PREREGISTERED-CONFIGS-SHA256: <sha of preregistered_configs.json>``. The
runner opens test evaluation only when HEAD contains both files with matching
hash, so this must be committed before Phase 4.
"""

from __future__ import annotations

import hashlib

from experiments.v4.aggregate import choose, sweep
from experiments.v4.common import MODEL_TAG, RESULTS, read_json

MARK = "## 4. Chosen configurations"


def main() -> None:
    cfg = choose(sweep())
    raw = (RESULTS / "preregistered_configs.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = (RESULTS / "PREREGISTRATION.md").read_text()
    head = text.split(MARK)[0].rstrip() + "\n\n"
    lines = [MARK, "",
             f"Rule: {cfg['rule']} (sweep seed {cfg['sweep_seed']}). Source: `tables/layer_sweep.csv`, "
             "`preregistered_configs.json`. Read position (Bias-in-Bios, CEBaB): `read_position.json`. Validation "
             "diagnostics and the decisions taken on them: `DIAGNOSTICS.md` (DEVIATIONS #20, #29); SSAE fits re-run with "
             "32 restarts + PCA/ICA initialisations and the amendment-4 tau rule (#27). Design check: `DESIGN_CHECK.md`. "
             "PCA integrity gate: `PCA_CHECKS.md` (passed; #33). Primary method: `ssae_gated_count` (§3c). Sensitivity "
             "rows (`__rsplit`) sit at their base method's layer.", "",
             "| dataset | model | method | run | layer | read | val MCC (seed 0) | label-free hyper-parameters at sweep | seeds |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cfg["configs"]:
        hyper = ", ".join(f"{k}={v}" for k, v in c["hyper_at_sweep_seed"].items())
        lines.append(f"| {c['dataset']} | {MODEL_TAG[c['model']]} | {c['method']} | {c['method_run_id']} | {c['layer']} | "
                     f"{c['read']} | {'—' if c['val_mcc'] is None else format(c['val_mcc'], '.3f')} | {hyper} | "
                     f"{'0–4' if c['stochastic'] else '0 (deterministic)'} |")
    dec = RESULTS / "records" / "diagnostics" / "_decisions.json"
    if dec.is_file():
        d = read_json(dec)
        lines += ["", f"Diagnostics decisions: gate removes the signal = {d['gate_removes_signal']}; optimizer failure "
                      f"rows = {d['optimizer_failure_rows'] or 'none'} (see DIAGNOSTICS.md)."]
    lines += ["", f"PREREGISTERED-CONFIGS-SHA256: {sha}", ""]
    (RESULTS / "PREREGISTRATION.md").write_text(head + "\n".join(lines))
    print(f"{len(cfg['configs'])} configurations; sha256 {sha}")


if __name__ == "__main__":
    main()
