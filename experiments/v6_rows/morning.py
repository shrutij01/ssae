"""``results/v6_rows/MORNING_SUMMARY.md``: state of the study for the user (what finished, what is blocked and why,
whether the choice file arrived, headline numbers). Reads records only; never computes a test metric.

    python -m experiments.v6_rows.morning --note "free text"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.v6_rows.common import RECORDS, V6, live_repo, read_json

CHOICE = Path(os.environ["SCRATCH"]) / "ssae_shared" / "constrained_choice.json"


def fits(root: Path, pattern: str) -> list[dict]:
    out = []
    for f in glob.glob(str(root / "fit" / "*" / "L*" / "s*" / pattern / "metrics.json")):
        p = Path(f).parts
        j = read_json(f)
        ds = p[-5].split("__")[0]
        out.append({"row": p[-5], "layer": int(p[-4][1:]), "seed": int(p[-3][1:]), "method": p[-2],
                    "val": j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"],
                    "info": j.get("info", {})})
    return out


def squeue() -> str:
    r = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j %T %M"], capture_output=True, text=True)
    return "\n".join(x for x in r.stdout.splitlines() if x.startswith(("v6-", "fw-")))


def write(note: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c = json.loads(CHOICE.read_text()) if CHOICE.is_file() else None
    ch = (c or {}).get("choice") or {}
    L = [f"# v6 rows — morning summary ({now})", ""]
    L += ["## Optimizer choice file", ""]
    if c is None:
        L.append(f"- `{CHOICE}` does **not** exist yet.")
    elif not ch:
        L.append(f"- `{CHOICE}` exists, but its `choice` is still empty (the step-1 agent has written "
                 f"{len(c.get('per_candidate', {}))} per-candidate entries: "
                 f"{', '.join(sorted(c.get('per_candidate', {})))}).")
    else:
        L += [f"- Arrived. SSAE-{v}: candidate {x.get('candidate')} ({x.get('optimizer')}): {x.get('reason')}"
              for v, x in ch.items()]
    L += ["", "## SSAE-A / SSAE-B fits (validation only)", ""]
    new = fits(RECORDS, "ssae_*")
    if not new:
        L.append("- None yet.")
    else:
        by = {}
        for r in new:
            by.setdefault(r["method"], []).append(r)
        L += ["| method | fits | feasible | converged | hit step cap | best validation direction accuracy (row, layer) |",
              "|---|---|---|---|---|---|"]
        for meth, rs in sorted(by.items()):
            fe = np.mean([bool(r["info"].get("feasible")) for r in rs])
            co = np.mean([bool(r["info"].get("converged")) for r in rs])
            cap = np.mean([bool(r["info"].get("hit_step_cap")) for r in rs])
            best = max(rs, key=lambda r: r["val"])
            L.append(f"| {meth} | {len(rs)} | {fe:.2f} | {co:.2f} | {cap:.2f} | {best['val']:.3f} ({best['row']}, "
                     f"L{best['layer']}) |")
    sel = RECORDS / "selection.json"
    if sel.is_file():
        L += ["", "Selected layers: `records/selection.json`."]
    fw = fits(live_repo() / "results" / "v5_fullwidth" / "records", "ssae_full")
    L += ["", "## Budget-free SSAE (v5_fullwidth sweep, Adam lr 0.03 sqrt(64/d) until rel. change < 1e-6)", "",
          f"- {len(fw)} ssae_full fits on disk ({len({(r['row'], r['layer']) for r in fw})} row-layers)."]
    ot = RECORDS / "original_multiplier.json"
    if ot.is_file():
        L += ["", "## Original SSAE (LayerNorm): multiplier trajectory re-runs", ""]
        for t in read_json(ot)["runs"]:
            mu = t["multiplier"]
            L.append(f"- {t['row']} L{t['layer']}: multiplier {mu[0]:.3g} (update {t['step'][0]}) -> {mu[-1]:.3g} "
                     f"(update {t['step'][-1]}); step-L0 count {t['final_count']:.1f} per pair vs budget "
                     f"{t['budget']:.2f}; v5 record's final multiplier {(t.get('v5_record') or {}).get('dual_multiplier')}.")
    pre = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%H:%M", "-1", "--",
                          "results/v6_rows/PREREGISTRATION.md"], capture_output=True, text=True, cwd=live_repo()).stdout.strip()
    L += ["", "## Preregistration and test", "",
          f"- PREREGISTRATION.md: {'committed ' + pre if pre else 'not committed yet (no test metric computed)'}.",
          f"- REPORT.md: {'written' if (V6 / 'REPORT.md').is_file() else 'not written yet'}."]
    L += ["", "## Jobs at writing time", "", "```", squeue() or "(none)", "```"]
    if note:
        L += ["", "## Notes", "", note]
    (V6 / "MORNING_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("MORNING_SUMMARY.md written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    ap.add_argument("--note-file", default=None)
    a = ap.parse_args()
    write(Path(a.note_file).read_text() if a.note_file else a.note)


if __name__ == "__main__":
    main()
