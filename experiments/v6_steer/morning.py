"""R.3 — MORNING_SUMMARY.md: finished / blocked (and why) / headline numbers / open questions, from PROGRESS.md and
the prediction records.

    python -m experiments.v6_steer.morning
"""

from __future__ import annotations

import json

from steer.common import RECORDS, RESULTS


def main() -> None:
    prog = (RESULTS / "PROGRESS.md").read_text().splitlines()
    rows = [l for l in prog if l.startswith("| ") and not l.startswith("| id") and not l.startswith("|---")]
    done = [l for l in rows if "| DONE |" in l]
    other = [l for l in rows if "| DONE |" not in l]
    preds = {}
    p = RECORDS / "predictions.json"
    if p.is_file():
        preds = json.loads(p.read_text())
    lines = ["# v6 steering — morning summary", "",
             f"{len(done)} tasks DONE; {len(other)} not done (N/A / BLOCKED, reasons below). Full status: "
             "`results/v6_steer/PROGRESS.md`; report: `results/v6_steer/REPORT_STEER.md`.", "",
             "## Predictions", ""]
    for k, v in preds.items():
        lines.append(f"- **{k}**: {v.get('outcome')} — {v.get('evidence', '')}")
    lines += ["", "## Not done", ""] + [f"- {l.split('|')[1].strip()} {l.split('|')[2].strip()}: "
                                         f"{l.split('|')[3].strip()} — {l.split('|')[4].strip()}" for l in other]
    q = RECORDS / "open_questions.md"
    if q.is_file():
        lines += ["", "## Open questions (recommended answers)", "", q.read_text().strip()]
    (RESULTS / "MORNING_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("wrote MORNING_SUMMARY.md")


if __name__ == "__main__":
    main()
