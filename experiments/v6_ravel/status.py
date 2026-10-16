"""results/v6_ravel/STATUS.md: one line per output (item, done / failed / running, files), kept current — a line is
replaced when its item is written again."""

from __future__ import annotations

import time

from experiments.v6_ravel.common import RESULTS

PATH = RESULTS / "STATUS.md"
HEAD = "# v6 RAVEL — status (one line per output; files relative to results/v6_ravel/)"


def mark(item: str, state: str, files: list[str] | str = (), note: str = "") -> None:
    files = [files] if isinstance(files, str) else list(files)
    line = f"- **{item}**: {state}" + (f" — {', '.join(f'`{f}`' for f in files)}" if files else "") + \
        (f" ({note})" if note else "") + f" [{time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC]"
    lines = PATH.read_text().splitlines() if PATH.exists() else [HEAD, ""]
    key = f"- **{item}**:"
    for i, x in enumerate(lines):
        if x.startswith(key):
            lines[i] = line
            break
    else:
        lines.append(line)
    PATH.write_text("\n".join(lines) + "\n")
