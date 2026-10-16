"""Naming migration (amendment 4/5): relabel existing run records (labels only; numbers unchanged).

    python -m experiments.v4.relabel [--dry-run]

SSAE = trained on shifts, SAE = trained on raw activations. ``act_sae_*`` -> ``sae_*``,
``ssae_v3_*`` -> ``ssae_linear_*``. Live record trees (sweep, planted, bench, smoke, final) and
their saved states are renamed; ``config.json`` gets the new method metadata and a
``relabelled_from`` field. Superseded trees (``records/_superseded_*``) keep their historical
names. The activation-SAE training records move from ``records/act_sae`` to ``records/sae``.
"""

from __future__ import annotations

import argparse
import json
import shutil

from experiments.v4.common import RECORDS, v4_root

RENAME = {
    **{f"act_sae_{t}": f"sae_{t}" for t in ("relu", "jumprelu", "topk", "relu_v3", "jumprelu_v3", "topk_v3")},
    **{f"ssae_v3_{v}": f"ssae_linear_{v}" for v in ("l1", "step_l0", "sigmoid_l0")},
}
TREES = ("sweep", "planted", "bench", "smoke", "final")


def _load(p):
    return json.loads(p.read_text())


def _dump(p, obj):
    tmp = p.with_suffix(p.suffix + ".relabel.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)


def main() -> None:
    from experiments.common.ssae_rms_protocol import v4_protocol_metadata

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    n_dirs = n_states = n_units = 0
    for tree in TREES:
        root = RECORDS / tree
        if not root.is_dir():
            continue
        for old, new in RENAME.items():
            for d in sorted(root.rglob(old)):
                if not d.is_dir():
                    continue
                target = d.with_name(new)
                if target.exists():
                    raise SystemExit(f"both {d} and {target} exist")
                if not args.dry_run:
                    for name in ("metrics.json", "config.json"):
                        f = d / name
                        if f.is_file():
                            obj = _load(f)
                            if name == "metrics.json" and obj.get("method") == old:
                                obj["method"] = new
                            if name == "config.json":
                                obj.update({"method": new, **v4_protocol_metadata(new), "relabelled_from": old})
                            _dump(f, obj)
                    d.rename(target)
                n_dirs += 1
                rel = d.parent.relative_to(RECORDS)
                state = v4_root() / "runs" / rel / f"{old}.npz"
                if state.is_file():
                    if not args.dry_run:
                        state.rename(state.with_name(f"{new}.npz"))
                    n_states += 1
        for unit in root.rglob("_unit/unit.json"):
            obj = _load(unit)
            methods = obj.get("methods", [])
            if any(m in RENAME for m in methods):
                if not args.dry_run:
                    obj["methods"] = [RENAME.get(m, m) for m in methods]
                    _dump(unit, obj)
                n_units += 1
    old_train, new_train = RECORDS / "act_sae", RECORDS / "sae"
    if old_train.is_dir() and not new_train.exists() and not args.dry_run:
        shutil.move(str(old_train), str(new_train))
    print(f"{'would relabel' if args.dry_run else 'relabelled'}: {n_dirs} record dirs, {n_states} states, "
          f"{n_units} unit specs; training records -> records/sae")


if __name__ == "__main__":
    main()
