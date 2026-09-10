# Result storage

`results/` is the only repository tree for experimental outputs. Code lives in
`experiments/`; narrative interpretation lives in `docs/`.

Each numbered experiment uses up to three lifecycle directories:

```text
results/
├── exp01/
│   ├── runs/       # mutable jobs, logs, caches, checkpoints (ignored)
│   ├── frozen/     # reviewed immutable inputs to the report
│   └── report/     # reproducible figures, tables, summary, manifest
├── …
├── exp06/
├── exp07/
├── shared/          # frozen inputs consumed by multiple experiments
├── findings/        # focused analyses outside the numbered headline set
└── artifact_index.json
```

Not every experiment needs every directory. Exp01 and Exp02, for example,
consume the same frozen identification matrix from `shared/identification/`.

The lifecycle is one-way:

```text
mutable run output  →  reviewed frozen bundle  →  deterministic report
     runs/                     frozen/                    report/
```

- `runs/` may be large and machine-specific. Put full checkpoints and cluster
  logs on scratch storage when possible; keep only small audit ledgers when they
  are scientifically useful.
- `frozen/` is source data. Never silently replace it during a report build.
- `report/` is derived. Each report carries a hashed `artifact_manifest.json`.
- Figure formats are declared by each report. Exp03 and Exp07 intentionally
  commit publication PDFs only; the other reports retain editable SVG and PDF
  variants. PNG remains an opt-in preview format, never a canonical copy.

External provenance paths are stored as portable `ssae-data://...` references,
relative to the directory named by `SSAE_DATA_ROOT`. Historical repository
locations may appear as non-resolvable `repo://...` audit references. Frozen
bundles must never publish a user's home directory, scratch prefix, or virtual
environment path.

Validate sources without writing, or rebuild all derived reports, with:

```bash
python -m experiments.build_results --check
python -m experiments.build_results --build
python -m experiments.build_results --build --only exp03
```

Documentation reaches report artifacts through read-only relative symlinks in
`docs/results/assets/`; the canonical files remain here.
