# Documentation tree

`docs/` contains prose, interpretation, and protocol explanations. It does not
own experiment outputs or generated data caches.

- Result narratives live in `docs/results/*.md`.
- Canonical artifacts live in top-level `results/`.
- `docs/results/assets/exp01` through `exp07` are relative, read-only symlinks
  to each experiment's `report/` directory so MkDocs can publish figures and
  tables without duplicating them.
- `site/` is a disposable ignored MkDocs build.
- Large feature-card caches are reproducible run outputs under
  `results/exp05/runs/feature_atlas/`; they are ignored rather than copied into
  the documentation tree.

When adding an experiment, put executable logic under `experiments/expNN/`,
data under `results/expNN/{runs,frozen,report}`, and only its explanation here.
See `experiments/README.md` and `results/README.md` for the complete contract.
