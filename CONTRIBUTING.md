# Contributing

Issues and pull requests are welcome. Keep experiment logic under a shallow
`experiments/expNN/` package and keep every generated artifact under
`results/expNN/`; do not add outputs to `docs/` or experiment packages.

Before opening a pull request, run:

```bash
python -m ruff check .
python -m pytest
python -m experiments.build_results --check
python scripts/generate_experiment_pages.py --check
```

Large checkpoints, activation files, logs, caches, and local environment files
must remain untracked. Frozen result inputs should be compact, hashed, and free
of machine-specific absolute paths. By contributing, you agree that your
contribution is licensed under the repository's MIT License.
