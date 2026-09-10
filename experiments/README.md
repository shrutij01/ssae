# Experiment code

The numbered packages are the public experiment map. Keep each package shallow:

| Package | Question | Main entry point |
|---|---|---|
| `exp01` | How faithfully do the models reconstruct held-out activations? | `python -m experiments.exp01.report` |
| `exp02` | Do learned features recover and align with planted concepts? | `python -m experiments.exp02.report` |
| `exp03` | Does one train-selected direction satisfy positive > baseline > negative cosine from the same held-out source? | `python -m experiments.exp03.run` |
| `exp04` | Do source-selected steering directions transfer OOD? | `python -m experiments.exp04.report` |
| `exp05` | Which features are selectable and stable across seeds? | `python -m experiments.exp05.report` |
| `exp06` | What is identifiable in the controlled synthetic setting? | `python -m experiments.exp06.run` |
| `exp07` | Does applying an English→French direction again to French activations preserve the target or overshoot it? | `python -m experiments.exp07.run` |

`common/` contains code used by more than one numbered experiment, including
the frozen scalar-RMS protocol and publication-report helpers. `legacy/`
isolates pre-numbering compatibility runners; they remain callable for older
workflows but are not the namespace for new experiments. New experiment logic
belongs in its numbered package; cluster launchers and data-conversion
utilities remain in `scripts/`.

Experiment code must not contain result bundles, checkpoints, logs, or rendered
figures. Those belong under the top-level `results/` tree described in
[`results/README.md`](../results/README.md). Build every report with:

```bash
python -m experiments.build_results --check
python -m experiments.build_results --build
```

The top level is intentionally limited to `expNN/`, `common/`, `legacy/`, and
the all-report builder. Do not add another unnumbered experiment module there.
