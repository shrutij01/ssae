# Feature-atlas experiments

This package separates two scientific questions that the original atlas
combined:

1. **Concept selection:** does an atom have a conditional, reproducible
   association with a known factor on `cfc_train`?
2. **Cross-seed identity:** do independently trained models recover the same
   functional response, the same decoder direction, or both?

`selection.py` uses a joint no-intercept factorial model, term-conditional
studentized sign flips, and Benjamini-Hochberg FDR.  This prevents joint-factor
transitions from automatically making an atom look mixed.

`stability.py` reports response recurrence, decoder recurrence, empirical
response/geometry coupling, and strong atom identity separately.  In
particular, a model can learn response-equivalent atoms without learning the
same decoder atoms across seeds; this case remains visible and does not produce
a stable feature card.

Large caches and source snapshots belong under `$SSAE_DATA_ROOT/studies/`.
After review, compact machine-readable study sources and provenance belong
under `results/exp05/frozen/<study>`.
The reproducible builder in `experiments/exp05/report.py` validates those
sources and writes publication figures, tables, summaries, and the artifact
manifest under `results/exp05/report/`.

The current frozen configuration is in `configs/semisynth_v3.json`. The public
snapshot retains compact tables instead of the former generated card browser.
