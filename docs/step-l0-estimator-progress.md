# SSAE step-L0 estimator progress

## Checkpoint 1 — baseline audit (2026-07-27)

- Preserved all pre-existing untracked artifacts.
- The current `554feda` baseline uses a hard `abs(h) > threshold` forward
  indicator, a rectangular surrogate derivative, `activation=none`, and an
  inequality constraint by default.
- The focused baseline passed: `8 passed` in `tests/test_ssae_sparsity.py`.
  The repository had no installed dev tools, so `pytest` and `ruff` were placed
  under `/tmp` only. The full-tree lint baseline has unrelated pre-existing
  findings.
- Historical June runs used the earlier pass-through STE, not the current
  rectangle. They therefore cannot validate the new estimator. They do show
  that achieved activity, rather than the numeric constraint target, is the
  quantity needed for L1/step-L0 comparison.
- Added tracker-level diagnostics for code scale, rectangular-gradient support,
  and the fraction of active entries reached by the rectangle. Focused tests
  cover boundary semantics, aggregation, reset, and invalid bandwidth.

Next: expose the diagnostics and multiplier in training logs, validate the
training path locally, then build and dry-run the exact 20-job SLURM pilot.

## Checkpoint 2 — diagnostics and tests (2026-07-27)

- Training now logs achieved constraint value, `p_sparsity`, mean/RMS code
  scale, dual multiplier, rectangle support density, and active-set rectangle
  coverage. Scalar metrics detach before conversion, removing the warning seen
  in historical jobs.
- A tiny end-to-end Cooper smoke run exercised the hard forward indicator and
  inequality update. At bandwidth 0.1, rectangle support was only 8–9% and
  achieved activity stayed at 1.0 in the first three epochs while the multiplier
  increased. This is a diagnostic observation, not yet a stall verdict.
- Validation: full suite `13 passed`; focused fatal-error lint and compilation
  passed. Full-tree lint still has unrelated baseline findings.

## Checkpoint 3 — pilot dry-run (2026-07-27)

- Added a submission-capped launcher that defaults to dry-run and refuses more
  than 20 jobs or a duplicate submission ledger.
- Exact matrix validated: each of `norm={ln,none}` has five L1 targets and five
  rectangular step-L0 bandwidths; fixed settings are threshold/`ind_th=0.01`,
  seed 0, warmup 100, schedule 500, epochs 2000, `activation=none`, hard step
  forward, and inequality constraint.
- Pilot dataset: refusal/Gemma-2-2B layer 25 (412 training pairs, one concept).
  Unspecified settings follow the current cluster baseline: relative loss,
  batch 64, LR 5e-4, standardize input norm, extrapolation/Lagrangian, and
  ExtraAdam dual optimization with divisor 2.
- Dry-run generated exactly 20 isolated job scripts; every script passed
  `bash -n`, and SLURM `--test-only` accepted a representative job without
  submitting it. The queue remained empty afterward.
- Added a tested analyzer that tail-averages reconstruction and achieved
  `p_sparsity`, then pairs step-L0 only with the nearest L1 achieved activity.
  It never matches estimators by configured numeric target.

Next: submit the 20-job pilot once, monitor only its ledger, and analyze the
reconstruction/activity frontier. If the rectangle has negligible active
coverage while the constraint remains violated, test active-only STE next.

## Checkpoint 4 — 20-job pilot results (2026-07-27)

- Jobs `10224746`–`10224765`: 20/20 `COMPLETED`, exit `0:0`; no training error
  was detected. No job was modified or cancelled.
- LN rectangle bandwidths 0.001 and 0.01 stalled: achieved
  `p_sparsity=0.9971/0.9984`, with only `0.00014/0.00152` of code entries in the
  surrogate window and active-set coverage below 0.001.
- LN bandwidth 1.0 was the sparsest pilot point:
  reconstruction `0.000953`, achieved `p_sparsity=0.0440`, rectangle support
  `0.985`, active coverage `0.660`. The nearest L1 achieved activity was
  `0.0691` at reconstruction `0.000937`; these are compared by achieved
  activity, not target.
- LN bandwidth 4.0 reached `p_sparsity=0.0869`, reconstruction `0.000597`; the
  nearest L1 point reached `0.0833`, reconstruction `0.000591`, so L1 slightly
  dominates that point. Bandwidth 0.1 is also dominated.
- `norm=none` did not produce useful sparsity in 2,000 epochs: all L1 points
  achieved about `p=0.75`, and step-L0 ranged from `0.75` to nearly `1.0`.
- No configuration achieved the step-L0 ceiling 0.005. The results reject
  narrow rectangles and `norm=none`; LN/bandwidth 1.0 remains the rectangular
  candidate.

## Checkpoint 5 — active-only fallback (2026-07-27)

- Added `step_l0_surrogate={rectangle,active_only}`. Active-only uses exactly
  the same strict hard-threshold forward indicator and inequality constraint,
  but its backward estimator reaches every active code entry.
- Focused tests verify forward identity, strict threshold semantics, gradients
  for far-above-threshold signed codes, invalid configuration handling, and
  distinct run hashing. Full suite: `18 passed`.
- A full local 512-dimensional run was stopped because CPU throughput was not a
  useful validation route; it made no repository or cluster changes. The
  fallback will instead be evaluated in the required confirmation.
- Prepared and dry-ran nine post-pilot confirmation jobs on the two-concept
  `2-binary` dataset: L1 reference, LN rectangle/bandwidth 1.0, and LN
  active-only, each at seeds 0/1/2. This is separate from the already capped
  20-job pilot.

Next: run the nine confirmation jobs, aggregate tail metrics across three
seeds, and select the estimator using reconstruction versus achieved
`p_sparsity`.

## Checkpoint 6 — three-seed multi-concept confirmation (2026-07-27)

- Dataset: `2-binary`/Gemma-2-2B layer 25, 485 training pairs, two concepts.
  All candidates used LN, `activation=none`, the strict hard forward indicator,
  inequality mode, threshold 0.01, warmup 100, schedule 500, and 2,000 epochs.
- The first active-only seed (`10224992`) exposed a progress-formatting
  `KeyError`: the formatter requested rectangle-only fields. The estimator had
  entered training correctly. The formatter was extracted into a tested helper,
  active-only support is now logged analytically, and only seed 0 was replaced
  (`10225725`). The original failed ID remains in the accounting record.
- Final confirmation accounting: all nine selected runs completed at exit
  `0:0` (three L1, three rectangle, three active-only). Saved configs and weights
  record distinct surrogate hashes and exact settings.

Tail-100 mean ± between-seed population standard deviation:

| estimator | achieved `p_sparsity` | reconstruction |
| --- | ---: | ---: |
| L1 target 0.001 | 0.01906 ± 0.00062 | 0.0004905 ± 0.0000105 |
| active-only step-L0 | 0.03426 ± 0.00243 | 0.0005203 ± 0.0000222 |
| rectangle step-L0, bandwidth 1.0 | 0.05043 ± 0.00120 | 0.0003027 ± 0.0000080 |

- Active-only improved activity over the rectangle but is dominated by the L1
  reference on both tail-mean axes. It is therefore rejected for this tested
  configuration, not universally.
- The LN rectangle at bandwidth 1.0 is the preferred hard-step/STE
  configuration among those tested: it is stable across three seeds and is
  nondominated among the three reported points. Relative to L1, it has about
  38% lower reconstruction error but uses 2.65 times as many active features.
  This is not a matched-sparsity comparison, and the observed three-point
  frontier is not a complete L1/step-L0 Pareto curve.
- Important limit: no step-L0 run satisfied the configured 0.005 ceiling within
  2,000 epochs. The candidate is gradient-covered and Pareto-nondominated among
  the tested points, not a calibrated target-achieving solver or a model proven
  superior to tuned L1 at matched achieved sparsity. Its activity was still
  decreasing at epoch 1,999, so claims of exact constraint satisfaction require
  a dual/schedule study.

## Checkpoint 7 — completion audit (2026-07-27)

- Re-ran the complete repository test suite: `20 passed`. Fatal-error Ruff
  checks, Python compilation, and `git diff --check` also passed.
- Re-queried SLURM accounting: all 20 pilot jobs completed with exit `0:0`.
  The nine selected confirmation runs also completed with exit `0:0`; the
  superseded active-only seed-0 job is preserved as `FAILED 1:0` and its
  replacement as `COMPLETED 0:0`.
- Re-generated both summaries directly from their submitted manifests and raw
  logs. The analyzers found 20/20 complete pilot runs and 9/9 complete selected
  confirmation runs, reproducing the tail-100 aggregates above.
- Kept the global rectangular bandwidth default at `1e-3`. Bandwidth `1.0` is
  the supported recommendation for this tested LayerNorm/Gemma setup, not a
  generally calibrated default; the study did not demonstrate target
  satisfaction or test enough datasets/training budgets to justify a global
  change. The estimator-level recommendation at this checkpoint is superseded
  by the matched-sparsity results below.

## Checkpoint 8 — schedule and claim audit (2026-07-27)

- Audited the six serialized step-L0 `cfg.yaml` files from the multi-concept
  confirmation. Every rectangle and active-only run records `warmup: 100`,
  `schedule: 500`, `epochs: 2000`, `target: 0.005`, `ind_th: 0.01`,
  `activation: none`, and `constraint_mode: inequality`; the run did not use
  the longer local defaults.
- The raw rectangle logs show target 1.0 at epoch 100, 0.00699 at epoch 599,
  and the final 0.005 target from epoch 600 through epoch 1999. Across the
  three seeds, the tail-100 means are `p_sparsity=0.05043294` and
  `constraint_violation=0.04543295`, consistent with
  `0.05043294 - 0.005`.
- Bandwidth 1.0 is broad relative to threshold 0.01: the rectangular backward
  window covers `0 <= |h| <= 0.51`. This gives an L1-like signed gradient over
  a broad interval, while the forward constraint remains the strict hard
  indicator `|h| > 0.01`; it is an STE choice, not a softened forward
  constraint.
- Interim conclusion before the matched L1 sweep: rectangle bandwidth 1.0 was
  the preferred hard-step/STE configuration among those tested—stable,
  gradient-covered, and Pareto-nondominated among the then-reported points. It
  was not target-calibrated or proven superior to tuned L1 at matched achieved
  sparsity.

Next: sweep L1 targets to bracket achieved activity near 0.034 and 0.050, then
sweep step-L0 dual strength and schedule to reduce activity from about 0.050
toward 0.005. Bandwidth annealing remains optional rather than the default next
experiment.

## Checkpoint 9 — matched L1 and optimizer follow-up (2026-07-27)

- Submitted exactly 20 seed-0 follow-up jobs in sequence: seven coarse L1
  targets, four refined L1 targets, then nine rectangle dual/schedule settings.
  All 20 completed with exit `0:0`; no unrelated jobs were modified or
  cancelled.
- The refined L1 sweep produced the requested activity matches on the same
  two-concept dataset and 2,000-epoch schedule:
  `target=0.0075` reached `p_sparsity=0.03378` with reconstruction
  `0.0002528`, and `target=0.00825` reached `p_sparsity=0.04937` with
  reconstruction `0.0002293`.
- The nine step-L0 runs kept bandwidth 1.0 and varied
  `schedule={100,250,500}` with `dual_lr_div={0.25,0.5,1.0}`. Serialized
  configs and final logs confirm the intended settings and final target 0.005.
  Achieved activity ranged from 0.02949 to 0.04172, but reconstruction worsened
  from `0.0004459` to `0.0022449` as pressure increased. The smallest violation
  was still 0.02449 (`p_sparsity=0.02949`), so no setting approached the 0.005
  ceiling within 2,000 epochs.
- Every retuned rectangle point is dominated by the measured seed-0 L1 curve.
  For example, the sparsest rectangle point has
  `(p,reconstruction)=(0.02949,0.0015939)`, while L1 reaches
  `(0.02948,0.0002672)`.
- Added the two missing seeds for L1 target 0.00825 as a post-pilot
  confirmation. Across seeds 0/1/2, L1 achieves
  `p_sparsity=0.048414 ± 0.000719` and reconstruction
  `0.00024038 ± 0.00000803`. The already confirmed rectangle bandwidth-1.0
  point achieves `0.050433 ± 0.001203` and
  `0.00030267 ± 0.00000795`, respectively. L1 has both lower activity and lower
  reconstruction error for every paired seed; rectangle reconstruction is
  25.9% higher on the three-seed mean.
- Final accounting reports all 20 follow-up jobs and both post-pilot
  confirmation jobs as `COMPLETED 0:0`. Validation passed with `34` repository
  tests, fatal-error Ruff checks, Python compilation, shell syntax checks, and
  `git diff --check`.

Final evidence-based conclusion: rectangle bandwidth 1.0 remains the best
hard-step/STE configuration from the original bandwidth sweep and is stable and
gradient-covered, but it is not the preferred estimator. Once L1 is tuned by
achieved activity, L1 dominates both the original and retuned rectangle points.
The hard-step estimator also remains materially above the requested 0.005
ceiling. The concrete blocker is therefore optimization/target calibration
within this 2,000-epoch budget, not missing gradient coverage.
Bandwidth annealing could be studied later, but the present experiments do not
justify it ahead of reporting the tuned-L1 baseline.
