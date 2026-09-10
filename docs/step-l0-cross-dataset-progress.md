# SSAE hard-step cross-dataset progress

## Checkpoint 1 — audit and isolated tooling (2026-07-27)

- Preserved the dirty worktree without reverting or staging anything. Existing
  changes include three tracked documentation edits plus unrelated untracked
  experiment outputs and generated files.
- Verified the requested Gemma-2-2B-it layer-25 last-token HDF5/YAML pairs:
  refusal (412 training pairs), sycophancy (18,165), truthful-qa (735), and
  bias-in-bios (900). All four current YAML files declare `num_concepts: 1`.
  The new launcher records that value and deliberately does not pass
  `--n-concepts`, so a future multi-concept YAML remains authoritative.
- Historical step-L0 runs outside refusal either used older code/settings or
  lack the current stable diagnostic fields. They cannot substitute for the
  requested reruns. The prior 2-binary result remains context: bandwidth 1.0
  was stable and gradient-covered, but tuned L1 dominated at matched achieved
  activity and the 0.005 ceiling was not reached.
- Audited the committed implementation. The forward constraint is the strict
  hard indicator `|h| > 0.01`. With threshold 0.01 and bandwidth 1.0, the
  rectangular backward window is `0 <= |h| <= 0.51`; this does not soften the
  forward constraint. The tracker uses the same strict reporting threshold.
- Added a separate cross-dataset launcher and raw-log analyzer. Each invocation
  has an isolated manifest and submission ledger; a study-wide ledger caps
  seed-0 calibration/fallback submissions at 32, L1 at six targets per dataset,
  and confirmation at 16 jobs. The analyzer audits saved `cfg.yaml`, computes
  tail-100 and final diagnostics, verifies estimator-specific violation
  identities, and matches only by achieved `p_sparsity` within each
  dataset/seed.
- Focused validation passed: 24 tests across the new tooling, hard-step
  implementation, and existing analyzer; fatal-error Ruff and Python
  compilation passed. The four rectangle jobs were dry-run, every generated
  script passed `bash -n`, and manual command/manifest inspection confirmed all
  fixed settings. A representative SLURM `--test-only` request was accepted.
- Cluster audit initially showed ten unrelated running `catstd_l1pilot` array
  tasks (`10230021_[0-9]`); they were not modified or cancelled. Subsequent
  controller reachability became intermittent. Three real submission attempts
  and a later `--test-only` retry failed before any job ID was issued. The first
  failure exposed and prompted a tested fix for safe retry after a header-only
  ledger. Current study accounting remains **0 seed-0 jobs submitted** and no
  unrelated job has been touched.

Next: submit only the four audited rectangle jobs when the controller accepts
writes, monitor those ledger IDs, audit raw logs and saved configs, then choose
dataset-specific L1 targets adaptively.

## Checkpoint 2 — partial seed-0 calibration and Truthful-QA confirmation

- Directly submitted the four rectangle runs as `10230227`, `10230231`,
  `10230246`, and `10230248`; all are reconciled in both ledgers. Direct
  `sbatch` was required because nested submission from the launcher inherited a
  restricted network path. A tested recorder now enforces the same caps and
  atomically reconciles direct job IDs with dry-run manifests.
- Refusal, truthful-qa, and bias-in-bios rectangle runs completed at epoch 1999
  with exact saved-config audits and no fatal log errors. Tail-100 diagnostics:

| dataset | reconstruction | `p_sparsity` | mean `|h|` | violation | dual | rectangle support | active coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| refusal | 0.00095260 | 0.04399996 | 0.0742240 | 0.03899996 | 2.79180 | 0.98505 | 0.66016 |
| truthful-qa | 0.00082757 | 0.05382212 | 0.0597181 | 0.04882212 | 5.08992 | 0.97718 | 0.57604 |
| bias-in-bios | 0.00078158 | 0.04887499 | 0.0224503 | 0.04387499 | 5.63199 | 0.99045 | 0.80456 |

  Each violation agrees with `p_sparsity - 0.005` to within `3e-9`. None
  reached the 0.005 ceiling. Active coverage is substantial, so none meets the
  requested condition for an active-only fallback.
- Submitted nine dry-run/syntax/test-only-audited seed-0 L1 jobs
  (`10230270`–`10230278`). Study seed-0 accounting is 13/32. Refusal used
  `{0, 0.0005, 0.001}` based on its existing same-setting curve; truthful-qa
  and bias-in-bios used independent coarse anchors `{0, 0.005, 0.01}`.
  All saved configs passed. The L1 violation identity used
  `mean_abs_activation - target`, never `p_sparsity - target`.
- Refusal cannot be closely matched: the strongest valid L1 target 0 reaches
  `p=0.06888281`, a gap of 0.02488285 above rectangle. Bias-in-bios has the
  same boundary blocker: L1 target 0 reaches `p=0.06665554`, a gap of
  0.01778055. Negative L1 targets are invalid, so further interpolation cannot
  bracket either rectangle point under the fixed optimizer/schedule.
- Truthful-QA L1 target 0 is usable: seed-0
  `(p,reconstruction)=(0.05507728,0.00073750)` versus rectangle
  `(0.05382212,0.00082757)`, absolute activity gap 0.00125516. Submitted only
  the missing seeds 1 and 2 for both estimators (`10230290`–`10230293`);
  confirmation accounting is 4/16.
- Truthful-QA three-seed tail-100 aggregates (population standard deviation):
  rectangle `p=0.05511883 ± 0.00264524`, reconstruction
  `0.00086235 ± 0.00003691`; L1 `p=0.05509359 ± 0.00005229`,
  reconstruction `0.00075332 ± 0.00003385`. All three activity gaps are below
  0.005. L1 strictly dominates seed 2 and the three-seed mean; seeds 0 and 1
  are strict Pareto tradeoffs because L1 reconstructs better but has
  `p_sparsity` higher by 0.00126/0.00231.

Next: finish the longer sycophancy rectangle run, calibrate its L1 target
independently, and confirm missing seeds if a close match exists.

## Checkpoint 3 — sycophancy calibration and confirmation launch

- Sycophancy rectangle job `10230231` completed normally at epoch 1999.
  Its tail-100 diagnostics are reconstruction `0.03075878`,
  `p_sparsity=0.0074683444`, mean absolute activation `0.038580502`,
  logged target `0.005`, violation `0.0024683446`, dual `98.745425`,
  rectangle support `0.99702588`, and active-set rectangle coverage
  `0.60176564`. The violation identity gap is `2e-10`. The 0.005 ceiling
  was not reached, but the substantial active coverage rules out the
  specified active-only fallback condition.
- The independent L1 calibration used targets `{0, 0.0025, 0.005}` in jobs
  `10230337`–`10230339`. All completed normally at epoch 1999, all saved
  configs passed, and all tail-100 L1 violations agree with
  `mean_abs_activation - target` within `4e-10`.
- The closest achieved-activity point is L1 target `0.0025`:
  `p_sparsity=0.002775117`, reconstruction `0.62381552`, mean absolute
  activation `0.0027193793`, violation `0.00021937939`, and dual
  `95.49058`. Its absolute `p_sparsity` gap from rectangle is
  `0.0046932274`, inside the requested `0.005` tolerance. This selection is
  based on achieved activity, not equality of configured targets.
- Dry-ran and syntax-checked the four missing sycophancy confirmation jobs.
  SLURM `--test-only` accepted representative rectangle and L1 scripts.
  Submitted rectangle/L1 seeds 1 and 2 as `10230575`–`10230578` and
  reconciled every ID into the isolated and study-wide ledgers. Seed-0
  accounting remains **16/32** and confirmation accounting is now **8/16**.

Next: monitor only ledger-owned jobs `10230575`–`10230578`; audit their
saved configs, final epochs, SLURM states, and raw-log aggregates; then record
the final dataset-specific conclusions and repository validation.

## Checkpoint 4 — final audit and conclusions

- Sycophancy confirmation jobs `10230575`–`10230578` all completed with
  SLURM state `COMPLETED`, exit code `0:0`, final epoch 1999, exact saved
  configs, and no fatal raw-log signatures. Per-seed tail-100 pairs:

| seed | rectangle `(p, recon)` | L1 target 0.0025 `(p, recon)` | `|p|` gap | tested-point relation |
| ---: | --- | --- | ---: | --- |
| 0 | `(0.00746834, 0.03075878)` | `(0.00277512, 0.62381552)` | 0.00469323 | tradeoff |
| 1 | `(0.00766287, 0.03059183)` | `(0.00283337, 0.62191096)` | 0.00482949 | tradeoff |
| 2 | `(0.00758499, 0.03165784)` | `(0.00253686, 0.62512051)` | 0.00504813 | tradeoff |

  Seed 2 exceeds the preferred matching tolerance by only `0.00004813`; the
  selected fixed target was the closest seed-0 calibration point and was not
  retuned by seed. The three-seed mean is closely matched:
  rectangle `p=0.00757207 ± 0.00007994`, reconstruction
  `0.03100282 ± 0.00046816`; L1 `p=0.00271512 ± 0.00012827`,
  reconstruction `0.62361566 ± 0.00131789`. The population-mean activity gap
  is `0.00485695`. Every seed and the mean are Pareto tradeoffs: rectangle
  reconstructs much better, while L1 is sparser. Rectangle is therefore
  **nondominated only among these tested points**, not a universal frontier.
- Truthful-QA's audited comparison remains: rectangle
  `p=0.05511883 ± 0.00264524`, reconstruction
  `0.00086235 ± 0.00003691`; L1 target 0
  `p=0.05509359 ± 0.00005229`, reconstruction
  `0.00075332 ± 0.00003385`. Seeds 0 and 1 are tradeoffs, seed 2 is strictly
  L1-dominated, and L1 strictly dominates the closely matched three-seed
  mean. The dataset conclusion is **rectangle dominated**.
- Refusal remains **unresolved because of a concrete calibration blocker**.
  Rectangle achieves `p=0.04399996`; the strongest admissible L1 inequality
  target, 0, achieves `p=0.06888281`, leaving a `0.02488285` gap. Negative
  L1 targets are invalid. Rectangle and L1 are a tradeoff at these unmatched
  points, so this is not evidence of dominance.
- Bias-in-bios is likewise **unresolved because of a concrete calibration
  blocker**. Rectangle achieves `p=0.04887499`; L1 target 0 achieves
  `p=0.06665554`, leaving a `0.01778055` gap. The unmatched points are a
  tradeoff and cannot support a matched-activity conclusion.
- No rectangle run reached the 0.005 step-L0 ceiling: tail-100
  `p_sparsity` is 0.04400 (refusal), 0.00757 ± 0.00008 (sycophancy),
  0.05512 ± 0.00265 (truthful-qa), and 0.04887 (bias-in-bios). For every
  step-L0 run, violation agrees with `p_sparsity - 0.005`; for every L1 run,
  it agrees with `mean_abs_activation - target`. Rectangle support and
  active coverage remain substantial, so no active-only fallback was
  justified.
- Final accounting is **16/32 seed-0 jobs** and **8/16 confirmation jobs**,
  24 submitted jobs total. All 24 are represented exactly once in the
  isolated and study-wide ledgers. No fallback or replacement jobs were
  submitted, and no unrelated cluster job was modified or cancelled.
- The reconstruction/activity evidence alone does not support step-L0 as a
  generally competitive estimator. It establishes a correctly implemented,
  stable hard-step/rectangular-STE experiment: the strict hard forward
  indicator is preserved and the broad backward window is well covered, but
  the requested ceiling is missed everywhere, Truthful-QA reconstruction is
  dominated by L1, sycophancy is only nondominated among the tested points,
  and the other two datasets remain calibration-blocked. These points do not
  establish a universal Pareto frontier. The later MCC follow-up below adds
  separate, mixed evidence about concept alignment.
- Final validation passed: 42 repository tests, fatal-error Ruff on every
  study file, full-tree Python compilation, `bash -n` on the launcher and
  every generated job script, and `git diff --check`. Full-tree fatal-error
  Ruff still reports four unrelated pre-existing F821 findings in
  `ssae/train_online.py`. That out-of-scope file was not modified.

## Checkpoint 5 — held-out SAE MCC follow-up

- The reconstruction study did not run identifiability evaluation, so the
  selected checkpoints initially had no MCC artifacts. Added a focused
  evaluator using the repository's canonical SAE MCC: Hungarian-matched
  absolute Pearson correlation between held-out raw-embedding codes and
  concept labels. All four datasets currently have one concept.
- CPU validation on refusal reproduced the canonical evaluation path; the
  GPU result differed by less than `2e-8`. The isolated MCC job `10232390`
  completed in SLURM state `COMPLETED`, exit code `0:0`, and evaluated only
  the selected rectangle/L1 checkpoints.
- Held-out MCC results (mean ± population standard deviation):

| dataset | rectangle | selected L1 |
| --- | ---: | ---: |
| refusal | 0.585158 | 0.610915 |
| sycophancy | 0.523060 ± 0.055304 | 0.643915 ± 0.009275 |
| truthful-qa | 0.476762 ± 0.018303 | 0.371632 ± 0.034257 |
| bias-in-bios | 0.767305 | 0.586975 |

  Sycophancy L1 has higher MCC in all three seeds; truthful-qa rectangle has
  higher MCC in all three. Refusal and bias-in-bios remain seed-0,
  activity-unmatched comparisons, so their MCC differences do not resolve
  the earlier calibration blockers. MCC measures concept alignment and does
  not reverse the separately reported reconstruction/activity dominance.
  It does make the estimator conclusion more nuanced: step-L0 merits
  retention as a concept-alignment candidate for further study, especially
  on truthful-qa, but the mixed dataset results and universal ceiling misses
  do not support making it the default estimator.
