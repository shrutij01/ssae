# Test identifiability overfitting

This experiment compares held-out DCI completeness and informativeness across
three predictor families for the Pythia ENG→FR constrained-L1 SSAE sweep:

- elastic-net logistic regression;
- an additive cubic-spline logistic GAM (no feature interactions);
- gradient-boosted trees.

The data split is created before model training. The original 80 training pairs
are deterministically divided into 64 train and 16 validation pairs; the
original 20 test pairs remain byte-for-byte unchanged. Reverse differences are
constructed only after splitting, so the two orientations of one pair can never
cross splits. The split uses the source dataset creator's RNG convention,
`random.Random(data_seed=21)`, including its in-place shuffle operation.

Models and predictors use only the training split for fitting. Predictor
hyperparameters are selected on validation. The final figure reports only the
untouched test split. DCI-I is test accuracy. Test DCI-C is computed from
pair-blocked permutation importance on test, then passed through
`identifiability_guard`'s completeness functional.

That DCI-C estimator is deliberately labeled as held-out permutation DCI-C.
It is not the package's usual training-derived, model-native importance score:
ordinary DCI-C has no evaluation split. For each test-code coordinate, the
permutation moves values only in complete forward/reverse pair blocks and adds
a random orientation flip. This preserves the antipodal pair structure while
breaking that coordinate's association with the concept direction.

Protocol:

1. Run `make_split.py` once.
2. Submit `train_target_sweep.sh` to retrain four targets × three seeds on only
   the 64 training pairs.
3. Run `compare_dci_predictors.py` after all 12 checkpoints finish. Predictors
   are fit on train, hyperparameters are chosen by validation accuracy, and
   neither predictor fitting nor selection uses test.

Generated artifacts live under
`results/findings/identifiability_overfitting_v1/report/`; large data,
checkpoints, and cluster logs live under
`/path/to/ssae-data/studies/test_identifiability_overfitting_v1/`.

## Completed ENG→French result

Values are test mean ± sample SD across the three independently trained SSAE
seeds. Each cell is `DCI-C / DCI-I`.

| Predictor | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|---:|
| Elastic Net | .309±.178 / .875±.050 | .877±.131 / .642±.104 | .742±.135 / .925±.000 | .903±.082 / .908±.029 |
| Spline GAM | .174±.026 / .875±.050 | .159±.003 / .925±.050 | .187±.018 / .958±.029 | .198±.035 / .958±.029 |
| GBT | .806±.108 / .558±.240 | .871±.122 / .692±.076 | .858±.004 / .858±.063 | .909±.117 / .908±.029 |

The main finding is predictor sensitivity, not a predictor-independent ranking:
the GAM recovers the concept direction very accurately while distributing its
held-out importance across many coordinates, whereas the GBT assigns
importance to very few coordinates but is unstable at the two smallest
targets. Thus DCI-C here describes recovery under a specified probe family; it
is not an intrinsic property of the representation by itself.

The x-axis contains configured targets. Target 0.02 ended at mean achieved L1
`0.02291` and was not fully feasible; maximum decoder-column norm error over
all 12 checkpoints was `1.20e-7`. The test set has only 20 independent pairs
(40 oriented rows), so seed variation and the predictor comparison should be
reported, not hidden behind a single pooled score.

Primary inspection files:

- `predictor_comparison/dci_predictor_comparison_eng_french.pdf`
- `predictor_comparison/dci_predictor_comparison_eng_french.svg`
- `predictor_comparison/per_run_test_dci.csv`
- `predictor_comparison/summary_test_dci.csv`
- `predictor_comparison/validation_grid.csv`
- `predictor_comparison/protocol.json`

## Train / validation / test findings

The all-metric page freezes every alignment or fitted map learned on train:
MCC-P/MCC-S feature assignments, the R² linear map, and seed-to-seed response
alignment. Elastic Net, spline GAM, and GBT probes are fit on train. Spline GAM
is selected once, globally, because it has the highest mean validation DCI-I
across the 12 target/seed runs (`1.000`, versus `.885` for Elastic Net and
`.779` for GBT). Its within-family hyperparameters are also chosen on
validation. No test score participates in either selection, so only the test
column is fully untouched.

Means ± sample SD use three SSAE seeds; Seedwise MCC instead uses the three
seed pairs.

| Target | Split | MCC-P | DCI-C | DCI-I | MCC-S | R² | Seedwise MCC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.02 | train | .423±.128 | .437±.172 | .951±.059 | .841±.018 | .984±.000 | .854±.005 |
| 0.02 | validation | .618±.133 | .492±.460 | 1.000±.000 | .106±.099 | .626±.015 | .705±.012 |
| 0.02 | test | .385±.271 | .174±.026 | .875±.050 | .108±.090 | .667±.008 | .684±.001 |
| 0.05 | train | .745±.069 | .269±.016 | .966±.033 | .846±.014 | .984±.000 | .823±.002 |
| 0.05 | validation | .375±.312 | .393±.347 | 1.000±.000 | .663±.153 | .724±.015 | .684±.007 |
| 0.05 | test | .304±.158 | .159±.003 | .925±.050 | .421±.250 | .696±.021 | .667±.007 |
| 0.1 | train | .753±.059 | .229±.045 | .987±.009 | .835±.017 | .984±.000 | .696±.004 |
| 0.1 | validation | .798±.066 | .267±.463 | 1.000±.000 | .801±.058 | .722±.038 | .621±.010 |
| 0.1 | test | .821±.026 | .187±.018 | .958±.029 | .819±.014 | .699±.019 | .620±.004 |
| 0.2 | train | .796±.061 | .202±.079 | .992±.000 | .829±.028 | .984±.000 | .504±.002 |
| 0.2 | validation | .873±.023 | .544±.475 | 1.000±.000 | .794±.059 | .712±.016 | .444±.004 |
| 0.2 | test | .830±.025 | .198±.035 | .958±.029 | .838±.008 | .690±.021 | .442±.008 |

The main findings are:

- Stable coordinate-wise concept recovery appears at targets 0.1 and 0.2.
  Test MCC-P/MCC-S jump from `.304/.421` at 0.05 to `.821/.819` at
  0.1, while their seed SDs collapse.
- Test R² remains almost flat (`.667–.699`) across targets. The language
  concept is therefore linearly recoverable at every target, but at 0.02/0.05
  it is distributed or assigned to train-specific coordinates rather than a
  stable single coordinate.
- Validation optimism consistent with predictor-selection overfitting is
  visible in DCI-I. The selected GAM scores `1.000` on validation at every
  target but only `.875`, `.925`, `.958`, and `.958` on untouched test. The
  largest optimism is therefore `.125` at target 0.02; the gap shrinks as the
  L1 target increases. One fixed split cannot by itself separate selection
  overfitting from validation/test sampling difficulty.
- GAM DCI-C is not stable on the 16-pair validation split: its seed SD reaches
  `.460–.475`, while untouched-test DCI-C stays in the narrow `.159–.198`
  range. This is evidence that validation DCI-C is a high-variance estimator
  here, not evidence that the test representations abruptly lose a compact
  concept coordinate. GAM's high test DCI-I with low test DCI-C instead says
  that its predictive signal is distributed across coordinates.
- The clearest concept-coordinate generalization gap is at low targets. At 0.05,
  train-to-test MCC-P falls from `.745` to `.304` and MCC-S from `.846` to
  `.421`; at 0.02, MCC-S falls from `.841` to `.108`. At targets 0.1 and 0.2,
  train and test coordinate metrics agree closely. This page alone cannot
  distinguish SSAE overfitting from instability in the train-selected
  coordinate assignment.
- Seedwise MCC moves in the opposite direction from concept-specific recovery:
  test seedwise MCC falls from `.684` at 0.02 to `.442` at 0.2 while MCC-P,
  MCC-S, and DCI-I improve. Global encoder-coordinate agreement across seeds
  is therefore not a proxy for recovery of the language coordinate.
- Target 0.1 is the first setting where validation and test agree across
  MCC-P, MCC-S, and DCI-I. Target 0.2 raises DCI-I slightly but sacrifices
  substantially more global seedwise agreement; choosing between them depends
  on whether concept recovery or whole-representation stability is primary.

Inspection files are under `split_metrics/`:

- `identifiability_train_validation_test_eng_french.pdf`
- `identifiability_train_validation_test_eng_french.svg`
- `per_seed_split_metrics.csv`
- `seedwise_mcc.csv`
- `summary_split_metrics.csv`
- `dci_validation_grid.csv`
- `dci_predictor_selection.csv`
- `protocol.json`
