---
title: Reproduce
description: Install SSAE, train one in ten minutes on a laptop, and rebuild every figure on this site from frozen sources.
---

<p class="kicker">Reproduce</p>

# Run it yourself

<p class="lede">Three paths, in increasing order of cost. Rebuilding the figures
on this site needs no GPU and no data. Training a small SSAE end to end takes
about ten minutes on a laptop. Reproducing a published study needs a cluster,
and the job scripts are in the repository.</p>

## Install

```bash
git clone https://github.com/shrutij01/ssae.git
cd ssae

python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --editable '.[results]'
```

The dependency list pulls PyTorch, Transformers, Datasets, `cooper-optim`, and
Weights & Biases. Install `cooper-optim`, not the unrelated PyPI package named
`cooper`. Python 3.10 or newer is required; a GPU is not, for anything on this
page except a full study.

Two environment variables matter. `SSAE_DATA_ROOT` has no default and
extraction fails without it:

```bash
export SSAE_DATA_ROOT="$PWD/.ssae-data" && mkdir -p "$SSAE_DATA_ROOT"
export WANDB_MODE=offline
```

Confirm the core model imports and runs:

```bash
python3 - <<'PY'
import torch
from ssae.model import DictLinearAE
model = DictLinearAE(rep_dim=8, hid=4, norm_type="none", input_norm="none")
x_hat, code = model(torch.randn(3, 8))
assert x_hat.shape == (3, 8) and code.shape == (3, 4)
print("SSAE forward pass: OK")
PY
```

## Path 1 · Rebuild every figure (no GPU, no data)

Everything on the [results pages](../results/index.md) regenerates from frozen,
hashed bundles committed to the repository.

```bash
python3 -m experiments.build_results --check   # validate sources
python3 -m experiments.build_results --build   # regenerate figures and tables
python3 scripts/generate_experiment_pages.py --write
mkdocs serve
```

Narrow to one experiment with `--only exp02`. The builders never
retrain; they read frozen bundles and regenerate presentation, which is what
makes the artifact catalog auditable. [Artifact details →](../results/artifacts.md)

## Path 2 · Train an SSAE in ten minutes

A complete pipeline check on the smallest available model. The numbers it
produces are not comparable with the frozen catalog — the schedule is far too
short — but every stage is the real one.

**Extract paired activations.** Pythia-70M downloads in under a minute.

```bash
python3 -m ssae.store_embeddings \
  --dataset 2-binary \
  --model_id EleutherAI/pythia-70m-deduped \
  --layer 5 \
  --num-samples 40 \
  --batch-size 16 \
  --pooling-method last_token

EMB="$SSAE_DATA_ROOT/2-binary/2-binary_pythia70m_5_last_token.h5"
CFG="${EMB%.h5}.yaml"
```

The extractor writes the HDF5 file and a YAML sidecar recording model, layer,
representation dimension, concept count, and pooling rule. Both are inputs to
training. [Extraction reference →](data.md)

**Train a constrained SSAE.**

```bash
python3 -m ssae.training "$EMB" "$CFG" \
  --oc 512 --sparsity-type l1 --target 0.1 \
  --epochs 200 --warmup 20 --schedule 80 \
  --no-use-amp \
  --output-dir "$SSAE_DATA_ROOT/runs/quickstart_seed0"
```

`--target 0.1` bounds the **mean absolute code value**, because `--sparsity-type`
is `l1`. It does not mean that 10% of coordinates are active; the two statistics
have different units and are not interchangeable. [Training reference →](train.md)

**Evaluate held-out recovery.**

```bash
python3 -m experiments.legacy.semi_synth \
  --embeddings "$EMB" \
  --model "$SSAE_DATA_ROOT/runs/quickstart_seed0" \
  --dataset 2-binary \
  --linear-probe --auc --support-recovery --heldout-eval --skip-diagnostics \
  --output "$SSAE_DATA_ROOT/runs/quickstart_metrics.jsonl"
```

This loads the stored split and labels, prints a summary, and writes one JSONL
record per metric.

**Train the baseline you should compare against.** The comparison that isolates
the contribution is the same architecture with the constraint removed, which is
[experiment 06](../results/synthetic.md)'s design:

```bash
python3 -m ssae.sae "$EMB" "$CFG" --sae-type relu --seed 0 --train-on-diffs
```

Adding `--train-on-diffs` makes this an SSAE by target; omitting it gives a
raw-activation SAE. That flag is the whole distinction between the families.

## Path 3 · Reproduce a published study

Each experiment has its own SLURM chain and a committed frozen bundle. The job
scripts hardcode Mila cluster paths and need editing before use.

| Experiment | Entry point |
|---|---|
| 01, 02 · semi-synthetic | `job_scripts/semisynth_heldout_2k_v1/` — `embed_array.sh` → `train_array.sh` (210 runs) → `prepare_and_submit_metrics.sh` |
| 01, 02 · behavioural | `scripts/run_canonical_15k_study.py dry-run\|submit\|status`, then `evaluate_canonical_15k_metrics.py prepare\|worker\|consolidate` |
| 03 · steering | `python3 -m experiments.exp03.run evaluate`, then `freeze`, then `python3 -m experiments.exp03.report --build` |
| 04 · transfer | `sbatch scripts/job_scripts/job_semisynth_ood_transfer_results.sh` — evaluate, freeze, build, check in one job |
| 05 · atlas | `scripts/run_semisynth_feature_atlas_study.py prepare\|submit\|worker\|consolidate\|publish` |
| 06 · synthetic | `python3 -m experiments.exp06.run plan`, then `run --all --device cpu`, then `aggregate` |
| 07 · target-side overshoot | `python3 -m experiments.exp07.run evaluate`, then `freeze`, then `python3 -m experiments.exp07.report --build` |

Experiment 06 runs on CPU without a cluster if you have the patience: 30 runs of
100 epochs over 100,000 samples.

```bash
python3 -m experiments.exp06.run run --all --device cpu \
  --output-root results/exp06/runs/synthetic_identification_v1
python3 -m experiments.exp06.run aggregate
python3 -m experiments.exp06.report --build
```

Experiment 07 does not retrain, but the end-to-end evaluator needs the frozen
v3 checkpoint and held-out-activation paths recorded in the shared inventory.
The committed frozen bundle is sufficient to rebuild its report without those
scratch files:

```bash
python3 -m experiments.exp07.run evaluate
python3 -m experiments.exp07.run freeze
python3 -m experiments.exp07.report --build
```

Experiment 03 likewise reuses the frozen v3 checkpoint inventory. It selects,
orients, and calibrates one decoder direction per concept from `cfc_train`, then
adds and subtracts that vector from the same `cfc_test` source activation and
scores both against the same paired target for every checkpoint:

```bash
python3 -m experiments.exp03.run evaluate
python3 -m experiments.exp03.run freeze
python3 -m experiments.exp03.report --build
```

## Configuration

There is no Hydra and there are no console entry points; everything runs as
`python3 -m <module>`. Configuration is a frozen dataclass, `ssae.config.Cfg`,
merged from the extractor's YAML sidecar and CLI flags, with `hash_cfg()`
producing a run identity over the fields that define it. Frozen study
configurations are JSON, under `experiments/*/configs/`.

Run names encode the estimator, and reading them saves time:

```text
ssae_l1_ln_t0.005_d2_e2000_absolute_instandardize
     │   │   │       │  │        │
     │   │   │       │  │        └─ input normalisation: standardize
     │   │   │       │  └─ loss scale: absolute
     │   │   │       └─ 2000 epochs
     │   │   └─ constrained L1 target 0.005
     │   └─ layer normalisation on the code
     └─ constrained L1 sparsity
```
