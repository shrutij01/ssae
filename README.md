# Sparse Shift Autoencoders: Identifying Concepts from LLM Activations

A research framework for training and evaluating **Sparse Shift Autoencoders (SSAEs)** on language model representations, with a focus on steering concepts, probing model behavior, and achieving identifiable representations.

Latest version of the paper at [OpenReview](https://openreview.net/forum?id=dGQubVJQx6&referrer=%5Bthe%20profile%20of%20Shruti%20Joshi%5D(%2Fprofile%3Fid%3D~Shruti_Joshi1)).

## Setup

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Install SSAE and the identifiability metrics
uv pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
uv pip install -e ".[metrics]"

# SAEBench evaluation (optional)
uv pip install -e ".[bench]"

# Set data root (where embeddings and trained models live)
export SSAE_DATA_ROOT=/path/to/your/data
```

<details>
<summary>Alternative: pip install (without uv)</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[metrics]"
export SSAE_DATA_ROOT=/path/to/your/data
```
</details>

### Cluster setup (SLURM)

```bash
# Copy and edit the environment config for your cluster
cp scripts/job_scripts/env.sh scripts/job_scripts/env.local.sh
# Edit env.local.sh with your paths: SSAE_DATA_ROOT, SSAE_VENV, etc.

# For gated models (Gemma), login to HuggingFace
huggingface-cli login
```

## Repository Structure

```text
ssae/
├── ssae/          core models, constraints, and training
├── evals/         reusable evaluation and model-loading code
├── experiments/   shallow exp01 … exp07 experiment packages
├── results/       runs → frozen inputs → reproducible reports
├── data/          small source datasets, vocabularies, and loaders
├── scripts/       orchestration, conversion, and cluster launchers
├── tests/         unit and frozen-result contract tests
└── docs/          narrative documentation only
```

Experiment code and outputs are deliberately separated. See
[`experiments/README.md`](experiments/README.md),
[`results/README.md`](results/README.md), and
[`docs/STRUCTURE.md`](docs/STRUCTURE.md) for the placement rules.

## Quick Start

### 1. Generate embeddings

```bash
python -m ssae.store_embeddings \
    --dataset eng-french \
    --model_id google/gemma-2-2b-it \
    --layer 25 --num-samples 100
```

### 2. Train SSAE

```bash
python -m ssae.training \
    $SSAE_DATA_ROOT/eng-french/eng-french_gemma2_25_last_token.h5 \
    $SSAE_DATA_ROOT/eng-french/eng-french_gemma2_25_last_token.yaml \
    --seed 0 --epochs 15000
```

### 3. Train penalized SAE/SSAE baselines

```bash
# SAE ReLU (penalized, trained on raw individual embeddings)
python -m ssae.sae data.h5 cfg.yaml --sae-type relu --seed 0

# SSAE ReLU (penalized, trained on paired differences)
# The legacy trainer/flag names are retained for checkpoint compatibility.
python -m ssae.sae data.h5 cfg.yaml --sae-type relu --train-on-diffs --seed 0
```

### 4. Evaluate

```bash
# Identifiability (all metrics)
python -m experiments.legacy.semi_synth \
    --embeddings data.h5 --model path/to/ssae_seed0 \
    --aggregate-seeds --linear-probe --auc --support-recovery --id-guard \
    --output results/exp02/runs/identification.jsonl

# Similarity
python -m experiments.legacy.semi_synth \
    --embeddings data.h5 --model path/to/ssae_seed0 \
    --aggregate-seeds --pca --dict-quality \
    --output results/exp02/runs/similarity.jsonl

# AxBench concept detection (needs GPU)
python -m evals.external.axbench \
    --model path/to/ssae_seed0 \
    --hf-dataset pyvene/axbench-concept500 \
    --llm google/gemma-2-2b-it --layer 20 \
    --output results/findings/axbench/runs/axbench.jsonl
```

### 5. Run all evals + LaTeX tables

```bash
cd scripts/job_scripts
bash job_run_all_evals.sh        # SLURM parallel
bash job_run_all_evals.sh --dry  # preview
python scripts/run_all_evals.py --tables-only  # regenerate tables only
```

## Evaluation Metrics

### Identifiability (`experiments/legacy/semi_synth.py`)

| Metric | Flag | What it measures |
|--------|------|------------------|
| SAE MCC | always | Hungarian-matched Pearson correlation per feature |
| SAE AUC | `--auc` | Hungarian-matched ROC-AUC per feature |
| Probe MCC | `--linear-probe` | MCC of linear probe on raw embeddings |
| Decoder proj MCC | always | MCC of projections onto decoder columns |
| Pairwise MCC | `--pairwise` | Representation stability across seeds |
| Support P/R/F1 | `--support-recovery` | Does the feature fire iff the concept is present? |
| MCC-P, MCC-S | `--id-guard` | identifiability-guard Pearson/Spearman MCC |
| DCI-C, DCI-I | `--id-guard` | Completeness + informativeness |

### Similarity (`experiments/legacy/semi_synth.py`)

| Metric | Flag | What it measures |
|--------|------|------------------|
| Decoder-diff cosine | always | cos(decoder_col, z_tilde - z) |
| PCA alignment | `--pca` | PCA directions vs decoder columns |
| Angular error | `--dict-quality` | Direction accuracy of decoder columns |
| Norm ratio | `--dict-quality` | Magnitude accuracy |
| Frac close | `--dict-quality` | Fraction aligned > 0.9 cosine |

### AxBench (`evals/external/axbench.py`)

| Metric | What it measures |
|--------|------------------|
| AUC-ROC | Concept detection (max-over-tokens activation) |
| AUC-ROC imbalanced | Same with 100:1 neg upsampling |
| Hard-neg accuracy | Detection vs semantically similar distractors |

See [the AxBench evaluation guide](docs/eval/axbench.md).

## Concept-to-Feature Mapping

Unified API for any data format:

```python
from evals.concept_mapping import prepare_from_embeddings, map_concepts

# Few-shot: "which SAE feature detects French?"
concepts = prepare_from_embeddings(sae, [french_emb], [english_emb], ["french"])
result = map_concepts(concepts, method="auc", matching="greedy")
```

Three input formats: `prepare_from_activations` (contrastive pairs), `prepare_from_embeddings` (pos/neg sets), `prepare_from_token_embeddings` (per-token with max pooling).

Three scoring methods: `pearson`, `auc`, `mean_diff`.

Two matching strategies: `hungarian` (one-to-one MCC), `greedy` (single concept probing).

## Training Metrics (W&B)

All metrics are logged to Weights & Biases during training. Logged every epoch unless noted.

### Core metrics

| Metric | What it measures | Logged |
|--------|------------------|--------|
| `recon_loss` | Reconstruction MSE | Every epoch |
| `l0_sparsity` | Active features per sample | Every epoch |
| `p_sparsity` | Fraction of all (sample, feature) pairs active | Every epoch |
| `activity_threshold` | Absolute-activation threshold used by both sparsity metrics | Every epoch |
| `constraint_violation` | Sparsity constraint violation (SSAE only) | Every epoch |
| `sparsity_target` | Current annealed target (SSAE only) | Every epoch |
| `dead_fraction` | Fraction of features that never activate | Every 100 epochs |
| `dead_count` / `alive_count` | Raw counts | Every 100 epochs |

### Diversity & assumption monitoring

These track whether the sufficient variability assumption (Assumption 4.1) holds during training. Logged every 100 epochs.

| Metric | What it tests | Model-free? | Robust to overcompleteness? |
|--------|---------------|-------------|---------------------------|
| `diff_effective_rank` | How many independent directions exist in input differences (Assumption 3: span condition) | Yes | Yes |
| `pca_subset_diversity` | Whether PCA components of input differences vary independently — directly tests Assumption 4.1 via model-free subset diversity check | Yes | Yes |
| `pca_unique_patterns` | Number of distinct PCA-based active patterns in input differences | Yes | Yes |
| `code_effective_rank` | Independent directions in learned codes. Handles overcomplete regime: correlated features (same concept subspace) share eigenvalues | No | Partially |
| `unique_supports` | Distinct binary activation patterns. Measures diversity of learned sparsity patterns | No | No — different seeds may spread same concept across different features |
| `max_coactivation` | Worst-case pairwise feature co-firing (1.0 = always co-fire). Flags potential confounding | No | Partially — co-firing could mean "same concept subspace" (fine) or "confounded" (bad) |

The model-free metrics (`diff_effective_rank`, `pca_subset_diversity`) are the most reliable diagnostics. The code-based metrics are informative but depend on the learned representation — interpret with care in the overcomplete regime.

## Training Configuration

### Cooper constrained optimisation

| Flag | Options | Default |
|------|---------|---------|
| `--constrained-optim` | `extrapolation`, `alternating`, `simultaneous` | `extrapolation` |
| `--formulation` | `lagrangian`, `augmented-lagrangian` | `lagrangian` |
| `--penalty-coeff` | float | `1.0` |
| `--dual-optim` | `extra-sgd`, `extra-adam` | `extra-adam` |
| `--sparsity-type` | `l1`, `step_l0` | `l1` |
| `--step-l0-threshold` | float | `0.01` |
| `--step-l0-bandwidth` | float | `1e-3` |

`l1` constrains mean absolute code magnitude; `step_l0` constrains the
threshold-active fraction. Their numeric targets are not directly comparable.
See the [training guide](docs/guide/training.md) for the LayerNorm × constraint
ablation and matched-threshold comparison.

### Legacy penalized trainer: `--train-on-diffs`

This produces a penalized SSAE: model family is determined by the difference
inputs, independently of its ReLU/TopK/JumpReLU activation. Historical
checkpoints keep their `_diffs` tag and `sae_*_reludiffs_*` directory name;
the compatibility wrapper maps them to the canonical four-axis identity.

## Datasets

| Dataset | Concepts | Type |
|---------|----------|------|
| `eng-french` | 1 (language) | Semi-synthetic |
| `eng-german` | 1 (language) | Semi-synthetic |
| `masc-fem-eng` | 1 (gender) | Semi-synthetic |
| `2-binary` | 2 (lang+gender) | Semi-synthetic |
| `corr-binary` | 2 (correlated) | Semi-synthetic |
| `truthful-qa` | 1 | HuggingFace |
| `sycophancy` | 1 | Bundled CC BY 4.0 JSON |
| `refusal` | 1 | User-supplied external JSON |
| `bias-in-bios` | 1 (gender) | HuggingFace |
| `categorical-contrastive` | 3 | Semi-synthetic |

New datasets require aligned source vocabularies and an entry in
`data/registry.py`; there is no public automatic-vocabulary generator. See
[data/SEMI_SYNTHETIC_DATA.md](data/SEMI_SYNTHETIC_DATA.md) and the
[data source/license inventory](data/README.md).

## Key Experiments

| Experiment | Script | What it tests |
|-----------|--------|---------------|
| Full eval (all metrics, all datasets) | `job_run_all_evals.sh` | Identifiability + similarity tables |
| Semi-synthetic SAE/SSAE sweep | `job_semi_synth.sh` | Reconstruction and recovery across raw- and difference-input models |
| AxBench concept detection | `job_eval_axbench.sh` | External benchmark (500 concepts) |

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SSAE_DATA_ROOT` | Embeddings + models root | Must be set by the user |
| `OODPROBE_ZIP_PATH` | OOD probe data | Derived from `SSAE_DATA_ROOT` |
| `HF_TOKEN` | HuggingFace (gated datasets) | — |

## Output Format

All evals produce `.jsonl`:

```json
{"dataset": "eng-french", "model_name": "ssae_..._seed0", "model_type": "SSAE", "seed": 0, "metric": "sae_mcc", "value": 0.912}
```

```python
from evals.common import read_results, summarise_results, EvalResult
results = [EvalResult(**d) for d in read_results("results.jsonl")]
summarise_results(results)
```

## Project History

This project began as the `psp` prototype on May 14, 2024. The public history
preserves its original repository-creation commit (`1527989`); the audited
public source snapshot follows in the `Public release` commit.

## License

Original code and documentation are available under the [MIT License](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for adapted code and the
licensing boundary around external datasets and model artifacts.
