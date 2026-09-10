#!/bin/bash
#SBATCH --job-name=ss-hold-sl0
#SBATCH --output=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/step_l0/logs/train_%A_%a.out
#SBATCH --error=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/step_l0/logs/train_%A_%a.err
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-29%30

set -euo pipefail
source /path/to/ssae/scripts/job_scripts/semisynth_heldout_2k_v1/common.sh
setup_semisynth_environment

DATASETS=(eng-french eng-german masc-fem-eng 2-binary corr-binary)
LLM_NAMES=(gemma pythia)
MODEL_STEMS=(gemma-2-2b-it pythia70m)
LAYERS=(25 5)
WIDTHS=(2304 512)

seed=$((SLURM_ARRAY_TASK_ID % 3))
llm_index=$(((SLURM_ARRAY_TASK_ID / 3) % 2))
dataset_index=$((SLURM_ARRAY_TASK_ID / 6))

dataset="${DATASETS[dataset_index]}"
llm="${LLM_NAMES[llm_index]}"
model_stem="${MODEL_STEMS[llm_index]}"
layer="${LAYERS[llm_index]}"
width="${WIDTHS[llm_index]}"
h5="${DATA_ROOT}/${dataset}/${dataset}_${model_stem}_${layer}_last_token.h5"
yaml="${h5%.h5}.yaml"

python scripts/audit_semisynth_splits.py \
    --data-root "${DATA_ROOT}" \
    --datasets "${dataset}" \
    --llms "${llm}"

export WANDB_RUN_GROUP="semisynth_heldout_2k_step_l0/${dataset}/${llm}"

# Matched against the existing Linear + L1 SSAE runs. The estimator is the
# only scientific change: rectangular Step-L0, threshold 0.01, bandwidth 1.0.
python -m ssae.training \
    "${h5}" "${yaml}" \
    --seed "${seed}" \
    --epochs 2000 \
    --batch 64 \
    --lr 0.0005 \
    --target 0.005 \
    --loss absolute \
    --norm ln \
    --activation none \
    --oc "${width}" \
    --sparsity-type step_l0 \
    --step-l0-surrogate rectangle \
    --step-l0-threshold 0.01 \
    --step-l0-bandwidth 1.0 \
    --warmup 500 \
    --schedule 500 \
    --dual-optim extra-adam \
    --dual-lr-div 2.0
