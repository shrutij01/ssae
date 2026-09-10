#!/bin/bash
#SBATCH --job-name=ss-hold-train
#SBATCH --output=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/study_v1/logs/train_%A_%a.out
#SBATCH --error=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/study_v1/logs/train_%A_%a.err
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-209%40

set -euo pipefail
source /path/to/ssae/scripts/job_scripts/semisynth_heldout_2k_v1/common.sh
setup_semisynth_environment

DATASETS=(eng-french eng-german masc-fem-eng 2-binary corr-binary)
LLM_NAMES=(gemma pythia)
MODEL_STEMS=(gemma-2-2b-it pythia70m)
LAYERS=(25 5)
WIDTHS=(2304 512)
FAMILIES=(ssae relu_raw relu_diff topk_raw topk_diff jumprelu_raw jumprelu_diff)

family_index=$((SLURM_ARRAY_TASK_ID % 7))
seed=$(((SLURM_ARRAY_TASK_ID / 7) % 3))
llm_index=$(((SLURM_ARRAY_TASK_ID / 21) % 2))
dataset_index=$((SLURM_ARRAY_TASK_ID / 42))

dataset="${DATASETS[dataset_index]}"
llm="${LLM_NAMES[llm_index]}"
model_stem="${MODEL_STEMS[llm_index]}"
layer="${LAYERS[llm_index]}"
width="${WIDTHS[llm_index]}"
family="${FAMILIES[family_index]}"
h5="${DATA_ROOT}/${dataset}/${dataset}_${model_stem}_${layer}_last_token.h5"
yaml="${h5%.h5}.yaml"

python scripts/audit_semisynth_splits.py \
    --data-root "${DATA_ROOT}" \
    --datasets "${dataset}" \
    --llms "${llm}"

export WANDB_RUN_GROUP="semisynth_heldout_2k_v1/${dataset}/${llm}"

if [ "${family}" = "ssae" ]; then
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
        --sparsity-type l1 \
        --warmup 500 \
        --schedule 500 \
        --dual-optim extra-adam \
        --dual-lr-div 2.0
else
    sae_type="${family%%_*}"
    regime="${family#*_}"
    diff_flag=()
    if [ "${regime}" = "diff" ]; then
        diff_flag=(--train-on-diffs)
    fi
    python -m ssae.sae \
        "${h5}" "${yaml}" \
        --sae-type "${sae_type}" \
        --seed "${seed}" \
        --epochs 2000 \
        --batch 64 \
        --lr 0.0005 \
        --loss absolute \
        --oc "${width}" \
        --gamma-reg 0.001 \
        --renorm-epochs 50 \
        "${diff_flag[@]}"
fi
