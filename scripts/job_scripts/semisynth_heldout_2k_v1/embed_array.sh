#!/bin/bash
#SBATCH --job-name=ss-hold-embed
#SBATCH --output=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/study_v1/logs/embed_%A_%a.out
#SBATCH --error=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/study_v1/logs/embed_%A_%a.err
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-9%10

set -euo pipefail
source /path/to/ssae/scripts/job_scripts/semisynth_heldout_2k_v1/common.sh
setup_semisynth_environment

DATASETS=(eng-french eng-german masc-fem-eng 2-binary corr-binary)
LLM_NAMES=(gemma pythia)
MODEL_IDS=(google/gemma-2-2b-it EleutherAI/pythia-70m-deduped)
MODEL_STEMS=(gemma-2-2b-it pythia70m)
LAYERS=(25 5)

dataset_index=$((SLURM_ARRAY_TASK_ID / 2))
llm_index=$((SLURM_ARRAY_TASK_ID % 2))
dataset="${DATASETS[dataset_index]}"
llm="${LLM_NAMES[llm_index]}"
model_id="${MODEL_IDS[llm_index]}"
model_stem="${MODEL_STEMS[llm_index]}"
layer="${LAYERS[llm_index]}"
h5="${DATA_ROOT}/${dataset}/${dataset}_${model_stem}_${layer}_last_token.h5"

if [ -e "${h5}" ]; then
    echo "Refusing to overwrite frozen study embedding: ${h5}" >&2
    exit 1
fi

python -m ssae.store_embeddings \
    --dataset "${dataset}" \
    --model_id "${model_id}" \
    --layer "${layer}" \
    --num-samples 1000 \
    --split 0.8 \
    --batch-size 128 \
    --pooling-method last_token

python scripts/audit_semisynth_splits.py \
    --data-root "${DATA_ROOT}" \
    --datasets "${dataset}" \
    --llms "${llm}"
