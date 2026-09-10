#!/bin/bash
#SBATCH --job-name=ss-atlas-steer
#SBATCH --output=/path/to/ssae/results/exp05/runs/feature_atlas/logs/generate_%A_%a.out
#SBATCH --error=/path/to/ssae/results/exp05/runs/feature_atlas/logs/generate_%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-69%10

set -euo pipefail

REPO_ROOT=/path/to/ssae
PYTHON=/path/to/ssae-venv/bin/python
ATLAS_ROOT=${REPO_ROOT}/results/exp05/runs/feature_atlas/cards

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export HF_HOME=/path/to/huggingface-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd ${REPO_ROOT}

${PYTHON} scripts/export_feature_atlas_generation.py \
  --atlas-root ${ATLAS_ROOT} \
  --group-index ${SLURM_ARRAY_TASK_ID} \
  --device cuda \
  --prompt-count 2 \
  --max-new-tokens 24 \
  --diagnostic-tokens 8
