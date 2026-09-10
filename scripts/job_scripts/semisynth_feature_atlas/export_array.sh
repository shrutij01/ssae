#!/bin/bash
#SBATCH --job-name=ss-feature-atlas
#SBATCH --output=/path/to/ssae/results/exp05/runs/feature_atlas/logs/export_%A_%a.out
#SBATCH --error=/path/to/ssae/results/exp05/runs/feature_atlas/logs/export_%A_%a.err
#SBATCH --partition=long-cpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-209%40

set -euo pipefail

REPO_ROOT=/path/to/ssae
PYTHON=/path/to/ssae-venv/bin/python
MANIFEST=${REPO_ROOT}/results/shared/semisynth_heldout_2k/manifests/frozen_manifest.jsonl
CACHE=${REPO_ROOT}/results/exp05/runs/feature_atlas/cache_v2

module load python/3.10 2>/dev/null || true
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd ${REPO_ROOT}

${PYTHON} scripts/export_feature_atlas.py worker \
  --manifest ${MANIFEST} \
  --cache-dir ${CACHE} \
  --index ${SLURM_ARRAY_TASK_ID} \
  --device cpu \
  --batch-size 512 \
  --permutations 1000 \
  --force
