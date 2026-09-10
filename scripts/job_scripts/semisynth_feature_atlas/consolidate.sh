#!/bin/bash
#SBATCH --job-name=ss-atlas-merge
#SBATCH --output=/path/to/ssae/results/exp05/runs/feature_atlas/logs/consolidate_%j.out
#SBATCH --error=/path/to/ssae/results/exp05/runs/feature_atlas/logs/consolidate_%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail

REPO_ROOT=/path/to/ssae
PYTHON=/path/to/ssae-venv/bin/python
MANIFEST=${REPO_ROOT}/results/shared/semisynth_heldout_2k/manifests/frozen_manifest.jsonl
CACHE=${REPO_ROOT}/results/exp05/runs/feature_atlas/cache_v2
OUT=${REPO_ROOT}/results/exp05/runs/feature_atlas/cards

module load python/3.10 2>/dev/null || true
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd ${REPO_ROOT}

${PYTHON} scripts/export_feature_atlas.py consolidate \
  --manifest ${MANIFEST} \
  --cache-dir ${CACHE} \
  --out ${OUT} \
  --force
