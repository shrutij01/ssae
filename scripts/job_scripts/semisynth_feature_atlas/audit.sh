#!/bin/bash
#SBATCH --job-name=ss-atlas-audit
#SBATCH --output=/path/to/ssae/results/exp05/runs/feature_atlas/logs/audit_%j.out
#SBATCH --error=/path/to/ssae/results/exp05/runs/feature_atlas/logs/audit_%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

REPO_ROOT=/path/to/ssae
PYTHON=/path/to/ssae-venv/bin/python
ATLAS_ROOT=${REPO_ROOT}/results/exp05/runs/feature_atlas/cards

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
cd ${REPO_ROOT}

${PYTHON} scripts/audit_feature_atlas.py \
  --atlas-root ${ATLAS_ROOT} \
  --out ${REPO_ROOT}/results/exp05/runs/feature_atlas/audit.json \
  --require-generation \
  --prune-unreferenced
