#!/bin/bash
#SBATCH --job-name=synth-id-v4
#SBATCH --output=/path/to/ssae-data/studies/synthetic_identification_repro_v4/logs/train_%A_%a.out
#SBATCH --error=/path/to/ssae-data/studies/synthetic_identification_repro_v4/logs/train_%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-29%8

set -euo pipefail

SOURCE_ROOT=/path/to/ssae-data/studies/synthetic_identification_repro_v4/source_snapshot
PYTHON=/path/to/ssae-venv/bin/python
OUTPUT=/path/to/ssae-data/studies/synthetic_identification_repro_v4

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH=${SOURCE_ROOT}:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd ${SOURCE_ROOT}

${PYTHON} -u -m experiments.exp06.run run \
  --cell "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --output-root ${OUTPUT}
