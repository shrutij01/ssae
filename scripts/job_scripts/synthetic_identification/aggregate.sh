#!/bin/bash
#SBATCH --job-name=synth-aggregate-v4
#SBATCH --output=/path/to/ssae-data/studies/synthetic_identification_repro_v4/logs/aggregate_%j.out
#SBATCH --error=/path/to/ssae-data/studies/synthetic_identification_repro_v4/logs/aggregate_%j.err
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

SOURCE_ROOT=/path/to/ssae-data/studies/synthetic_identification_repro_v4/source_snapshot
PYTHON=/path/to/ssae-venv/bin/python
OUTPUT=/path/to/ssae-data/studies/synthetic_identification_repro_v4

module load python/3.10 2>/dev/null || true
export PYTHONPATH=${SOURCE_ROOT}:${PYTHONPATH:-}
cd ${SOURCE_ROOT}

${PYTHON} -u -m experiments.exp06.run aggregate \
  --output-root ${OUTPUT}
