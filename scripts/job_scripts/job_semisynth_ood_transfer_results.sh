#!/bin/bash
#SBATCH --job-name=ssae-ood-xfer
#SBATCH --output=/path/to/ssae/results/exp04/runs/semisynth_ood_transfer_v3/logs/%j.out
#SBATCH --error=/path/to/ssae/results/exp04/runs/semisynth_ood_transfer_v3/logs/%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

REPO_ROOT=/path/to/ssae
PYTHON=/path/to/ssae-venv/bin/python
DATA_ROOT=/path/to/ssae-data
STUDY_ROOT=${DATA_ROOT}/studies/ssae_rms_global_matrix_v3
SOURCE_ROOT=${STUDY_ROOT}/source_snapshot
MODEL_MANIFEST=${STUDY_ROOT}/manifests/frozen_training_manifest.jsonl
RAW=${REPO_ROOT}/results/exp04/runs/semisynth_ood_transfer_v3/raw_results.json
FROZEN=${REPO_ROOT}/results/exp04/frozen/semisynth_rms_v3/bundle.json

module load python/3.10 2>/dev/null || true
export PYTHONPATH=${SOURCE_ROOT}:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=${REPO_ROOT}/results/exp04/runs/semisynth_ood_transfer_v3/matplotlib
cd ${SOURCE_ROOT}
mkdir -p "$(dirname "${RAW}")" "$(dirname "${FROZEN}")" "${MPLCONFIGDIR}"

${PYTHON} ${SOURCE_ROOT}/scripts/evaluate_semisynth_ood_transfer.py \
  --data-root ${DATA_ROOT} \
  --llms gemma pythia \
  --regimes binary_to_lang corr_to_lang corr_occupation_to_lang \
  --methods \
    ssae_linear_l1 ssae_linear_step_l0 ssae_linear_sigmoid_l0 \
    ssae_relu ssae_jumprelu ssae_topk \
    sae_relu sae_jumprelu sae_topk \
  --model-manifest ${MODEL_MANIFEST} \
  --seeds 0 1 2 \
  --output ${RAW}

cd ${REPO_ROOT}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
${PYTHON} scripts/freeze_semisynth_ood_transfer.py \
  --input ${RAW} \
  --output ${FROZEN}

${PYTHON} scripts/freeze_semisynth_ood_transfer.py --check --output ${FROZEN}
