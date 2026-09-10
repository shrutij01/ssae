#!/bin/bash
set -euo pipefail

REPO_ROOT="/path/to/ssae"
DATA_ROOT="/path/to/ssae-data/semisynth_heldout_2k_v1"
STUDY_ROOT="${REPO_ROOT}/results/shared/semisynth_heldout_2k/runs/study_v1"
VENV_ACTIVATE="/path/to/ssae-venv/bin/activate"
HF_CACHE="/path/to/huggingface-cache"

setup_semisynth_environment() {
    module load python/3.10 2>/dev/null || true
    module load cuda/12.6.0/cudnn 2>/dev/null || true
    source "${VENV_ACTIVATE}"
    export SSAE_DATA_ROOT="${DATA_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export HF_HOME="${HF_CACHE}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export WANDB_MODE=offline
    cd "${REPO_ROOT}"
}
