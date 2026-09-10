#!/bin/bash
# ==============================================================================
# Shared environment config for all SSAE job scripts.
#
# Setup:
#   cp env.sh env.local.sh
#   vim env.local.sh   # edit paths for your cluster
# ==============================================================================

# Resolve the directory this script lives in (works even inside SLURM)
_ENV_DIR="${SSAE_ENV_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"

# Source local overrides FIRST (sets all the paths)
if [ -f "${_ENV_DIR}/env.local.sh" ]; then
    source "${_ENV_DIR}/env.local.sh"
fi

# --- Defaults (used if env.local.sh doesn't set them) ------------------------

export SSAE_DATA_ROOT="${SSAE_DATA_ROOT:?Set SSAE_DATA_ROOT in env.local.sh}"
export SSAE_REPO_ROOT="${SSAE_REPO_ROOT:?Set SSAE_REPO_ROOT in env.local.sh}"
export SSAE_VENV="${SSAE_VENV:?Set SSAE_VENV in env.local.sh}"

export HF_HOME="${HF_HOME:-${SSAE_DATA_ROOT}/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH="${SSAE_REPO_ROOT}:${PYTHONPATH}"

# LLM sweep config: "model_id|layer|short_name"
# Override in env.local.sh to change which LLMs are used
if [ -z "${SSAE_LLMS+x}" ]; then
    SSAE_LLMS=(
        "google/gemma-2-2b-it|25|gemma2"
        "EleutherAI/pythia-70m-deduped|5|pythia70m"
    )
fi

# Module loader (override in env.local.sh)
if ! type ssae_load_modules &>/dev/null; then
    ssae_load_modules() {
        module load python/3.10 2>/dev/null || true
        module load cuda/12.6.0/cudnn 2>/dev/null || true
    }
fi
