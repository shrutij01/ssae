#!/bin/bash

# ==============================================================================
# Re-run of the concluded real-world SSAE study's Result 1 (headline
# SSAE-vs-baseline table) under the input-norm correctness fix (commit 8919685).
#
# The fix froze the (previously learnable) affine of the LayerNorm applied to
# the SSAE reconstruction TARGET. Pre-fix that affine could shrink the target
# toward a constant, trivialising reconstruction. It was applied to EVERY SSAE
# run regardless of the encoder --norm, so both norm=ln and norm=none variants
# are affected. Baselines (ssae/sae.py's SAE class) are independent and are NOT
# retrained here — their Result 1 numbers stay valid.
#
# This reproduces the EXACT committed Result 1 SSAE config (verified against
# an existing run's cfg.yaml: init=default, loss=relative, dual_lr_div=2.0,
# l1@0.005, oc=rep_dim, epochs=15000, batch=64, lr=5e-4, schedule=1000,
# extra-adam, extrapolation) and changes ONLY the code (which now freezes the
# input-norm affine). loss=relative is deliberate: the committed runs are all
# relative; job_ssae_real.sh's LOSS=absolute never matched Result 1.
#
# input_norm is now in _HASH_KEYS, so these runs land in NEW run dirs and do
# not overwrite the pre-fix committed runs (which lack the input_norm key).
#
# Usage (from scripts/job_scripts/):
#   bash job_inputnorm_rerun.sh --dry                       # preview all
#   bash job_inputnorm_rerun.sh --datasets refusal --norms ln   # wave 1
#   bash job_inputnorm_rerun.sh                             # all 4 x {ln,none} seed0
#   bash job_inputnorm_rerun.sh --seeds "0 1 2"             # n=3 reseed (gate escalation)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# --- Defaults (override via flags) -------------------------------------------

DATASETS=(refusal truthful-qa sycophancy bias-in-bios)
NORMS=(ln none)
SEEDS=(0)

MODE="slurm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry) MODE="dry"; echo "=== DRY RUN ==="; shift ;;
        --datasets) IFS=' ' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --norms) IFS=' ' read -r -a NORMS <<< "$2"; shift 2 ;;
        --seeds) IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --epochs) EPOCHS_OVERRIDE="$2"; shift 2 ;;
        --time) TIME_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# --- Fixed model / LLM (gemma-2-2b-it L25 only; pythia deprioritised) ---------

MODEL_ID="google/gemma-2-2b-it"
LAYER=25

# Committed Result 1 SSAE HPs (see header)
TARGET=0.005
LOSS=relative
# Committed epochs: 15000 for the small real datasets (refusal 412 / truthful 735 /
# bios 900 pairs), but 2000 for sycophancy (18165 pairs). Override per dataset via --epochs.
EPOCHS=${EPOCHS_OVERRIDE:-15000}
BATCH=64
LR=0.0005
SCHEDULE=1000
DLR=2.0
INPUT_NORM=layernorm

# --- SLURM: long + l40s (tiny jobs on precomputed embeddings) -----------------

# cpus-per-task=8 matches the DataLoader's num_workers=8 (ssae/training.py:159)
# for non-small datasets; the original job_ssae_real.sh used 8. Requesting only
# 4 caused co-located jobs to deadlock in the DataLoader (sycophancy timed out).
TRAIN_TIME="${TIME_OVERRIDE:-0:45:00}"
TRAIN_MEM="16Gb"
PARTITION="long"
GPU="gpu:l40s:1"
CPU="cpus-per-task=8"

mkdir -p "${SCRIPT_DIR}/generated_jobs" "${SCRIPT_DIR}/logs"

# Keep offline wandb logs off the repo tree
export WANDB_DIR="${SSAE_DATA_ROOT}/wandb_inputnorm"
mkdir -p "${WANDB_DIR}"

counter=0

echo "=== SSAE input-norm-fix re-run (Result 1) ==="
echo "datasets=[${DATASETS[*]}] norms=[${NORMS[*]}] seeds=[${SEEDS[*]}]  loss=${LOSS} input_norm=${INPUT_NORM}"

for dataset in "${DATASETS[@]}"; do
    h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_gemma-2-2b-it_${LAYER}_last_token.h5"
    yaml="${SSAE_DATA_ROOT}/${dataset}/${dataset}_gemma-2-2b-it_${LAYER}_last_token.yaml"
    group="real_inputnorm/${dataset}/gemma2${LAYER}"

    if [ ! -f "${h5}" ]; then
        echo "  Skip ${dataset}: no embeddings (${h5})"
        continue
    fi

    for norm in "${NORMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            name="ssaefix_${dataset}_gemma2_n${norm}_s${seed}"
            script="${SCRIPT_DIR}/generated_jobs/${name}.sh"

            cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GPU}
#SBATCH --${CPU}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --time=${TRAIN_TIME}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR}
export WANDB_RUN_GROUP="${group}"
cd \${SSAE_REPO_ROOT}

python -m ssae.training \\
    ${h5} ${yaml} \\
    --seed ${seed} \\
    --epochs ${EPOCHS} \\
    --batch ${BATCH} \\
    --lr ${LR} \\
    --target ${TARGET} \\
    --loss ${LOSS} \\
    --norm ${norm} \\
    --schedule ${SCHEDULE} \\
    --dual-optim extra-adam \\
    --dual-lr-div ${DLR} \\
    --input-norm ${INPUT_NORM}
EOF
            chmod +x "${script}"

            if [ "$MODE" == "dry" ]; then
                echo "  [${counter}] ${dataset} norm=${norm} seed=${seed}"
                echo "        python -m ssae.training ${dataset}...  --loss ${LOSS} --norm ${norm} --input-norm ${INPUT_NORM} --dual-lr-div ${DLR} --target ${TARGET}"
            else
                jid=$(sbatch --parsable "${script}")
                echo "  ${name} -> ${jid}"
            fi
            ((counter++))
        done
    done
done

echo ""
echo "Total: ${counter} jobs"
