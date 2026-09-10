#!/bin/bash

# ==============================================================================
# Penalized SAE/SSAE activation baselines on real datasets.
# Run AFTER job_ssae_real.sh completes.
#
# Usage:
#   bash job_sae_real.sh --type relu --dry       # preview relu jobs
#   bash job_sae_real.sh --type relu             # submit relu
#   bash job_sae_real.sh --type topk             # submit topk (after relu done)
#   bash job_sae_real.sh --type jumprelu         # submit jumprelu
#   bash job_sae_real.sh --type relu --diffs     # SSAE ReLU (penalized)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATASETS=(truthful-qa sycophancy refusal bias-in-bios)

LLMS=(
    "google/gemma-2-2b-it|25|gemma2"
    "EleutherAI/pythia-70m-deduped|5|pythia70m"
)

GAMMAS=(1e-5 1e-4 1e-3)
SEEDS=(0 1 2)

TRAIN_TIME="0:30:00"
TRAIN_MEM="32Gb"
TRAIN_GPU="gpu:1"
CPU="cpus-per-task=8"

# --- Parse args ---------------------------------------------------------------

MODE="slurm"
SAE_TYPE=""
USE_DIFFS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry) MODE="dry"; echo "=== DRY RUN ==="; shift ;;
        --type) SAE_TYPE="$2"; shift 2 ;;
        --diffs) USE_DIFFS=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$SAE_TYPE" ]; then
    echo "Usage: bash job_sae_real.sh --type {relu|topk|jumprelu} [--diffs] [--dry]"
    exit 1
fi

DIFFS_FLAG=""
DIFFS_LABEL=""
if [ "$USE_DIFFS" = true ]; then
    DIFFS_FLAG="--train-on-diffs"
    DIFFS_LABEL="_diffs"
fi

mkdir -p generated_jobs logs

counter=0

echo "=== SAE ${SAE_TYPE}${DIFFS_LABEL} on real datasets ==="

for dataset in "${DATASETS[@]}"; do
    for llm_cfg in "${LLMS[@]}"; do
        IFS='|' read -r model_id layer short <<< "${llm_cfg}"

        h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.h5"
        yaml="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.yaml"
        group="real/${dataset}/${short}${layer}"

        if [ ! -f "${h5}" ]; then
            echo "  Skip ${dataset}/${short}: no embeddings"
            continue
        fi

        for gamma in "${GAMMAS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                name="sae_${SAE_TYPE}${DIFFS_LABEL}_${dataset}_${short}_g${gamma}_s${seed}"
                script="generated_jobs/${name}.sh"

                cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=logs/${name}_%j.out
#SBATCH --error=logs/${name}_%j.err
#SBATCH --time=${TRAIN_TIME}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --gres=${TRAIN_GPU}
#SBATCH --${CPU}

source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export PYTHONPATH="\${SSAE_REPO_ROOT}:\$PYTHONPATH"
export SSAE_DATA_ROOT="\${SSAE_DATA_ROOT}"
export HF_HOME="\${HF_HOME}"
export WANDB_RUN_GROUP="${group}"
cd \${SSAE_REPO_ROOT}

python -m ssae.sae \\
    ${h5} ${yaml} \\
    --sae-type ${SAE_TYPE} \\
    --seed ${seed} \\
    --epochs 15000 \\
    --batch 64 \\
    --lr 0.0005 \\
    --loss absolute \\
    --gamma-reg ${gamma} \\
    --renorm-epochs 50 \\
    ${DIFFS_FLAG}
EOF
                chmod +x "${script}"

                if [ "$MODE" == "dry" ]; then
                    echo "  [${counter}] ${dataset}/${short} g=${gamma} s=${seed}"
                else
                    jid=$(sbatch --parsable "${script}")
                    echo "  ${name} → ${jid}"
                fi
                ((counter++))
            done
        done
    done
done

echo ""
echo "Total: ${counter} jobs (${SAE_TYPE}${DIFFS_LABEL})"
