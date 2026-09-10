#!/bin/bash

# ==============================================================================
# SSAE training on real datasets (truthful-qa, sycophancy, refusal, bias-in-bios)
#
# Step 1: Extract embeddings (if not already done)
# Step 2: Train SSAE with HP sweep (target × dual_lr_div × seed)
#
# Usage:
#   cd scripts/job_scripts
#   bash job_ssae_real.sh --dry      # preview
#   bash job_ssae_real.sh            # submit
#   bash job_ssae_real.sh --embed-only
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# --- Datasets -----------------------------------------------------------------

DATASETS=(truthful-qa sycophancy refusal bias-in-bios)

# LLMs: "model_id|layer|short_name"
LLMS=(
    "google/gemma-2-2b-it|25|gemma2"
    "EleutherAI/pythia-70m-deduped|5|pythia70m"
)

# --- SSAE HP sweep ------------------------------------------------------------

TARGETS=(0.005 0.01 0.05 0.1)
DUAL_LR_DIVS=(2.0 5.0)
SEEDS=(0 1 2)

# Fixed training HPs
EPOCHS=15000
BATCH=64
LR=0.0005
LOSS=absolute
NORM=ln
SCHEDULE=1000

# --- SLURM --------------------------------------------------------------------

EMBED_TIME="2:00:00"
EMBED_MEM="48Gb"
EMBED_GPU="gpu:a100:1"

TRAIN_TIME="0:30:00"
TRAIN_MEM="32Gb"
TRAIN_GPU="gpu:1"

CPU="cpus-per-task=8"

# --- Parse mode ---------------------------------------------------------------

MODE="slurm"
EMBED_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry) MODE="dry"; echo "=== DRY RUN ==="; shift ;;
        --embed-only) EMBED_ONLY=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p generated_jobs logs

counter=0
embed_ids=()

# --- Step 1: Embeddings ------------------------------------------------------

echo "=== Embeddings ==="

for dataset in "${DATASETS[@]}"; do
    for llm_cfg in "${LLMS[@]}"; do
        IFS='|' read -r model_id layer short <<< "${llm_cfg}"

        h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.h5"

        if [ -f "${h5}" ]; then
            echo "  Skip ${dataset}/${short}: exists"
            continue
        fi

        script="generated_jobs/embed_${dataset}_${short}.sh"
        cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=emb_${dataset}_${short}
#SBATCH --output=logs/emb_${dataset}_${short}_%j.out
#SBATCH --error=logs/emb_${dataset}_${short}_%j.err
#SBATCH --time=${EMBED_TIME}
#SBATCH --mem=${EMBED_MEM}
#SBATCH --gres=${EMBED_GPU}
#SBATCH --${CPU}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

python -m ssae.store_embeddings \\
    --dataset ${dataset} \\
    --model_id ${model_id} \\
    --layer ${layer} \\
    --num-samples 1000 \\
    --pooling-method last_token
EOF
        chmod +x "${script}"

        if [ "$MODE" == "dry" ]; then
            echo "  [${counter}] EMBED ${dataset}/${short}"
        else
            jid=$(sbatch --parsable "${script}")
            embed_ids+=("${jid}")
            echo "  Submitted ${dataset}/${short} → ${jid}"
        fi
        ((counter++))
    done
done

if [ "$EMBED_ONLY" = true ]; then
    echo "Submitted ${counter} embedding jobs"
    exit 0
fi

# --- Step 2: SSAE training ----------------------------------------------------

echo ""
echo "=== SSAE Training ==="

DEP=""
if [ ${#embed_ids[@]} -gt 0 ]; then
    dep_str=$(IFS=:; echo "${embed_ids[*]}")
    DEP="--dependency=afterok:${dep_str}"
fi

for dataset in "${DATASETS[@]}"; do
    for llm_cfg in "${LLMS[@]}"; do
        IFS='|' read -r model_id layer short <<< "${llm_cfg}"

        h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.h5"
        yaml="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.yaml"
        group="real/${dataset}/${short}${layer}"

        for target in "${TARGETS[@]}"; do
            for dlr in "${DUAL_LR_DIVS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    name="ssae_${dataset}_${short}_t${target}_d${dlr}_s${seed}"
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

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export WANDB_RUN_GROUP="${group}"
cd \${SSAE_REPO_ROOT}

python -m ssae.training \\
    ${h5} ${yaml} \\
    --seed ${seed} \\
    --epochs ${EPOCHS} \\
    --batch ${BATCH} \\
    --lr ${LR} \\
    --target ${target} \\
    --loss ${LOSS} \\
    --norm ${NORM} \\
    --schedule ${SCHEDULE} \\
    --dual-optim extra-adam \\
    --dual-lr-div ${dlr}
EOF
                    chmod +x "${script}"

                    if [ "$MODE" == "dry" ]; then
                        echo "  [${counter}] ${dataset}/${short} t=${target} d=${dlr} s=${seed}"
                    else
                        jid=$(sbatch --parsable ${DEP} "${script}")
                        echo "  ${name} → ${jid}"
                    fi
                    ((counter++))
                done
            done
        done
    done
done

echo ""
echo "Total: ${counter} jobs"
echo "  4 datasets × 2 LLMs × 4 targets × 2 dual_lr × 3 seeds = $(( 4 * 2 * 4 * 2 * 3 )) training"
echo ""
echo "W&B: https://wandb.ai/causalrepl/ssae"
