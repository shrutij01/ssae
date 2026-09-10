#!/bin/bash

# ==============================================================================
# Fast SSAE training on C4: extract-once, train-many
#
# Step 1 (GPU-heavy, ~1-2h): Extract activation differences from Gemma on C4
# Step 2 (GPU-light, ~30min each): Train SSAE with different hyperparameters
#
# Usage:
#   cd scripts/job_scripts
#   bash job_train_pile_fast.sh            # submit all
#   bash job_train_pile_fast.sh --dry      # preview
#   bash job_train_pile_fast.sh --train-only  # skip extraction
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

OUTPUT_DIR="${SSAE_DATA_ROOT}/pile"

# --- Experiment config --------------------------------------------------------

total_tokens=200000000

# Layer 20: AxBench/SAEBench standard
# Layer 25: where existing concept-specific SSAEs were trained
# Cross-layer skip: middle layer → target
strategies=(
    # --- Layer 20 ---
    "adjacent|20||adj_l20"
    "sentence|20||sent_l20"
    "cross_layer|19|20|xl_l19-20"
    "cross_layer|13|20|xl_l13-20"
    # --- Layer 25 ---
    "adjacent|25||adj_l25"
    "sentence|25||sent_l25"
    "cross_layer|13|25|xl_l13-25"
)

# SSAE training sweep
widths=(16384)
targets=(0.01 0.02 0.05 0.1)
seeds=(0 1 2)

# SLURM settings
extract_time="3:00:00"
extract_mem="64Gb"
extract_gpu="gpu:a100:1"

train_time="1:00:00"
train_mem="32Gb"
train_gpu="gpu:1"

cpu_req="cpus-per-task=8"

# --- Parse mode ---------------------------------------------------------------

MODE="slurm"
TRAIN_ONLY=false
if [[ "$1" == "--dry" ]]; then
    MODE="dry"
    echo "=== DRY RUN ==="
elif [[ "$1" == "--train-only" ]]; then
    TRAIN_ONLY=true
    echo "=== TRAIN ONLY ==="
fi

mkdir -p generated_jobs logs "${OUTPUT_DIR}"

counter=0
extract_job_ids=()

# --- Step 1: Extraction jobs --------------------------------------------------

if [ "$TRAIN_ONLY" = false ]; then
    echo "=== Step 1: Extract activation differences ==="

    for strat_cfg in "${strategies[@]}"; do
        IFS='|' read -r strategy layer layer_b label <<< "${strat_cfg}"

        h5_file="${OUTPUT_DIR}/diffs_${label}.h5"

        # Skip if already extracted
        if [ -f "${h5_file}" ]; then
            echo "  Skip extract ${label}: ${h5_file} exists"
            continue
        fi

        script_name="generated_jobs/job_extract_${label}.sh"

        layer_b_flag=""
        if [ -n "${layer_b}" ]; then
            layer_b_flag="--layer-b ${layer_b}"
        fi

        cat > "${script_name}" << EOF
#!/bin/bash
#SBATCH --job-name=extract_${label}
#SBATCH --output=logs/extract_${label}_%j.out
#SBATCH --error=logs/extract_${label}_%j.err
#SBATCH --time=${extract_time}
#SBATCH --mem=${extract_mem}
#SBATCH --gres=${extract_gpu}
#SBATCH --${cpu_req}

source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export PYTHONPATH="\${SSAE_REPO_ROOT}:\$PYTHONPATH"
export HF_HOME="\${HF_HOME}"
export HF_TOKEN="\${HF_TOKEN}"
cd \${SSAE_REPO_ROOT}

echo "=== Extracting ${strategy} diffs ==="
echo "  Output: ${h5_file}"

python -m ssae.extract_online \\
    --llm google/gemma-2-2b-it \\
    --dataset \${SSAE_HF_DATA:-allenai/c4} \\
    --dataset-subset en \\
    --diff-strategy ${strategy} \\
    --layer ${layer} ${layer_b_flag} \\
    --total-tokens ${total_tokens} \\
    --text-batch-size 32 \\
    --output ${h5_file}

echo "Done: \$?"
ls -lh ${h5_file} 2>/dev/null
EOF

        chmod +x "${script_name}"

        if [ "$MODE" == "dry" ]; then
            echo "[${counter}] EXTRACT ${label} → ${h5_file}"
        else
            job_id=$(sbatch --parsable "${script_name}")
            extract_job_ids+=("${job_id}")
            echo "Submitted extract ${label} → job ${job_id}"
        fi
        ((counter++))
    done

    echo ""
fi

# --- Step 2: SSAE training jobs -----------------------------------------------

echo "=== Step 2: SSAE training sweep ==="

for strat_cfg in "${strategies[@]}"; do
    IFS='|' read -r strategy layer layer_b label <<< "${strat_cfg}"

    h5_file="${OUTPUT_DIR}/diffs_${label}.h5"
    yaml_file="${OUTPUT_DIR}/diffs_${label}.yaml"

    for width in "${widths[@]}"; do
        for target in "${targets[@]}"; do
            for seed in "${seeds[@]}"; do
                script_name="generated_jobs/job_train_${label}_t${target}_s${seed}.sh"

                cat > "${script_name}" << EOF
#!/bin/bash
#SBATCH --job-name=ssae_${label}_t${target}_s${seed}
#SBATCH --output=logs/ssae_${label}_t${target}_s${seed}_%j.out
#SBATCH --error=logs/ssae_${label}_t${target}_s${seed}_%j.err
#SBATCH --time=${train_time}
#SBATCH --mem=${train_mem}
#SBATCH --gres=${train_gpu}
#SBATCH --${cpu_req}

source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export PYTHONPATH="\${SSAE_REPO_ROOT}:\$PYTHONPATH"
export SSAE_DATA_ROOT="\${SSAE_DATA_ROOT}"
export HF_HOME="\${HF_HOME}"
cd \${SSAE_REPO_ROOT}

echo "=== SSAE: ${label} target=${target} seed=${seed} ==="

python -m ssae.training \\
    ${h5_file} ${yaml_file} \\
    --seed ${seed} \\
    --epochs 15000 \\
    --batch 64 \\
    --lr 0.0005 \\
    --target ${target} \\
    --loss absolute \\
    --norm ln \\
    --schedule 1000 \\
    --dual-optim extra-adam \\
    --dual-lr-div 2.0

echo "Done: \$?"
EOF

                chmod +x "${script_name}"

                if [ "$MODE" == "dry" ]; then
                    echo "[${counter}] TRAIN ${label} target=${target} seed=${seed}"
                else
                    dep_flag=""
                    if [ ${#extract_job_ids[@]} -gt 0 ]; then
                        dep_str=$(IFS=:; echo "${extract_job_ids[*]}")
                        dep_flag="--dependency=afterok:${dep_str}"
                    fi
                    job_id=$(sbatch --parsable ${dep_flag} "${script_name}")
                    echo "Submitted train ${label} t=${target} s=${seed} → job ${job_id}"
                fi
                ((counter++))
            done
        done
    done
done

echo ""
echo "Submitted ${counter} jobs total"
echo "Extraction: ${OUTPUT_DIR}/diffs_*.h5"
echo "Models:     \${SSAE_DATA_ROOT}/pile/run_out/"
