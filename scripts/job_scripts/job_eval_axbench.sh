#!/bin/bash

# ==============================================================================
# AxBench-style evaluation of SSAE models
#
# Evaluates concept detection (AUC-ROC, hard-negative accuracy) on
# AxBench concepts using zero-shot feature mapping.
#
# Assumes SSAE models were trained and saved under:
#   {SSAE_DATA_ROOT}/{dataset}/run_out/ssae_*_seed{N}/
#
# Usage:
#   cd ssae/scripts/job_scripts
#   bash job_eval_axbench.sh          # submit all jobs
#   bash job_eval_axbench.sh --dry    # print commands without submitting
# ==============================================================================

# --- Config -------------------------------------------------------------------

# Models to evaluate (paths to SSAE run directories)
# These should contain weights.pth and cfg.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
ssae_data_root="${SSAE_DATA_ROOT}"
run_out_dirs=(
    "${ssae_data_root}/2-binary/run_out"
    "${ssae_data_root}/corr-binary/run_out"
    "${ssae_data_root}/eng-french/run_out"
    "${ssae_data_root}/bias-in-bios/run_out"
    "${ssae_data_root}/refusal/run_out"
    "${ssae_data_root}/sycophancy/run_out"
)

# HuggingFace token — set this or export HF_TOKEN before running
# You can also run: huggingface-cli login  (once, interactively)
hf_token="${HF_TOKEN:-}"

# AxBench datasets (HuggingFace IDs)
axbench_datasets=(
    "pyvene/axbench-concept10"        # quick test (10 concepts)
    # "pyvene/axbench-concept500"     # full benchmark (500 concepts)
)

# LLM + layer combinations
# Layer 25 = what your SSAEs were trained on (Gemma)
# Layer 20 = AxBench leaderboard default
# Running both lets you compare: does the layer matter for detection?
llm_configs=(
    "google/gemma-2-2b-it|25"
    "google/gemma-2-2b-it|20"
)

# Zero-shot mapping method
mapping_method="auc"    # auc | pearson | mean_diff

# Seeds to evaluate
seeds=(0 1 2)

# SLURM settings
job_name="axbench_eval"
time_limit="1:00:00"
memory="48Gb"
gpu_req="gpu:a100:1"
cpu_req="cpus-per-task=8"

# Dry run mode
DRY_RUN=false
if [[ "$1" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# --- Directories --------------------------------------------------------------

OUTPUT_DIR="${SSAE_REPO_ROOT}/results/findings/axbench/runs"

mkdir -p generated_jobs
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# --- Job generation -----------------------------------------------------------

counter=0

submit_job() {
    local model_dir="$1"
    local axbench_dataset="$2"
    local llm_id="$3"
    local layer="$4"
    local output_file="$5"
    local script_name="generated_jobs/job_axbench_${counter}.sh"

    # Short names for the job
    local model_name=$(basename "${model_dir}")
    local dataset_short=$(echo "${axbench_dataset}" | sed 's|.*/||')

    cat > "${script_name}" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_${counter}
#SBATCH --output=logs/axbench_%j.out
#SBATCH --error=logs/axbench_%j.err
#SBATCH --time=${time_limit}
#SBATCH --mem=${memory}
#SBATCH --gres=${gpu_req}
#SBATCH --${cpu_req}

module load python/3.10
module load cuda/12.6.0/cudnn
source ${SSAE_VENV}
export PYTHONPATH="${SSAE_REPO_ROOT}:\$PYTHONPATH"
export SSAE_DATA_ROOT="${ssae_data_root}"
export HF_HOME="${HF_HOME}"
export HF_TOKEN="${hf_token}"
cd ${SSAE_REPO_ROOT}

echo "=== AxBench Eval ==="
echo "Model: ${model_dir}"
echo "Dataset: ${axbench_dataset}"
echo "LLM: ${llm_id} layer ${layer}"
echo "Output: ${output_file}"
echo "===================="

python -m evals.external.axbench \\
    --model "${model_dir}" \\
    --hf-dataset "${axbench_dataset}" \\
    --llm "${llm_id}" \\
    --layer ${layer} \\
    --mapping-method ${mapping_method} \\
    --batch-size 16 \\
    --hf-token "${hf_token}" \\
    --output "${output_file}"

echo "Done. Exit code: \$?"
EOF

    chmod +x "${script_name}"

    if [ "$DRY_RUN" = true ]; then
        echo "[${counter}] ${model_name} × ${dataset_short} × layer${layer} → ${output_file}"
    else
        sbatch "${script_name}"
    fi
    ((counter++))
}

# --- Submit jobs --------------------------------------------------------------

for run_out_dir in "${run_out_dirs[@]}"; do
    if [ ! -d "${run_out_dir}" ]; then
        echo "Warning: ${run_out_dir} does not exist, skipping"
        continue
    fi

    for axbench_dataset in "${axbench_datasets[@]}"; do
        dataset_short=$(echo "${axbench_dataset}" | sed 's|.*/||')

        for llm_cfg in "${llm_configs[@]}"; do
            IFS='|' read -r llm_id layer <<< "${llm_cfg}"

            # Discover SSAE models across seeds
            for seed in "${seeds[@]}"; do
                # Find matching model directory
                model_dirs=$(find "${run_out_dir}" -maxdepth 1 -type d -name "ssae_*_seed${seed}" 2>/dev/null)

                for model_dir in ${model_dirs}; do
                    # Verify it has weights
                    if [ ! -f "${model_dir}/weights.pth" ]; then
                        echo "Warning: ${model_dir} has no weights.pth, skipping"
                        continue
                    fi

                    model_name=$(basename "${model_dir}")
                    output_file="${OUTPUT_DIR}/${model_name}_${dataset_short}_layer${layer}.jsonl"

                    # Skip if output already exists
                    if [ -f "${output_file}" ]; then
                        echo "Skipping ${model_name} × ${dataset_short} (output exists)"
                        continue
                    fi

                    submit_job "${model_dir}" "${axbench_dataset}" "${llm_id}" "${layer}" "${output_file}"
                done
            done
        done
    done
done

echo ""
echo "Submitted ${counter} jobs"

# ==============================================================================
# After all jobs complete, aggregate results:
#
#   python -c "
#   from evals.common import read_results, summarise_results, EvalResult
#   import glob
#   results = []
#   for f in glob.glob('results/findings/axbench/runs/*.jsonl'):
#       for d in read_results(f):
#           results.append(EvalResult(**d))
#   summarise_results(results)
#   "
# ==============================================================================
