#!/bin/bash

# ==============================================================================
# SLURM job script: run all evaluations across all datasets
#
# Submits one job per dataset (parallel evaluation), then a final
# table-generation job that depends on all of them.
#
# Usage:
#   cd ssae/scripts/job_scripts
#   bash job_run_all_evals.sh            # submit all
#   bash job_run_all_evals.sh --dry      # preview
#   bash job_run_all_evals.sh --local    # run locally (no SLURM)
# ==============================================================================

# --- Config -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
ssae_data_root="${SSAE_DATA_ROOT}"
OUTPUT_DIR="${SSAE_REPO_ROOT}/results/findings/legacy_eval/runs"

# Datasets to evaluate (categorical-contrastive excluded: needs embeddings + models)
datasets=(
    "eng-french"
    "eng-german"
    "masc-fem-eng"
    "2-binary"
    "corr-binary"
    "truthful-qa"
    "sycophancy"
    "refusal"
    "bias-in-bios"
)

# SLURM settings (no GPU needed for identifiability/similarity)
time_limit="1:00:00"
memory="32Gb"
cpu_req="cpus-per-task=4"

# --- Parse mode ---------------------------------------------------------------

MODE="slurm"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)   MODE="dry";   echo "=== DRY RUN MODE ==="; shift ;;
        --local) MODE="local"; echo "=== LOCAL MODE ===";   shift ;;
        --force) FORCE_FLAG="--force"; echo "=== FORCE REGEN ==="; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --- Setup --------------------------------------------------------------------

mkdir -p generated_jobs
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/results"
mkdir -p "${OUTPUT_DIR}/tables"

counter=0
job_ids=()

# --- Per-dataset eval jobs ----------------------------------------------------

for dataset in "${datasets[@]}"; do
    script_name="generated_jobs/job_eval_${dataset}.sh"

    cat > "${script_name}" << EOF
#!/bin/bash
#SBATCH --job-name=eval_${dataset}
#SBATCH --output=logs/eval_${dataset}_%j.out
#SBATCH --error=logs/eval_${dataset}_%j.err
#SBATCH --time=${time_limit}
#SBATCH --mem=${memory}
#SBATCH --${cpu_req}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

echo "=== Evaluating ${dataset} ==="

python scripts/run_all_evals.py \\
    --datasets ${dataset} \\
    --output-dir ${OUTPUT_DIR} \\
    --data-root ${ssae_data_root} \\
    ${FORCE_FLAG}

echo "Done: ${dataset}"
EOF

    chmod +x "${script_name}"

    if [ "$MODE" == "dry" ]; then
        echo "[${counter}] ${dataset}"
    elif [ "$MODE" == "local" ]; then
        echo "--- Running ${dataset} ---"
        bash "${script_name}"
    else
        job_id=$(sbatch --parsable "${script_name}")
        job_ids+=("${job_id}")
        echo "Submitted ${dataset} as job ${job_id}"
    fi
    ((counter++))
done

# --- Table generation job (depends on all eval jobs) --------------------------

table_script="generated_jobs/job_generate_tables.sh"

cat > "${table_script}" << EOF
#!/bin/bash
#SBATCH --job-name=eval_tables
#SBATCH --output=logs/eval_tables_%j.out
#SBATCH --error=logs/eval_tables_%j.err
#SBATCH --time=0:10:00
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=1

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

echo "=== Generating LaTeX tables ==="

python scripts/run_all_evals.py \\
    --tables-only \\
    --output-dir ${OUTPUT_DIR} \\
    --data-root ${ssae_data_root}

echo ""
echo "Tables in ${OUTPUT_DIR}/tables/"
ls -la ${OUTPUT_DIR}/tables/
EOF

chmod +x "${table_script}"

if [ "$MODE" == "dry" ]; then
    echo ""
    echo "[TABLE] depends on all above"
elif [ "$MODE" == "local" ]; then
    echo ""
    echo "--- Generating tables ---"
    bash "${table_script}"
elif [ "$MODE" == "slurm" ]; then
    # Build dependency string
    if [ ${#job_ids[@]} -gt 0 ]; then
        dep_str=$(IFS=:; echo "${job_ids[*]}")
        table_job_id=$(sbatch --parsable --dependency=afterok:${dep_str} "${table_script}")
        echo ""
        echo "Submitted table generation as job ${table_job_id} (depends on ${#job_ids[@]} eval jobs)"
    else
        echo "No eval jobs submitted, skipping table generation"
    fi
fi

echo ""
echo "Submitted ${counter} eval jobs"
echo "Results → ${OUTPUT_DIR}/results/"
echo "Tables  → ${OUTPUT_DIR}/tables/"
