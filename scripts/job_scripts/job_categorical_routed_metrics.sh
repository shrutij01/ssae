#!/usr/bin/env bash
#
# Recompute categorical Pearson MCC, Spearman MCC, and DCI-D with the correct
# representation route. Dry-run is the default; --submit launches one CPU job
# per LLM and records the job IDs to prevent accidental duplicate submission.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SSAE_REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
data_root="${SSAE_DATA_ROOT:-/path/to/ssae-data}"
venv_activate="${SSAE_VENV:-/path/to/ssae-venv/bin/activate}"

mode="dry-run"
model_root="${data_root}/categorical-contrastive/run_out"
output_dir="${repo_root}/results/findings/categorical_routed/runs"
job_dir="${script_dir}/generated_jobs/categorical_routed"
log_dir="${script_dir}/logs/categorical_routed"
partition="${PARTITION:-long}"
time_limit="${TIME_LIMIT:-12:00:00}"
memory="${MEMORY:-32Gb}"
cpus="${CPUS_PER_TASK:-8}"

usage() {
    cat <<'EOF'
Usage: job_categorical_routed_metrics.sh [--dry-run|--submit] [options]

Options:
  --model-root PATH  Categorical run_out containing SSAE/SAE checkpoints
  --output-dir PATH  Destination for the two JSONL result files
  --partition NAME   SLURM partition (default: long)
  --time LIMIT       Per-LLM time limit (default: 12:00:00)
  --mem MEMORY       Per-LLM memory (default: 32Gb)
  --cpus N           CPUs per task (default: 8)

Each successful model emits exactly one routed triplet:
  raw SAE             mcc, mcc_s, dci_d
  SSAE / *_DIFFS SAE  mcc_diff, mcc_s_diff, dci_d_diff
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) mode="dry-run"; shift ;;
        --submit) mode="submit"; shift ;;
        --model-root) model_root=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --time) time_limit=$2; shift 2 ;;
        --mem) memory=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in "${venv_activate}" "${model_root}"; do
    if [[ ! -r "${required}" ]]; then
        echo "Required path is not readable: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${output_dir}" "${job_dir}" "${log_dir}"
ledger="${job_dir}/submitted_jobs.tsv"
if [[ "${mode}" == "submit" && -s "${ledger}" ]]; then
    echo "Refusing duplicate submission; ledger exists: ${ledger}" >&2
    exit 3
fi

llms=(gemma pythia)
embedding_names=(
    "categorical-contrastive_gemma-2-2b-it_25_last_token.h5"
    "categorical-contrastive_pythia70m_5_last_token.h5"
)
submitted_ids=()
report="${output_dir%/results}/final_analysis/comparison_report.md"
if [[ -e "${report}" ]]; then
    echo "Refusing to overwrite existing report: ${report}" >&2
    exit 3
fi

for index in "${!llms[@]}"; do
    llm="${llms[$index]}"
    embeddings="${data_root}/categorical-contrastive/${embedding_names[$index]}"
    output="${output_dir}/categorical-contrastive_${llm}_routed.jsonl"
    if [[ ! -r "${embeddings}" ]]; then
        echo "Embedding file is not readable: ${embeddings}" >&2
        exit 2
    fi
    if [[ -e "${output}" ]]; then
        echo "Refusing to overwrite existing output: ${output}" >&2
        exit 3
    fi
done

for index in "${!llms[@]}"; do
    llm="${llms[$index]}"
    embeddings="${data_root}/categorical-contrastive/${embedding_names[$index]}"
    output="${output_dir}/categorical-contrastive_${llm}_routed.jsonl"
    job_name="cat_route_${llm}"
    job_script="${job_dir}/${job_name}.sh"

    cat > "${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_dir}/${job_name}_%j.out
#SBATCH --error=${log_dir}/${job_name}_%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${memory}
#SBATCH --time=${time_limit}

set -euo pipefail
module load python/3.10 2>/dev/null || true
source ${venv_activate}
export PYTHONPATH=${repo_root}:\${PYTHONPATH:-}
cd ${repo_root}

python -u scripts/evaluate_categorical_routed.py \\
    --embeddings ${embeddings} \\
    --model-root ${model_root} \\
    --llm ${llm} \\
    --categorical-source ${repo_root}/data/categorical/train.json \\
    --categorical-data-seed 42 \\
    --output ${output}
EOF
    chmod +x "${job_script}"

    if [[ "${mode}" == "submit" ]]; then
        job_id="$(sbatch --parsable "${job_script}")"
        submitted_ids+=("${job_id}")
        printf '%s\t%s\t%s\n' "${llm}" "${job_id}" "${output}" >> "${ledger}"
        echo "${llm}: submitted ${job_id} -> ${output}"
    else
        echo "${llm}: ${job_script} -> ${output}"
    fi
done

analysis_script="${job_dir}/cat_route_analysis.sh"
cat > "${analysis_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=cat_route_analysis
#SBATCH --output=${log_dir}/cat_route_analysis_%j.out
#SBATCH --error=${log_dir}/cat_route_analysis_%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4Gb
#SBATCH --time=0:10:00

set -euo pipefail
module load python/3.10 2>/dev/null || true
source ${venv_activate}
export PYTHONPATH=${repo_root}:\${PYTHONPATH:-}
cd ${repo_root}

python -u scripts/analyze_categorical_routed.py \\
    --results-dir ${output_dir} \\
    --output ${report}
EOF
chmod +x "${analysis_script}"

if [[ "${mode}" == "submit" ]]; then
    dependency="$(IFS=:; echo "${submitted_ids[*]}")"
    analysis_id="$(sbatch --parsable --dependency="afterok:${dependency}" "${analysis_script}")"
    printf '%s\t%s\t%s\n' "analysis" "${analysis_id}" "${report}" >> "${ledger}"
    echo "analysis: submitted ${analysis_id} after ${dependency} -> ${report}"
else
    echo "analysis: ${analysis_script} -> ${report}"
fi
