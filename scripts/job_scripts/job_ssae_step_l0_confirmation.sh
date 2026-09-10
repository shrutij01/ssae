#!/usr/bin/env bash
#
# Three-seed, multi-concept confirmation:
#   L1 reference + best rectangular candidate + active-only fallback.
# This is a post-pilot confirmation, isolated from the capped 20-job pilot.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SSAE_REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
data_root="${SSAE_DATA_ROOT:-/path/to/ssae-data}"
venv_activate="${SSAE_VENV:-/path/to/ssae-venv/bin/activate}"

mode="dry-run"
embeddings="${data_root}/2-binary/2-binary_gemma-2-2b-it_25_last_token.h5"
data_cfg="${data_root}/2-binary/2-binary_gemma-2-2b-it_25_last_token.yaml"
output_dir="${script_dir}/generated_jobs/step_l0_confirmation"
log_dir="${script_dir}/logs/step_l0_confirmation"
partition="${PARTITION:-long}"
gpu="${TRAIN_GPU:-gpu:l40s:1}"
time_limit="${TIME_LIMIT:-0:30:00}"
memory="${MEMORY:-16Gb}"
cpus="${CPUS_PER_TASK:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) mode="dry-run"; shift ;;
        --submit) mode="submit"; shift ;;
        --emb) embeddings=$2; shift 2 ;;
        --data-cfg) data_cfg=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --log-dir) log_dir=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --time) time_limit=$2; shift 2 ;;
        --mem) memory=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run|--submit] [--emb PATH] [--data-cfg PATH]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

for required in "${embeddings}" "${data_cfg}"; do
    if [[ ! -r "${required}" ]]; then
        echo "Required file is not readable: ${required}" >&2
        exit 2
    fi
done
if [[ "${mode}" == "submit" && ! -r "${venv_activate}" ]]; then
    echo "Virtualenv activation script is not readable: ${venv_activate}" >&2
    exit 2
fi

mkdir -p "${output_dir}" "${log_dir}"
manifest="${output_dir}/manifest.tsv"
submission_ledger="${output_dir}/submitted_jobs.tsv"
if [[ "${mode}" == "submit" && -s "${submission_ledger}" ]]; then
    echo "Refusing duplicate submission; ledger exists: ${submission_ledger}" >&2
    exit 3
fi

seeds=(0 1 2)
candidate_names=(l1 rectangle active_only)
max_jobs=9
threshold=0.01
warmup=100
schedule=500
epochs=2000

job_names=()
job_scripts=()
job_estimators=()
job_surrogates=()
job_targets=()
job_bandwidths=()
job_seeds=()

write_job() {
    local candidate=$1
    local seed=$2
    local estimator target surrogate bandwidth candidate_args job_name

    case "${candidate}" in
        l1)
            estimator=l1
            target=0.001
            surrogate=NA
            bandwidth=NA
            candidate_args=""
            ;;
        rectangle)
            estimator=step_l0
            target=0.005
            surrogate=rectangle
            bandwidth=1.0
            candidate_args="--step-l0-surrogate rectangle --step-l0-threshold ${threshold} --step-l0-bandwidth ${bandwidth}"
            ;;
        active_only)
            estimator=step_l0
            target=0.005
            surrogate=active_only
            bandwidth=NA
            candidate_args="--step-l0-surrogate active_only --step-l0-threshold ${threshold}"
            ;;
        *) echo "Unknown candidate: ${candidate}" >&2; exit 2 ;;
    esac

    job_name="sl0c_${candidate}_s${seed}"
    job_script="${output_dir}/${job_name}.sh"
    cat > "${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_dir}/${job_name}_%j.out
#SBATCH --error=${log_dir}/${job_name}_%j.err
#SBATCH --partition=${partition}
#SBATCH --gres=${gpu}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${memory}
#SBATCH --time=${time_limit}

set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
source ${venv_activate}
export PYTHONPATH=${repo_root}:\${PYTHONPATH:-}
export WANDB_MODE=offline
export WANDB_DIR=${data_root}/wandb_step_l0_confirmation
export WANDB_RUN_GROUP=step_l0_confirmation/2-binary/gemma25
cd ${repo_root}

python -u -m ssae.training \\
    ${embeddings} ${data_cfg} \\
    --norm ln \\
    --activation none \\
    --sparsity-type ${estimator} \\
    --constraint-mode inequality \\
    --target ${target} \\
    --ind-th ${threshold} \\
    ${candidate_args} \\
    --seed ${seed} \\
    --warmup ${warmup} \\
    --schedule ${schedule} \\
    --epochs ${epochs} \\
    --batch 64 \\
    --lr 0.0005 \\
    --loss relative \\
    --input-norm standardize \\
    --constrained-optim extrapolation \\
    --formulation lagrangian \\
    --dual-optim extra-adam \\
    --dual-lr-div 2.0
EOF
    chmod +x "${job_script}"

    job_names+=("${job_name}")
    job_scripts+=("${job_script}")
    job_estimators+=("${estimator}")
    job_surrogates+=("${surrogate}")
    job_targets+=("${target}")
    job_bandwidths+=("${bandwidth}")
    job_seeds+=("${seed}")
}

for candidate in "${candidate_names[@]}"; do
    for seed in "${seeds[@]}"; do
        write_job "${candidate}" "${seed}"
    done
done

job_count=${#job_names[@]}
if [[ "${job_count}" -ne "${max_jobs}" ]]; then
    echo "Refusing confirmation: expected ${max_jobs} jobs, got ${job_count}" >&2
    exit 4
fi

printf 'index\tjob_name\tnorm\testimator\tsurrogate\ttarget\tbandwidth\tthreshold\tseed\tjob_id\tstdout\tstderr\tscript\n' > "${manifest}"
if [[ "${mode}" == "submit" ]]; then
    printf 'index\tjob_name\tjob_id\n' > "${submission_ledger}"
fi

for ((index = 0; index < job_count; index++)); do
    job_id="DRY_RUN"
    if [[ "${mode}" == "submit" ]]; then
        job_id="$(sbatch --parsable "${job_scripts[index]}")"
        printf '%s\t%s\t%s\n' \
            "${index}" "${job_names[index]}" "${job_id}" >> "${submission_ledger}"
    fi
    stdout="${log_dir}/${job_names[index]}_${job_id}.out"
    stderr="${log_dir}/${job_names[index]}_${job_id}.err"
    printf '%s\t%s\tln\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${index}" \
        "${job_names[index]}" \
        "${job_estimators[index]}" \
        "${job_surrogates[index]}" \
        "${job_targets[index]}" \
        "${job_bandwidths[index]}" \
        "${threshold}" \
        "${job_seeds[index]}" \
        "${job_id}" \
        "${stdout}" \
        "${stderr}" \
        "${job_scripts[index]}" >> "${manifest}"
    printf '[%02d] %-24s estimator=%-7s surrogate=%-11s seed=%s job=%s\n' \
        "${index}" \
        "${job_names[index]}" \
        "${job_estimators[index]}" \
        "${job_surrogates[index]}" \
        "${job_seeds[index]}" \
        "${job_id}"
done

echo "${mode}: ${job_count}/${max_jobs} confirmation jobs"
echo "manifest: ${manifest}"
