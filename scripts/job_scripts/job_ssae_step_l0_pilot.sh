#!/usr/bin/env bash
#
# Twenty-job SSAE sparsity pilot on one embedding dataset:
#   2 norms x (5 L1 targets + 5 rectangular step-L0 bandwidths) = 20 jobs.
#
# Dry-run is the default. Submission requires an explicit --submit and is
# refused if this pilot output directory already has a submission ledger.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SSAE_REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
data_root="${SSAE_DATA_ROOT:-/path/to/ssae-data}"
venv_activate="${SSAE_VENV:-/path/to/ssae-venv/bin/activate}"

mode="dry-run"
embeddings="${data_root}/refusal/refusal_gemma-2-2b-it_25_last_token.h5"
data_cfg="${data_root}/refusal/refusal_gemma-2-2b-it_25_last_token.yaml"
output_dir="${script_dir}/generated_jobs/step_l0_pilot"
log_dir="${script_dir}/logs/step_l0_pilot"
partition="${PARTITION:-long}"
gpu="${TRAIN_GPU:-gpu:l40s:1}"
time_limit="${TIME_LIMIT:-0:30:00}"
memory="${MEMORY:-16Gb}"
cpus="${CPUS_PER_TASK:-4}"

usage() {
    cat <<'EOF'
Usage: job_ssae_step_l0_pilot.sh [--dry-run|--submit] [options]

Options:
  --emb PATH          Embedding HDF5 (default: refusal/Gemma layer 25)
  --data-cfg PATH     Matching YAML config
  --output-dir PATH   Generated scripts and manifest
  --log-dir PATH      SLURM stdout/stderr directory
  --partition NAME    SLURM partition (default: long)
  --gpu RESOURCE      SLURM GRES (default: gpu:l40s:1)
  --time LIMIT        Per-job time limit (default: 0:30:00)
  --mem MEMORY        Per-job memory (default: 16Gb)
  --cpus N            CPUs per task (default: 4)

The fixed experimental settings are:
  norm={ln,none}; rectangular bandwidth={0.001,0.01,0.1,1.0,4.0};
  step-L0 target=0.005; L1 target={0.001,0.002,0.005,0.01,0.02};
  threshold/ind_th=0.01; seed=0; warmup=100; schedule=500; epochs=2000;
  activation=none; constraint-mode=inequality; input-norm=standardize.
EOF
}

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
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -r "${embeddings}" ]]; then
    echo "Embedding file is not readable: ${embeddings}" >&2
    exit 2
fi
if [[ ! -r "${data_cfg}" ]]; then
    echo "Data config is not readable: ${data_cfg}" >&2
    exit 2
fi
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

norms=(ln none)
l1_targets=(0.001 0.002 0.005 0.01 0.02)
bandwidths=(0.001 0.01 0.1 1.0 4.0)
step_l0_target=0.005
threshold=0.01
seed=0
warmup=100
schedule=500
epochs=2000
max_jobs=20

job_names=()
job_scripts=()
job_norms=()
job_estimators=()
job_targets=()
job_bandwidths=()

float_label() {
    local value=$1
    printf '%s' "${value//./p}"
}

write_job() {
    local norm=$1
    local estimator=$2
    local target=$3
    local bandwidth=$4
    local suffix=$5
    local job_name="sl0p_${norm}_${suffix}_s${seed}"
    local job_script="${output_dir}/${job_name}.sh"
    local estimator_args=""

    if [[ "${estimator}" == "step_l0" ]]; then
        estimator_args="--step-l0-surrogate rectangle --step-l0-threshold ${threshold} --step-l0-bandwidth ${bandwidth}"
    fi

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
export WANDB_DIR=${data_root}/wandb_step_l0_pilot
export WANDB_RUN_GROUP=step_l0_pilot/refusal/gemma25
cd ${repo_root}

python -m ssae.training \\
    ${embeddings} ${data_cfg} \\
    --norm ${norm} \\
    --activation none \\
    --sparsity-type ${estimator} \\
    --constraint-mode inequality \\
    --target ${target} \\
    --ind-th ${threshold} \\
    ${estimator_args} \\
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
    job_norms+=("${norm}")
    job_estimators+=("${estimator}")
    job_targets+=("${target}")
    job_bandwidths+=("${bandwidth}")
}

for norm in "${norms[@]}"; do
    for target in "${l1_targets[@]}"; do
        write_job \
            "${norm}" l1 "${target}" NA \
            "l1_t$(float_label "${target}")"
    done
    for bandwidth in "${bandwidths[@]}"; do
        write_job \
            "${norm}" step_l0 "${step_l0_target}" "${bandwidth}" \
            "rect_b$(float_label "${bandwidth}")"
    done
done

job_count=${#job_names[@]}
if [[ "${job_count}" -ne "${max_jobs}" ]]; then
    echo "Refusing pilot: expected exactly ${max_jobs} jobs, got ${job_count}" >&2
    exit 4
fi

printf 'index\tjob_name\tnorm\testimator\ttarget\tbandwidth\tthreshold\tseed\tjob_id\tstdout\tstderr\tscript\n' > "${manifest}"

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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${index}" \
        "${job_names[index]}" \
        "${job_norms[index]}" \
        "${job_estimators[index]}" \
        "${job_targets[index]}" \
        "${job_bandwidths[index]}" \
        "${threshold}" \
        "${seed}" \
        "${job_id}" \
        "${stdout}" \
        "${stderr}" \
        "${job_scripts[index]}" >> "${manifest}"

    printf '[%02d] %-31s norm=%-4s estimator=%-7s target=%-5s bandwidth=%-5s job=%s\n' \
        "${index}" \
        "${job_names[index]}" \
        "${job_norms[index]}" \
        "${job_estimators[index]}" \
        "${job_targets[index]}" \
        "${job_bandwidths[index]}" \
        "${job_id}"
done

echo "${mode}: ${job_count}/${max_jobs} pilot jobs"
echo "manifest: ${manifest}"
if [[ "${mode}" == "submit" ]]; then
    echo "submission ledger: ${submission_ledger}"
fi
