#!/usr/bin/env bash
#
# Seed-0 follow-up calibration, run in two phases:
#   1. L1 targets that bracket achieved p_sparsity near 0.034 and 0.050.
#   2. Rectangular step-L0 dual/schedule settings aimed at reducing p_sparsity.
#
# Dry-run is the default. Each phase has an isolated submission ledger and
# stays below the original 20-job pilot cap.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SSAE_REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
data_root="${SSAE_DATA_ROOT:-/path/to/ssae-data}"
venv_activate="${SSAE_VENV:-/path/to/ssae-venv/bin/activate}"

mode="dry-run"
phase="l1"
embeddings="${data_root}/2-binary/2-binary_gemma-2-2b-it_25_last_token.h5"
data_cfg="${data_root}/2-binary/2-binary_gemma-2-2b-it_25_last_token.yaml"
output_dir=""
log_dir=""
partition="${PARTITION:-long}"
gpu="${TRAIN_GPU:-gpu:l40s:1}"
time_limit="${TIME_LIMIT:-0:30:00}"
memory="${MEMORY:-16Gb}"
cpus="${CPUS_PER_TASK:-4}"

usage() {
    cat <<'EOF'
Usage: job_ssae_step_l0_followup.sh [--dry-run|--submit] [options]

Options:
  --phase PHASE       l1 (default), l1-refine, step-l0, or l1-confirm
  --emb PATH          Embedding HDF5 (default: 2-binary/Gemma layer 25)
  --data-cfg PATH     Matching YAML config
  --output-dir PATH   Generated scripts and manifest
  --log-dir PATH      SLURM stdout/stderr directory
  --partition NAME    SLURM partition (default: long)
  --gpu RESOURCE      SLURM GRES (default: gpu:l40s:1)
  --time LIMIT        Per-job time limit (default: 0:30:00)
  --mem MEMORY        Per-job memory (default: 16Gb)
  --cpus N            CPUs per task (default: 4)

Run and analyze the l1 phase first. It contains seven targets:
  {0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01}.

The l1-refine phase contains four targets inside the initial bracket:
  {0.007, 0.0075, 0.00825, 0.0085}.

The post-pilot l1-confirm phase adds seeds {1,2} at target 0.00825.

The step-l0 phase contains nine rectangle/bandwidth=1.0 settings:
  schedule={100,250,500} x dual-lr-div={0.25,0.5,1.0}.

Both phases fix seed=0, warmup=100, epochs=2000, threshold/ind_th=0.01,
norm=ln, activation=none, input-norm=standardize, and inequality mode.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) mode="dry-run"; shift ;;
        --submit) mode="submit"; shift ;;
        --phase) phase=$2; shift 2 ;;
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

if [[ "${phase}" != "l1" && "${phase}" != "l1-refine" && "${phase}" != "step-l0" && "${phase}" != "l1-confirm" ]]; then
    echo "Invalid --phase: ${phase} (expected l1, l1-refine, step-l0, or l1-confirm)" >&2
    exit 2
fi

output_dir="${output_dir:-${script_dir}/generated_jobs/step_l0_followup/${phase}}"
log_dir="${log_dir:-${script_dir}/logs/step_l0_followup/${phase}}"

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

l1_targets=(0.002 0.003 0.004 0.005 0.006 0.008 0.01)
l1_refine_targets=(0.007 0.0075 0.00825 0.0085)
step_schedules=(100 250 500)
step_dual_divs=(0.25 0.5 1.0)

seed=0
threshold=0.01
bandwidth=1.0
step_l0_target=0.005
warmup=100
epochs=2000
hard_max_jobs=20

job_names=()
job_scripts=()
job_estimators=()
job_surrogates=()
job_targets=()
job_bandwidths=()
job_schedules=()
job_dual_divs=()
job_seeds=()

float_label() {
    local value=$1
    printf '%s' "${value//./p}"
}

write_job() {
    local estimator=$1
    local target=$2
    local schedule=$3
    local dual_div=$4
    local suffix=$5
    local run_seed=${6:-${seed}}
    local surrogate="NA"
    local estimator_args=""
    local job_name="sl0f_${suffix}_s${run_seed}"
    local job_script="${output_dir}/${job_name}.sh"

    if [[ "${estimator}" == "step_l0" ]]; then
        surrogate="rectangle"
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
export WANDB_DIR=${data_root}/wandb_step_l0_followup
export WANDB_RUN_GROUP=step_l0_followup/${phase}/2-binary/gemma25
cd ${repo_root}

python -u -m ssae.training \\
    ${embeddings} ${data_cfg} \\
    --norm ln \\
    --activation none \\
    --sparsity-type ${estimator} \\
    --constraint-mode inequality \\
    --target ${target} \\
    --ind-th ${threshold} \\
    ${estimator_args} \\
    --seed ${run_seed} \\
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
    --dual-lr-div ${dual_div}
EOF
    chmod +x "${job_script}"

    job_names+=("${job_name}")
    job_scripts+=("${job_script}")
    job_estimators+=("${estimator}")
    job_surrogates+=("${surrogate}")
    job_targets+=("${target}")
    job_bandwidths+=("$([[ "${estimator}" == "step_l0" ]] && printf '%s' "${bandwidth}" || printf 'NA')")
    job_schedules+=("${schedule}")
    job_dual_divs+=("${dual_div}")
    job_seeds+=("${run_seed}")
}

if [[ "${phase}" == "l1" ]]; then
    for target in "${l1_targets[@]}"; do
        write_job l1 "${target}" 500 2.0 "l1_t$(float_label "${target}")"
    done
    expected_jobs=7
elif [[ "${phase}" == "l1-refine" ]]; then
    for target in "${l1_refine_targets[@]}"; do
        write_job l1 "${target}" 500 2.0 "l1r_t$(float_label "${target}")"
    done
    expected_jobs=4
elif [[ "${phase}" == "l1-confirm" ]]; then
    for confirm_seed in 1 2; do
        write_job l1 0.00825 500 2.0 "l1c_t0p00825" "${confirm_seed}"
    done
    expected_jobs=2
else
    for schedule in "${step_schedules[@]}"; do
        for dual_div in "${step_dual_divs[@]}"; do
            write_job \
                step_l0 "${step_l0_target}" "${schedule}" "${dual_div}" \
                "rect_q${schedule}_d$(float_label "${dual_div}")"
        done
    done
    expected_jobs=9
fi

job_count=${#job_names[@]}
if [[ "${job_count}" -ne "${expected_jobs}" || "${job_count}" -gt "${hard_max_jobs}" ]]; then
    echo "Refusing follow-up: expected ${expected_jobs} jobs, got ${job_count}" >&2
    exit 4
fi

printf 'index\tjob_name\tnorm\testimator\tsurrogate\ttarget\tbandwidth\tthreshold\tseed\twarmup\tschedule\tdual_lr_div\tjob_id\tstdout\tstderr\tscript\n' > "${manifest}"
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
    printf '%s\t%s\tln\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${index}" \
        "${job_names[index]}" \
        "${job_estimators[index]}" \
        "${job_surrogates[index]}" \
        "${job_targets[index]}" \
        "${job_bandwidths[index]}" \
        "${threshold}" \
        "${job_seeds[index]}" \
        "${warmup}" \
        "${job_schedules[index]}" \
        "${job_dual_divs[index]}" \
        "${job_id}" \
        "${stdout}" \
        "${stderr}" \
        "${job_scripts[index]}" >> "${manifest}"
    printf '[%02d] %-28s estimator=%-7s target=%-5s schedule=%-3s dual_div=%-4s job=%s\n' \
        "${index}" \
        "${job_names[index]}" \
        "${job_estimators[index]}" \
        "${job_targets[index]}" \
        "${job_schedules[index]}" \
        "${job_dual_divs[index]}" \
        "${job_id}"
done

echo "${mode}: ${job_count}/${expected_jobs} ${phase} follow-up jobs"
echo "manifest: ${manifest}"
if [[ "${mode}" == "submit" ]]; then
    echo "submission ledger: ${submission_ledger}"
fi
