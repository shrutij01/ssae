#!/usr/bin/env bash
#
# Capped, auditable launcher for the four-dataset hard-step/rectangle study.
#
# Every invocation has an isolated manifest/submission ledger. A study-wide
# ledger enforces the requested ceilings across adaptive waves:
#   * at most 32 submitted seed-0 calibration/fallback jobs;
#   * at most six submitted seed-0 L1 targets per dataset;
#   * at most 16 submitted confirmation jobs.
#
# Dry-run is the default. Submission requires --submit.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SSAE_REPO_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
data_root="${SSAE_DATA_ROOT:-/path/to/ssae-data}"
venv_activate="${SSAE_VENV:-/path/to/ssae-venv/bin/activate}"

mode="dry-run"
phase="rectangle"
run_label=""
datasets_text="refusal sycophancy truthful-qa bias-in-bios"
study_root="${script_dir}/generated_jobs/step_l0_cross_dataset"
log_root="${script_dir}/logs/step_l0_cross_dataset"
partition="${PARTITION:-long}"
gpu="${TRAIN_GPU:-gpu:l40s:1}"
time_limit="${TIME_LIMIT:-1:30:00}"
memory="${MEMORY:-16Gb}"
cpus="${CPUS_PER_TASK:-8}"
l1_specs=()

usage() {
    cat <<'EOF'
Usage: job_ssae_step_l0_cross_dataset.sh [--dry-run|--submit] [options]

Phases:
  rectangle     One seed-0 rectangle/bandwidth=1.0 run per dataset.
  l1            Explicit seed-0 targets supplied by repeated --l1-target.
  active-only   One justified seed-0 active-only fallback per dataset.
  confirmation  Seeds 1 and 2 for rectangle and each dataset's selected L1.

Options:
  --run-label LABEL          Isolated output/log directory name (required).
  --datasets "A B ..."       Dataset subset.
  --l1-target DATASET=VALUE  Repeat for each L1 calibration/confirmation target.
  --study-root PATH          Generated scripts, manifests, and global ledger.
  --log-root PATH            SLURM stdout/stderr root.
  --partition NAME           SLURM partition (default: long).
  --gpu RESOURCE             SLURM GRES (default: gpu:l40s:1).
  --time LIMIT               Per-job limit (default: 1:30:00).
  --mem MEMORY               Per-job memory (default: 16Gb).
  --cpus N                   CPUs per task (default: 8).

Fixed training settings:
  norm=ln, activation=none, input_norm=standardize, inequality constraint,
  threshold/ind_th=0.01, step-L0 target=0.005, rectangle bandwidth=1.0,
  warmup=100, schedule=500, epochs=2000, batch=64, lr=5e-4,
  relative loss, extrapolation/Lagrangian, ExtraAdam dual, dual_lr_div=2.0.

The launcher never passes --n-concepts. It reads num_concepts from each YAML
for the manifest and leaves the training parser to preserve that metadata.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) mode="dry-run"; shift ;;
        --submit) mode="submit"; shift ;;
        --phase) phase=$2; shift 2 ;;
        --run-label) run_label=$2; shift 2 ;;
        --datasets) datasets_text=$2; shift 2 ;;
        --l1-target) l1_specs+=("$2"); shift 2 ;;
        --study-root) study_root=$2; shift 2 ;;
        --log-root) log_root=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --time) time_limit=$2; shift 2 ;;
        --mem) memory=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${phase}" in
    rectangle|l1|active-only|confirmation) ;;
    *) echo "Invalid --phase: ${phase}" >&2; exit 2 ;;
esac
if [[ -z "${run_label}" || ! "${run_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run-label is required and must contain only A-Z, a-z, 0-9, ., _, -" >&2
    exit 2
fi
if [[ "${phase}" == "l1" || "${phase}" == "confirmation" ]]; then
    if [[ ${#l1_specs[@]} -eq 0 ]]; then
        echo "${phase} requires at least one --l1-target DATASET=VALUE" >&2
        exit 2
    fi
elif [[ ${#l1_specs[@]} -ne 0 ]]; then
    echo "--l1-target is valid only for l1 or confirmation phases" >&2
    exit 2
fi

read -r -a datasets <<< "${datasets_text}"
if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "--datasets must not be empty" >&2
    exit 2
fi
allowed_datasets=" refusal sycophancy truthful-qa bias-in-bios "
for dataset in "${datasets[@]}"; do
    if [[ "${allowed_datasets}" != *" ${dataset} "* ]]; then
        echo "Unsupported dataset: ${dataset}" >&2
        exit 2
    fi
done

declare -A selected_l1=()
for spec in "${l1_specs[@]}"; do
    if [[ "${spec}" != *=* ]]; then
        echo "Invalid --l1-target ${spec}; expected DATASET=VALUE" >&2
        exit 2
    fi
    dataset=${spec%%=*}
    target=${spec#*=}
    if [[ "${allowed_datasets}" != *" ${dataset} "* ]]; then
        echo "Unsupported L1 dataset: ${dataset}" >&2
        exit 2
    fi
    if ! [[ "${target}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
        echo "Invalid L1 target: ${target}" >&2
        exit 2
    fi
    if [[ "${phase}" == "confirmation" && -n "${selected_l1[${dataset}]:-}" ]]; then
        echo "confirmation accepts exactly one L1 target per dataset" >&2
        exit 2
    fi
    selected_l1["${dataset}"]+="${selected_l1[${dataset}]:+ }${target}"
done

if [[ "${phase}" == "confirmation" ]]; then
    for dataset in "${datasets[@]}"; do
        if [[ -z "${selected_l1[${dataset}]:-}" ]]; then
            echo "Missing confirmation L1 target for ${dataset}" >&2
            exit 2
        fi
    done
fi

output_dir="${study_root}/${run_label}"
log_dir="${log_root}/${run_label}"
mkdir -p "${output_dir}" "${log_dir}"
manifest="${output_dir}/manifest.tsv"
submission_ledger="${output_dir}/submitted_jobs.tsv"
accounting_ledger="${study_root}/submitted_jobs.tsv"
if [[ "${mode}" == "submit" && -f "${submission_ledger}" ]]; then
    submitted_rows="$(
        awk 'END {print (NR > 0 ? NR - 1 : 0)}' "${submission_ledger}"
    )"
    if (( submitted_rows > 0 )); then
        echo "Refusing duplicate submission; ledger has jobs: ${submission_ledger}" >&2
        exit 3
    fi
fi

threshold=0.01
bandwidth=1.0
step_target=0.005
warmup=100
schedule=500
epochs=2000
batch=64
lr=0.0005
loss=relative
dual_lr_div=2.0

job_names=()
job_scripts=()
job_cohorts=()
job_datasets=()
job_embeddings=()
job_data_cfgs=()
job_num_concepts=()
job_estimators=()
job_surrogates=()
job_targets=()
job_bandwidths=()
job_seeds=()

float_label() {
    local value=$1
    printf '%s' "${value//./p}" | tr '+-' 'pm'
}

yaml_num_concepts() {
    local cfg=$1
    awk -F: '$1 ~ /^[[:space:]]*num_concepts[[:space:]]*$/ {
        gsub(/[[:space:]]/, "", $2); print $2; exit
    }' "${cfg}"
}

add_job() {
    local cohort=$1
    local dataset=$2
    local estimator=$3
    local surrogate=$4
    local target=$5
    local run_seed=$6
    local embeddings="${data_root}/${dataset}/${dataset}_gemma-2-2b-it_25_last_token.h5"
    local data_cfg="${data_root}/${dataset}/${dataset}_gemma-2-2b-it_25_last_token.yaml"
    local candidate_args=""
    local candidate_label
    local job_name
    local job_script
    local num_concepts

    for required in "${embeddings}" "${data_cfg}" "${venv_activate}"; do
        if [[ ! -r "${required}" ]]; then
            echo "Required file is not readable: ${required}" >&2
            exit 2
        fi
    done
    num_concepts="$(yaml_num_concepts "${data_cfg}")"
    if ! [[ "${num_concepts}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Missing/invalid num_concepts in ${data_cfg}" >&2
        exit 2
    fi

    if [[ "${estimator}" == "step_l0" ]]; then
        candidate_label="${surrogate//_/-}"
        candidate_args="--step-l0-surrogate ${surrogate} --step-l0-threshold ${threshold}"
        if [[ "${surrogate}" == "rectangle" ]]; then
            candidate_args+=" --step-l0-bandwidth ${bandwidth}"
        fi
    else
        candidate_label="l1-t$(float_label "${target}")"
    fi
    job_name="sl0x_${dataset}_${candidate_label}_s${run_seed}"
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
export WANDB_DIR=${data_root}/wandb_step_l0_cross_dataset
export WANDB_RUN_GROUP=step_l0_cross_dataset/${run_label}/${dataset}/gemma25
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
    --seed ${run_seed} \\
    --warmup ${warmup} \\
    --schedule ${schedule} \\
    --epochs ${epochs} \\
    --batch ${batch} \\
    --lr ${lr} \\
    --loss ${loss} \\
    --input-norm standardize \\
    --constrained-optim extrapolation \\
    --formulation lagrangian \\
    --dual-optim extra-adam \\
    --dual-lr-div ${dual_lr_div}
EOF
    chmod +x "${job_script}"
    bash -n "${job_script}"

    job_names+=("${job_name}")
    job_scripts+=("${job_script}")
    job_cohorts+=("${cohort}")
    job_datasets+=("${dataset}")
    job_embeddings+=("${embeddings}")
    job_data_cfgs+=("${data_cfg}")
    job_num_concepts+=("${num_concepts}")
    job_estimators+=("${estimator}")
    job_surrogates+=("${surrogate}")
    job_targets+=("${target}")
    job_bandwidths+=("$([[ "${surrogate}" == "rectangle" ]] && printf '%s' "${bandwidth}" || printf 'NA')")
    job_seeds+=("${run_seed}")
}

case "${phase}" in
    rectangle)
        for dataset in "${datasets[@]}"; do
            add_job seed0 "${dataset}" step_l0 rectangle "${step_target}" 0
        done
        ;;
    l1)
        for dataset in "${datasets[@]}"; do
            for target in ${selected_l1[${dataset}]:-}; do
                add_job seed0 "${dataset}" l1 NA "${target}" 0
            done
        done
        ;;
    active-only)
        for dataset in "${datasets[@]}"; do
            add_job seed0 "${dataset}" step_l0 active_only "${step_target}" 0
        done
        ;;
    confirmation)
        for dataset in "${datasets[@]}"; do
            target=${selected_l1[${dataset}]}
            for run_seed in 1 2; do
                add_job confirmation "${dataset}" step_l0 rectangle "${step_target}" "${run_seed}"
                add_job confirmation "${dataset}" l1 NA "${target}" "${run_seed}"
            done
        done
        ;;
esac

job_count=${#job_names[@]}
if [[ "${job_count}" -eq 0 ]]; then
    echo "No jobs generated for phase=${phase} datasets=${datasets_text}" >&2
    exit 4
fi
if [[ "${phase}" == "confirmation" && "${job_count}" -gt 16 ]]; then
    echo "Refusing confirmation: ${job_count} jobs exceeds cap 16" >&2
    exit 4
fi

header='index	cohort	phase	run_label	dataset	embeddings	data_cfg	num_concepts	norm	activation	input_norm	estimator	surrogate	target	bandwidth	threshold	seed	warmup	schedule	epochs	batch	lr	loss	constraint_mode	constrained_optim	formulation	dual_optim	dual_lr_div	job_id	stdout	stderr	script'
printf '%s\n' "${header}" > "${manifest}"

if [[ "${mode}" == "submit" ]]; then
    printf 'index\tjob_name\tjob_id\n' > "${submission_ledger}"
    if [[ ! -s "${accounting_ledger}" ]]; then
        printf 'cohort\tphase\trun_label\tdataset\testimator\tsurrogate\ttarget\tseed\tjob_name\tjob_id\tmanifest\n' > "${accounting_ledger}"
    fi

    seed0_count="$(awk -F'\t' 'NR > 1 && $1 == "seed0" {n++} END {print n+0}' "${accounting_ledger}")"
    confirmation_count="$(awk -F'\t' 'NR > 1 && $1 == "confirmation" {n++} END {print n+0}' "${accounting_ledger}")"
    new_seed0=0
    new_confirmation=0
    for cohort in "${job_cohorts[@]}"; do
        if [[ "${cohort}" == "seed0" ]]; then
            ((new_seed0 += 1))
        else
            ((new_confirmation += 1))
        fi
    done
    if (( seed0_count + new_seed0 > 32 )); then
        echo "Refusing submission: seed-0 total would exceed 32" >&2
        exit 4
    fi
    if (( confirmation_count + new_confirmation > 16 )); then
        echo "Refusing submission: confirmation total would exceed 16" >&2
        exit 4
    fi

    for dataset in "${datasets[@]}"; do
        prior_l1="$(awk -F'\t' -v d="${dataset}" \
            'NR > 1 && $1 == "seed0" && $4 == d && $5 == "l1" {n++}
             END {print n+0}' "${accounting_ledger}")"
        new_l1=0
        for ((index = 0; index < job_count; index++)); do
            if [[ "${job_cohorts[index]}" == "seed0" \
                && "${job_datasets[index]}" == "${dataset}" \
                && "${job_estimators[index]}" == "l1" ]]; then
                ((new_l1 += 1))
            fi
        done
        if (( prior_l1 + new_l1 > 6 )); then
            echo "Refusing submission: ${dataset} would exceed six seed-0 L1 targets" >&2
            exit 4
        fi
    done

    for ((index = 0; index < job_count; index++)); do
        duplicate="$(awk -F'\t' \
            -v c="${job_cohorts[index]}" \
            -v d="${job_datasets[index]}" \
            -v e="${job_estimators[index]}" \
            -v u="${job_surrogates[index]}" \
            -v t="${job_targets[index]}" \
            -v s="${job_seeds[index]}" \
            'NR > 1 && $1 == c && $4 == d && $5 == e && $6 == u &&
             $7 == t && $8 == s {print 1; exit}' "${accounting_ledger}")"
        if [[ "${duplicate}" == "1" ]]; then
            echo "Refusing duplicate submitted configuration: ${job_names[index]}" >&2
            exit 4
        fi
    done
fi

for ((index = 0; index < job_count; index++)); do
    job_id="DRY_RUN"
    if [[ "${mode}" == "submit" ]]; then
        job_id="$(sbatch --parsable "${job_scripts[index]}")"
        printf '%s\t%s\t%s\n' \
            "${index}" "${job_names[index]}" "${job_id}" >> "${submission_ledger}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${job_cohorts[index]}" "${phase}" "${run_label}" \
            "${job_datasets[index]}" "${job_estimators[index]}" \
            "${job_surrogates[index]}" "${job_targets[index]}" \
            "${job_seeds[index]}" "${job_names[index]}" "${job_id}" \
            "${manifest}" >> "${accounting_ledger}"
    fi

    stdout="${log_dir}/${job_names[index]}_${job_id}.out"
    stderr="${log_dir}/${job_names[index]}_${job_id}.err"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tln\tnone\tstandardize\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tinequality\textrapolation\tlagrangian\textra-adam\t%s\t%s\t%s\t%s\t%s\n' \
        "${index}" "${job_cohorts[index]}" "${phase}" "${run_label}" \
        "${job_datasets[index]}" "${job_embeddings[index]}" \
        "${job_data_cfgs[index]}" "${job_num_concepts[index]}" \
        "${job_estimators[index]}" "${job_surrogates[index]}" \
        "${job_targets[index]}" "${job_bandwidths[index]}" "${threshold}" \
        "${job_seeds[index]}" "${warmup}" "${schedule}" "${epochs}" \
        "${batch}" "${lr}" "${loss}" "${dual_lr_div}" "${job_id}" \
        "${stdout}" "${stderr}" "${job_scripts[index]}" >> "${manifest}"

    printf '[%02d] %-46s dataset=%-12s estimator=%-7s target=%-8s seed=%s concepts=%s job=%s\n' \
        "${index}" "${job_names[index]}" "${job_datasets[index]}" \
        "${job_estimators[index]}" "${job_targets[index]}" \
        "${job_seeds[index]}" "${job_num_concepts[index]}" "${job_id}"
done

echo "${mode}: ${job_count} ${phase} jobs"
echo "manifest: ${manifest}"
if [[ "${mode}" == "submit" ]]; then
    echo "submission ledger: ${submission_ledger}"
    echo "study accounting: ${accounting_ledger}"
fi
