#!/bin/bash
set -euo pipefail

# Clean, dependency-chained Corr(occupation) -> Lang(household) OOD run.
#
# The source dataset contains two correlated transformations with a shared
# English occupation anchor: English->French and English->German.  It is
# lexically disjoint from the existing English->French household target.
#
# Usage:
#   scripts/job_scripts/job_semisynth_ood_corr_clean.sh --dry-run
#   scripts/job_scripts/job_semisynth_ood_corr_clean.sh --submit
#   scripts/job_scripts/job_semisynth_ood_corr_clean.sh --status

REPO_ROOT="/path/to/ssae"
DATA_ROOT="/path/to/ssae-data"
VENV_ACTIVATE="/path/to/ssae-venv/bin/activate"
HF_CACHE="/path/to/huggingface-cache"
RUN_ROOT="${REPO_ROOT}/results/exp04/runs/semisynth_ood_corr_clean"
GENERATED_DIR="${RUN_ROOT}/generated_jobs"
LOG_DIR="${RUN_ROOT}/logs"
JOB_FILE="${RUN_ROOT}/job_ids.txt"
DATASET="corr-binary-occupation"
RESULTS="${RUN_ROOT}/results.json"

PARTITION="${PARTITION:-long}"
GPU="${GPU:-gpu:l40s:1}"
MODE=""

case "${1:-}" in
    --dry-run) MODE="dry-run" ;;
    --submit) MODE="submit" ;;
    --status) MODE="status" ;;
    *)
        echo "Usage: $0 --dry-run|--submit|--status" >&2
        exit 2
        ;;
esac

mkdir -p "${GENERATED_DIR}" "${LOG_DIR}"

if [ "${MODE}" = "status" ]; then
    if [ ! -s "${JOB_FILE}" ]; then
        echo "No clean Corr->Lang jobs recorded."
        exit 0
    fi
    while IFS='|' read -r stage name jid; do
        state=$(sacct -j "${jid}" --noheader --format=State%24 2>/dev/null | head -1 | xargs)
        printf '%-8s %-46s %-10s %s\n' "${stage}" "${name}" "${jid}" "${state:-PENDING}"
    done < "${JOB_FILE}"
    exit 0
fi

if [ "${MODE}" = "submit" ] && [ -s "${JOB_FILE}" ]; then
    echo "Refusing duplicate submission: ${JOB_FILE} already exists." >&2
    echo "Use --status to inspect the recorded run." >&2
    exit 1
fi

if [ "${MODE}" = "submit" ]; then
    : > "${JOB_FILE}"
fi

write_header() {
    local path="$1" name="$2" time="$3" mem="$4" partition="$5"
    cat > "${path}" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${LOG_DIR}/${name}_%j.out
#SBATCH --error=${LOG_DIR}/${name}_%j.err
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --mem=${mem}
#SBATCH --cpus-per-task=8
EOF
}

write_gpu_header() {
    local path="$1" name="$2" time="$3" mem="$4"
    write_header "${path}" "${name}" "${time}" "${mem}" "${PARTITION}"
    echo "#SBATCH --gres=${GPU}" >> "${path}"
}

write_environment() {
    local path="$1"
    cat >> "${path}" <<EOF
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
source ${VENV_ACTIVATE}
export SSAE_DATA_ROOT=${DATA_ROOT}
export PYTHONPATH=${REPO_ROOT}
export HF_HOME=${HF_CACHE}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
cd ${REPO_ROOT}
EOF
}

submit_job() {
    local stage="$1" name="$2" path="$3" dependency="${4:-}"
    chmod +x "${path}"
    if [ "${MODE}" = "dry-run" ]; then
        printf 'DRY %-8s %s%s\n' "${stage}" "${name}" "${dependency:+ after ${dependency}}" >&2
        echo "dry-${name}"
        return
    fi
    local jid
    if [ -n "${dependency}" ]; then
        jid=$(sbatch --parsable --dependency="afterok:${dependency}" "${path}")
    else
        jid=$(sbatch --parsable "${path}")
    fi
    echo "${stage}|${name}|${jid}" >> "${JOB_FILE}"
    printf '%-8s %-46s -> %s\n' "${stage}" "${name}" "${jid}" >&2
    echo "${jid}"
}

declare -A EMBED_IDS
TRAIN_IDS=()

LLM_CONFIGS=(
    "gemma|google/gemma-2-2b-it|gemma-2-2b-it|25|2304"
    "pythia|EleutherAI/pythia-70m-deduped|pythia70m|5|512"
)

for config in "${LLM_CONFIGS[@]}"; do
    IFS='|' read -r llm model_id short layer dimension <<< "${config}"
    name="oodcorr_emb_${llm}"
    path="${GENERATED_DIR}/${name}.sh"
    write_gpu_header "${path}" "${name}" "1:00:00" "48G"
    write_environment "${path}"
    cat >> "${path}" <<EOF
python -m ssae.store_embeddings \
    --dataset ${DATASET} \
    --model_id ${model_id} \
    --layer ${layer} \
    --num-samples 100 \
    --split 1.0 \
    --batch-size 128 \
    --pooling-method last_token
python scripts/validate_corr_occupation_dataset.py \
    --data-root ${DATA_ROOT} \
    --llms ${llm} \
    --require-embeddings
EOF
    EMBED_IDS["${llm}"]=$(submit_job "embed" "${name}" "${path}")

    h5="${DATA_ROOT}/${DATASET}/${DATASET}_${short}_${layer}_last_token.h5"
    yaml="${DATA_ROOT}/${DATASET}/${DATASET}_${short}_${layer}_last_token.yaml"

    for seed in 0 1 2; do
        name="oodcorr_ssae_${llm}_s${seed}"
        path="${GENERATED_DIR}/${name}.sh"
        write_gpu_header "${path}" "${name}" "3:00:00" "32G"
        write_environment "${path}"
        cat >> "${path}" <<EOF
export WANDB_RUN_GROUP="semisynth_ood_corr_clean/${llm}"
python -m ssae.training \
    ${h5} ${yaml} \
    --seed ${seed} \
    --epochs 15000 \
    --batch 64 \
    --lr 0.0005 \
    --target 0.005 \
    --loss absolute \
    --norm ln \
    --activation none \
    --oc ${dimension} \
    --sparsity-type l1 \
    --schedule 1000 \
    --dual-optim extra-adam \
    --dual-lr-div 2.0
EOF
        jid=$(submit_job "ssae" "${name}" "${path}" "${EMBED_IDS[${llm}]}")
        TRAIN_IDS+=("${jid}")
    done

    for method in relu jumprelu topk; do
        for seed in 0 1 2; do
            name="oodcorr_${method}_${llm}_s${seed}"
            path="${GENERATED_DIR}/${name}.sh"
            write_gpu_header "${path}" "${name}" "3:00:00" "32G"
            write_environment "${path}"
            cat >> "${path}" <<EOF
export WANDB_RUN_GROUP="semisynth_ood_corr_clean/${llm}"
python -m ssae.sae \
    ${h5} ${yaml} \
    --sae-type ${method} \
    --seed ${seed} \
    --epochs 15000 \
    --batch 64 \
    --lr 0.0005 \
    --loss absolute \
    --oc ${dimension} \
    --gamma-reg 0.001 \
    --renorm-epochs 50
EOF
            jid=$(submit_job "sae" "${name}" "${path}" "${EMBED_IDS[${llm}]}")
            TRAIN_IDS+=("${jid}")
        done
    done
done

train_dependency=$(IFS=:; echo "${TRAIN_IDS[*]}")
name="oodcorr_eval"
path="${GENERATED_DIR}/${name}.sh"
write_header "${path}" "${name}" "0:30:00" "16G" "long-cpu"
write_environment "${path}"
cat >> "${path}" <<EOF
python scripts/validate_corr_occupation_dataset.py \
    --data-root ${DATA_ROOT} \
    --llms gemma pythia \
    --require-embeddings \
    --output ${RUN_ROOT}/data_audit.json
python scripts/evaluate_semisynth_ood_transfer.py \
    --data-root ${DATA_ROOT} \
    --llms gemma pythia \
    --regimes corr_occupation_to_lang \
    --methods ssae relu jumprelu topk \
    --seeds 0 1 2 \
    --random-draws 100 \
    --output ${RESULTS}
EOF
EVAL_ID=$(submit_job "eval" "${name}" "${path}" "${train_dependency}")

echo "Evaluation job: ${EVAL_ID}"
echo "Results: ${RESULTS}"
