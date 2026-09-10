#!/bin/bash
#SBATCH --job-name=ss-sl0-prepare
#SBATCH --output=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/step_l0/logs/prepare_%j.out
#SBATCH --error=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/step_l0/logs/prepare_%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
source /path/to/ssae/scripts/job_scripts/semisynth_heldout_2k_v1/common.sh
setup_semisynth_environment

STEP_L0_STUDY_ROOT="${REPO_ROOT}/results/shared/semisynth_heldout_2k/runs/step_l0"

python scripts/audit_semisynth_splits.py \
    --data-root "${DATA_ROOT}" \
    --output "${STEP_L0_STUDY_ROOT}/data_audit.json"

python scripts/evaluate_semisynth_metrics.py prepare \
    --study-root "${STEP_L0_STUDY_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --python /path/to/ssae-venv/bin/python \
    --epochs 2000 \
    --trainer ssae \
    --sparsity-type step_l0 \
    --require-heldout \
    --partition long \
    --gres gpu:l40s:1 \
    --cpus 8 \
    --memory 32G \
    --time 01:00:00 \
    --max-parallel 30 \
    --batch-size 512

eval_job=$(sbatch --parsable "${STEP_L0_STUDY_ROOT}/metrics/evaluate_array.sh")
consolidate_job=$(sbatch --parsable \
    --dependency="afterok:${eval_job}" \
    "${STEP_L0_STUDY_ROOT}/metrics/consolidate.sh")
echo "METRIC_ARRAY_JOB=${eval_job}"
echo "CONSOLIDATE_JOB=${consolidate_job}"
