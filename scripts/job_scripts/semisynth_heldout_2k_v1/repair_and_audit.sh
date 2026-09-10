#!/bin/bash
#SBATCH --job-name=ss-hold-repair
#SBATCH --output=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/study_v1/logs/repair_%j.out
#SBATCH --error=/path/to/ssae/results/shared/semisynth_heldout_2k/runs/study_v1/logs/repair_%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
source /path/to/ssae/scripts/job_scripts/semisynth_heldout_2k_v1/common.sh
setup_semisynth_environment

python scripts/repair_semisynth_label_encoding.py --data-root "${DATA_ROOT}"
python scripts/audit_semisynth_splits.py \
    --data-root "${DATA_ROOT}" \
    --output "${STUDY_ROOT}/data_audit.json"
