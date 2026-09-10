#!/usr/bin/env bash
# Submit this job with an afterok dependency on the calibration/full launcher.
# It reads the resulting 18-job full-run ledger, then submits metric workers
# that wait for every full checkpoint and a final 2x2 consolidation job.
#SBATCH --job-name=csoft-eval-followup
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/ssae-data/studies/canonical_15k_softshrink_factorial_v1/logs/orchestration/eval_%j.out
#SBATCH --error=/path/to/ssae-data/studies/canonical_15k_softshrink_factorial_v1/logs/orchestration/eval_%j.err

set -euo pipefail

repo_root=/path/to/ssae
study_root=/path/to/ssae-data/studies/canonical_15k_softshrink_factorial_v1
eval_root=${study_root}/evaluation_source_enforcement
venv_activate=/path/to/ssae-venv/bin/activate

source "${venv_activate}"
export PYTHONPATH="${eval_root}:${study_root}/source_snapshot"
cd "${repo_root}"

"${venv_activate%/activate}/python" \
  "${eval_root}/scripts/evaluate_canonical_softshrink_factorial.py" \
  submit-auto --study-root "${study_root}" --repo-root "${repo_root}" \
  --venv-activate "${venv_activate}"
