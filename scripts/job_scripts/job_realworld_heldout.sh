#!/bin/bash
# ==============================================================================
# Submit the held-out single-concept detection/monosemanticity eval as a CPU
# SLURM job (too heavy for the login node: sklearn probes over ~14 models x
# several dataset/LLM combos).
#
# Usage:
#   bash job_realworld_heldout.sh                       # all 4 datasets
#   bash job_realworld_heldout.sh refusal bias-in-bios  # subset
#
# Output: results/findings/realworld_heldout/runs/heldout.jsonl.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATASETS="${*:-refusal truthful-qa bias-in-bios sycophancy}"
name="rw_heldout"
script="${SCRIPT_DIR}/generated_jobs/${name}.sh"
mkdir -p "${SCRIPT_DIR}/generated_jobs" "${SCRIPT_DIR}/logs"

# CPU-only (no --gres): the eval runs model forwards on CPU + sklearn probes.
# On `long` a GPU-less job is auto-routed to `long-cpu`.
cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=2:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=4

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

python scripts/realworld_heldout_eval.py \\
    --datasets ${DATASETS} \\
    --data-root \${SSAE_DATA_ROOT} \\
    --output \${SSAE_REPO_ROOT}/results/findings/realworld_heldout/runs/heldout.jsonl
EOF
chmod +x "${script}"

jid=$(sbatch --parsable "${script}")
echo "Submitted held-out eval (${DATASETS}) -> job ${jid}"
echo "  log: ${SCRIPT_DIR}/logs/${name}_${jid}.out"
echo "  out: ${SSAE_REPO_ROOT}/results/findings/realworld_heldout/runs/heldout.jsonl"
