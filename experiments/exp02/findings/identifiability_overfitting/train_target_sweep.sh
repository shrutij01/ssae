#!/usr/bin/env bash
#SBATCH --job-name=dci_predictor_train
#SBATCH --output=/path/to/ssae/results/findings/identifiability_overfitting_v1/runs/logs/%A_%a.out
#SBATCH --error=/path/to/ssae/results/findings/identifiability_overfitting_v1/runs/logs/%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16Gb
#SBATCH --time=0:30:00
#SBATCH --array=0-11

set -euo pipefail

repo_root=/path/to/ssae
python_bin=/path/to/ssae-venv/bin/python
study_root=/path/to/ssae-data/studies/test_identifiability_overfitting_v1
embeddings=${study_root}/data/eng-french_pythia70m_5_last_token.h5
data_cfg=${study_root}/data/eng-french_pythia70m_5_last_token.yaml
targets=(0.02 0.05 0.1 0.2)
seeds=(0 1 2)

target_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
target=${targets[${target_index}]}
seed=${seeds[${seed_index}]}
target_label=${target/./p}
output_dir=${study_root}/checkpoints/target_${target_label}/seed${seed}

if [[ -e ${output_dir} ]] && [[ -n $(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "Refusing to overwrite non-empty output: ${output_dir}" >&2
    exit 3
fi

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
mkdir -p "${output_dir}" "${study_root}/wandb"
export PYTHONPATH=${repo_root}:${PYTHONPATH:-}
export WANDB_MODE=offline
export WANDB_DIR=${study_root}/wandb
export WANDB_RUN_GROUP=test_identifiability_overfitting_v1/eng-french/pythia
cd "${repo_root}"

echo "target=${target} seed=${seed} task=${SLURM_ARRAY_TASK_ID}"
echo "git_head=$(git rev-parse HEAD)"
git diff --no-ext-diff -- ssae/training.py | sha256sum

"${python_bin}" -u -m ssae.training \
    "${embeddings}" "${data_cfg}" \
    --batch 64 \
    --max-updates 105000 \
    --lr 0.0005 \
    --loss relative \
    --oc 512 \
    --output-dir "${output_dir}" \
    --input-norm rms \
    --zero-biases \
    --augment-reverse-diffs \
    --schedule-unit updates \
    --warmup 14000 \
    --schedule 7000 \
    --renorm-updates 1 \
    --norm ln \
    --activation none \
    --sparsity-type l1 \
    --target "${target}" \
    --constraint-mode inequality \
    --constrained-optim extrapolation \
    --formulation lagrangian \
    --dual-optim extra-adam \
    --dual-lr-div 2.0 \
    --ind-th 0.01 \
    --seed "${seed}"
