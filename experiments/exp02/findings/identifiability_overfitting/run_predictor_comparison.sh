#!/usr/bin/env bash
#SBATCH --job-name=dci_predictor_eval
#SBATCH --output=/path/to/ssae/results/findings/identifiability_overfitting_v1/runs/logs/%j_eval.out
#SBATCH --error=/path/to/ssae/results/findings/identifiability_overfitting_v1/runs/logs/%j_eval.err
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32Gb
#SBATCH --time=2:00:00

set -euo pipefail

repo_root=/path/to/ssae
venv_root=/path/to/ssae-venv

module load python/3.10 2>/dev/null || true
if [[ -x ${venv_root}/bin/python ]]; then
    python_bin=${venv_root}/bin/python
    export PYTHONPATH=${repo_root}:${PYTHONPATH:-}
else
    # Some long-cpu nodes do not provide /usr/bin/python3.10, which is the
    # venv interpreter symlink target. Use the loaded 3.10 module with the
    # exact same venv site-packages in that case.
    python_bin=$(command -v python)
    export PYTHONPATH=${repo_root}:${venv_root}/lib/python3.10/site-packages:${PYTHONPATH:-}
fi
export MPLCONFIGDIR=/tmp/ssae-matplotlib-${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "${repo_root}"

"${python_bin}" -c 'import matplotlib, sklearn, torch; print("analysis environment OK")'

"${python_bin}" -u \
    experiments/exp02/findings/identifiability_overfitting/compare_dci_predictors.py \
    --permutation-repeats 32 \
    --predictor-jobs 4
