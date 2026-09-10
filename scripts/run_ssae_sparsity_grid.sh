#!/usr/bin/env bash
#
# Run the code-normalisation x sparsity-constraint ablation:
#
#                 l1                 step_l0
#   norm=ln       baseline           constraint ablation
#   norm=none     norm ablation       interaction check
#
# Usage:
#   scripts/run_ssae_sparsity_grid.sh embeddings.h5 data.yaml --epochs 15000
#   scripts/run_ssae_sparsity_grid.sh --dry-run embeddings.h5 data.yaml
#
# L1 and step-L0 targets have different units. The default reproduces the
# diagnostic 2x2 grid at the same numeric target; for a fair Pareto comparison,
# sweep L1 targets, for example:
#
#   L1_TARGETS="0.001 0.002 0.005 0.01 0.02" \
#     scripts/run_ssae_sparsity_grid.sh embeddings.h5 data.yaml

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--dry-run] <embeddings.h5> <data.yaml> [training options...]" >&2
    exit 2
fi

embeddings=$1
data_cfg=$2
shift 2
extra_args=("$@")

python_bin="${PYTHON_BIN:-python}"
active_threshold="${ACTIVE_THRESHOLD:-0.01}"
step_l0_threshold="${STEP_L0_THRESHOLD:-${active_threshold}}"
step_l0_bandwidth="${STEP_L0_BANDWIDTH:-0.001}"
step_l0_target="${STEP_L0_TARGET:-0.005}"
read -r -a l1_targets <<< "${L1_TARGETS:-0.005}"

run_one() {
    local norm=$1
    local sparsity_type=$2
    local target=$3
    shift 3

    local cmd=(
        "${python_bin}" -m ssae.training
        "${embeddings}" "${data_cfg}"
        "${extra_args[@]}"
        --norm "${norm}"
        --activation none
        --sparsity-type "${sparsity_type}"
        --constraint-mode inequality
        --target "${target}"
        --ind-th "${active_threshold}"
        "$@"
    )

    echo
    echo "norm=${norm} constraint=${sparsity_type} target=${target}"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    if [[ "${dry_run}" == false ]]; then
        "${cmd[@]}"
    fi
}

for norm in ln none; do
    for target in "${l1_targets[@]}"; do
        run_one "${norm}" l1 "${target}"
    done

    run_one "${norm}" step_l0 "${step_l0_target}" \
        --step-l0-threshold "${step_l0_threshold}" \
        --step-l0-bandwidth "${step_l0_bandwidth}"
done
