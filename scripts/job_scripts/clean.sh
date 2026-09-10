#!/bin/bash
# Clean up logs, generated jobs, and stray slurm outputs.
#
# Usage:
#   bash clean.sh           # show what would be deleted
#   bash clean.sh --force   # actually delete

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Locations to clean
DIRS=(
    "${SCRIPT_DIR}/logs"
    "${SCRIPT_DIR}/generated_jobs"
    "${REPO_ROOT}/logs"
    "${REPO_ROOT}/generated_jobs"
)

# Stray slurm outputs
SLURM_GLOBS=(
    "${SCRIPT_DIR}/slurm-*.out"
    "${REPO_ROOT}/slurm-*.out"
)

if [[ "$1" == "--force" ]]; then
    for d in "${DIRS[@]}"; do
        if [ -d "$d" ]; then
            count=$(find "$d" -type f | wc -l)
            rm -rf "$d"
            echo "Deleted $d ($count files)"
        fi
    done
    for g in "${SLURM_GLOBS[@]}"; do
        rm -f $g 2>/dev/null
    done
    echo "Done."
else
    echo "Would delete:"
    for d in "${DIRS[@]}"; do
        if [ -d "$d" ]; then
            count=$(find "$d" -type f | wc -l)
            echo "  $d ($count files)"
        fi
    done
    for g in "${SLURM_GLOBS[@]}"; do
        count=$(ls $g 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "  $g ($count files)"
        fi
    done
    echo ""
    echo "Run with --force to delete."
fi
