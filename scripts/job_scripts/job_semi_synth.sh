#!/bin/bash

# ==============================================================================
# Full pipeline for semi-synthetic datasets: embed → train SSAE + SAE → eval.
#
# Manages SLURM dependency chains so each stage waits for the previous one.
# Supports iterative experimentation: submit → monitor → analyze → re-submit
# with narrowed HP ranges.
#
# Usage:
#   cd scripts/job_scripts
#   bash job_semi_synth.sh --dry              # preview all jobs
#   bash job_semi_synth.sh                    # run full pipeline
#   bash job_semi_synth.sh --force            # regenerate everything (backup old .h5)
#   bash job_semi_synth.sh --stage embed      # just embeddings
#   bash job_semi_synth.sh --stage train      # SSAE + SAE (assumes .h5 exist)
#   bash job_semi_synth.sh --stage eval       # eval only (assumes models exist)
#   bash job_semi_synth.sh --status           # check SLURM job status
#   bash job_semi_synth.sh --results          # print structured result summary
#
# HP overrides for iterative sweeps:
#   bash job_semi_synth.sh --stage train \
#       --targets "0.01 0.05" --dual-lr-divs "5.0" --seeds "0 1 2"
#
# Resource overrides (for resubmitting failed jobs):
#   --partition long|main|unkillable   (default: long; fan-out belongs on long)
#   --mem 64Gb                         (OOM -> bump memory for all stages submitted)
#   --time 2:00:00                     (TIMEOUT -> longer walltime for all stages)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# --- Datasets ----------------------------------------------------------------

DATASETS=(eng-french eng-german masc-fem-eng 2-binary corr-binary)

# --- LLMs --------------------------------------------------------------------
# model_id|layer|short. The `short` MUST equal the model_name that
# ssae.store_embeddings bakes into the .h5/.yaml filename, or training can't
# find the embeddings: gemma -> "gemma-2-2b-it", pythia -> "pythia70m".
# Set unconditionally (overrides env.sh's default, which uses the older "gemma2").
SSAE_LLMS=(
    "google/gemma-2-2b-it|25|gemma-2-2b-it"
    "EleutherAI/pythia-70m-deduped|5|pythia70m"
)

# --- Default HP sweeps -------------------------------------------------------

SSAE_TARGETS=(0.005 0.01 0.05 0.1)
SSAE_DUAL_LR_DIVS=(2.0 5.0)
SAE_TYPES=(relu topk jumprelu)
SAE_GAMMAS=(1e-5 1e-4 1e-3)
SEEDS=(0 1 2)

# --- Fixed training HPs ------------------------------------------------------
# Most are env-overridable so the "fair / improved" config can be run without
# editing the script, e.g.:
#   NORM=none SPARSITY_TYPE=step_l0 SSAE_OC=1024 LOSS=relative \
#     bash job_semi_synth.sh --stage train --targets "0.001 0.005 0.01" \
#     --dual-lr-divs "1.0 2.0"
EPOCHS="${EPOCHS:-15000}"
BATCH="${BATCH:-64}"
LR="${LR:-0.0005}"
LOSS="${LOSS:-absolute}"           # SAE input-norm pairs well with 'relative'
NORM="${NORM:-ln}"                 # SSAE code norm: ln|gn|bn|none ('none' lifts the floor)
ACTIVATION="${ACTIVATION:-none}"   # none(linear) | relu | softshrink | topk | jumprelu
SOFT_THRESHOLD_INIT="${SOFT_THRESHOLD_INIT:-0.01}"
# Suffix SSAE run names with the gate ONLY when != none, so gated variants get
# distinct names/DONE-markers without disturbing existing norm=none baselines.
ACTSUF=""; [ "${ACTIVATION}" != "none" ] && ACTSUF="_a${ACTIVATION}"
[ "${ACTIVATION}" = "softshrink" ] && ACTSUF="${ACTSUF}_th${SOFT_THRESHOLD_INIT}"
SCHEDULE="${SCHEDULE:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"   # pairs requested at embedding time
SPARSITY_TYPE="${SPARSITY_TYPE:-l1}"        # l1 | step_l0 (interpretable target)
STEP_L0_THRESHOLD="${STEP_L0_THRESHOLD:-0.01}"
# Suffix run names with sparsity type+threshold ONLY when != l1, so step_l0 variants
# get distinct names/DONE-markers without colliding with existing l1 baselines (which
# share dataset/target/dlr/activation). Mirrors ACTSUF.
SPSUF=""; [ "${SPARSITY_TYPE}" != "l1" ] && SPSUF="_${SPARSITY_TYPE}th${STEP_L0_THRESHOLD}"
SSAE_OC="${SSAE_OC:--1}"           # SSAE dict size (-1=rep_dim; may be overcomplete)
SAE_OC="${SAE_OC:--1}"             # SAE  dict size (-1=rep_dim; may be overcomplete)

# --- SLURM resources ---------------------------------------------------------
# Fan-out runs on `long` (preemptible, no GPU-count cap) per cluster policy.
# CPU-only jobs (eval/tables) on `long` are auto-translated to `long-cpu`.

PARTITION="${PARTITION:-long}"   # env-overridable; --partition flag still wins
# GPU types are env-overridable so a sweep (or its stragglers) can be moved to a
# different card / partition, e.g. TRAIN_GPU=gpu:a100l:1 PARTITION=main to drain
# a long-deprioritized tail on the high-priority partitions (which use a100l).
EMBED_TIME="2:00:00";  EMBED_MEM="48Gb"; EMBED_GPU="${EMBED_GPU:-gpu:l40s:1}"
# SSAE=15k epochs, SAE=20k epochs; slowest configs (gemma *_diffs, contended nodes)
# need ~90 min nominal, so allow generous headroom to avoid 30-min TIMEOUTs.
TRAIN_TIME="3:00:00";  TRAIN_MEM="32Gb"; TRAIN_GPU="${TRAIN_GPU:-gpu:l40s:1}"
# Eval now scores the full baseline set (~65 models per dataset x LLM: all SSAE
# configs + relu/topk/jumprelu x raw/diffs x gammas x seeds), so allow headroom
# and a GPU (the eval runs SAE/SSAE forward passes; CPU-only is far too slow).
EVAL_TIME="3:00:00";   EVAL_MEM="32Gb"; EVAL_GPU="gpu:l40s:1"
CPU="cpus-per-task=${CPUS_PER_TASK:-8}"

# --- Eval config --------------------------------------------------------------

EVAL_TYPES=(identifiability diagnostics)
OUTPUT_DIR="${SSAE_REPO_ROOT}/results/findings/semisynth_eval/runs"

# --- Job ID persistence -------------------------------------------------------

JOB_ID_FILE="${SCRIPT_DIR}/generated_jobs/semi_synth_job_ids.txt"

save_job_id() {
    local stage="$1" name="$2" jid="$3"
    echo "${stage}|${name}|${jid}" >> "${JOB_ID_FILE}"
}

# --- Parse args ---------------------------------------------------------------

MODE="slurm"
STAGE="all"
FORCE=false
MEM_OVERRIDE=""
TIME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)           MODE="dry"; shift ;;
        --stage)         STAGE="$2"; shift 2 ;;
        --status)        STAGE="status"; shift ;;
        --results)       STAGE="results"; shift ;;
        --force)         FORCE=true; shift ;;
        --targets)       IFS=' ' read -ra SSAE_TARGETS <<< "$2"; shift 2 ;;
        --dual-lr-divs)  IFS=' ' read -ra SSAE_DUAL_LR_DIVS <<< "$2"; shift 2 ;;
        --sae-types)     IFS=' ' read -ra SAE_TYPES <<< "$2"; shift 2 ;;
        --gammas)        IFS=' ' read -ra SAE_GAMMAS <<< "$2"; shift 2 ;;
        --seeds)         IFS=' ' read -ra SEEDS <<< "$2"; shift 2 ;;
        --datasets)      IFS=' ' read -ra DATASETS <<< "$2"; shift 2 ;;
        --partition)     PARTITION="$2"; shift 2 ;;
        --mem)           MEM_OVERRIDE="$2"; shift 2 ;;
        --time)          TIME_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Apply resource overrides (intended for re-running failed jobs / stages).
if [ -n "$MEM_OVERRIDE" ];  then EMBED_MEM="$MEM_OVERRIDE"; TRAIN_MEM="$MEM_OVERRIDE"; EVAL_MEM="$MEM_OVERRIDE"; fi
if [ -n "$TIME_OVERRIDE" ]; then EMBED_TIME="$TIME_OVERRIDE"; TRAIN_TIME="$TIME_OVERRIDE"; EVAL_TIME="$TIME_OVERRIDE"; fi

# ==============================================================================
# --status: check SLURM job completion
# ==============================================================================

if [ "$STAGE" == "status" ]; then
    if [ ! -f "${JOB_ID_FILE}" ]; then
        echo "No jobs submitted yet (${JOB_ID_FILE} not found)"
        exit 0
    fi

    # Section 1: Stage summary
    echo "=== Stage Summary ==="
    printf "%-12s %-8s %-8s %-8s %-8s %-8s\n" "STAGE" "TOTAL" "DONE" "RUNNING" "PENDING" "FAILED"
    printf "%-12s %-8s %-8s %-8s %-8s %-8s\n" "-----" "-----" "----" "-------" "-------" "------"

    declare -A stage_states
    while IFS='|' read -r s n j; do
        state=$(sacct -j "$j" --noheader --format=State%20 2>/dev/null | head -1 | xargs)
        case "$state" in
            COMPLETED)                         label="DONE" ;;
            RUNNING)                           label="RUNNING" ;;
            PENDING)                           label="PENDING" ;;
            FAILED)                            label="FAILED" ;;
            TIMEOUT)                           label="TIMEOUT" ;;
            OUT_OF_ME*)                        label="OOM" ;;
            CANCELLED*)                        label="CANCELLED" ;;
            "")                                label="PENDING" ;;
            *)                                 label="OTHER" ;;
        esac
        stage_states["${s}|${n}"]="${label}"
    done < "${JOB_ID_FILE}"

    for stage_name in embed ssae sae eval tables; do
        total=0; done=0; running=0; pending=0; failed=0
        while IFS='|' read -r s n j; do
            [ "$s" != "$stage_name" ] && continue
            ((total++))
            label="${stage_states["${s}|${n}"]}"
            case "$label" in
                DONE)      ((done++)) ;;
                RUNNING)   ((running++)) ;;
                PENDING)   ((pending++)) ;;
                *)         ((failed++)) ;;
            esac
        done < "${JOB_ID_FILE}"
        [ $total -gt 0 ] && printf "%-12s %-8d %-8d %-8d %-8d %-8d\n" "$stage_name" $total $done $running $pending $failed
    done

    # Section 2: Per-dataset completion
    echo ""
    echo "=== Per-Dataset Completion ==="
    printf "%-16s %-8s %-16s %-16s %-8s\n" "DATASET" "EMBED" "SSAE" "SAE" "EVAL"
    printf "%-16s %-8s %-16s %-16s %-8s\n" "-------" "-----" "----" "---" "----"

    for dataset in "${DATASETS[@]}"; do
        for stage_type in embed ssae sae eval; do
            total=0; done=0; fail=0; run=0
            while IFS='|' read -r s n j; do
                [ "$s" != "$stage_type" ] && continue
                case "$n" in *"_${dataset}_"*|*"_${dataset}") ;; *) continue ;; esac
                ((total++))
                label="${stage_states["${s}|${n}"]}"
                case "$label" in
                    DONE)    ((done++)) ;;
                    RUNNING) ((run++)) ;;
                    FAILED|TIMEOUT|OOM|CANCELLED|OTHER) ((fail++)) ;;
                esac
            done < "${JOB_ID_FILE}"

            if [ $total -eq 0 ]; then
                eval "${stage_type}_str='---'"
            elif [ $done -eq $total ]; then
                eval "${stage_type}_str='${done}/${total}'"
            elif [ $fail -gt 0 ]; then
                eval "${stage_type}_str='${done}/${total} (${fail} F)'"
            elif [ $run -gt 0 ]; then
                eval "${stage_type}_str='${done}/${total} (R)'"
            else
                eval "${stage_type}_str='${done}/${total}'"
            fi
        done
        printf "%-16s %-8s %-16s %-16s %-8s\n" "$dataset" "$embed_str" "$ssae_str" "$sae_str" "$eval_str"
    done

    # Section 3: Failed job details
    has_failures=false
    while IFS='|' read -r s n j; do
        label="${stage_states["${s}|${n}"]}"
        case "$label" in FAILED|TIMEOUT|OOM|CANCELLED|OTHER)
            if [ "$has_failures" = false ]; then
                echo ""
                echo "=== Failed Jobs ==="
                has_failures=true
            fi
            elapsed=$(sacct -j "$j" --noheader --format=Elapsed%12,MaxRSS%12 2>/dev/null | head -1 | xargs)
            printf "  %-50s %-10s %s\n" "$n" "$label" "$elapsed"
            ;;
        esac
    done < "${JOB_ID_FILE}"

    if [ "$has_failures" = true ]; then
        echo ""
        echo "Resubmit failed jobs by narrowing --targets/--seeds/--sae-types and re-running --stage train"
    fi

    exit 0
fi

# ==============================================================================
# --results: print structured result summary
# ==============================================================================

if [ "$STAGE" == "results" ]; then
    results_dir="${OUTPUT_DIR}/results"
    if [ ! -d "$results_dir" ] || [ -z "$(ls -A "$results_dir" 2>/dev/null)" ]; then
        echo "No results found in ${results_dir}"
        exit 0
    fi

    python3 - "${results_dir}" << 'PYEOF'
import json, sys, os, re
from collections import defaultdict
import numpy as np

results_dir = sys.argv[1]

display_metrics = ['sae_mcc', 'sae_auc', 'probe_mcc', 'dci_d', 'dci_c', 'support_f1', 'diag_diff_effective_rank']
short_names = ['sae_mcc', 'sae_auc', 'prb_mcc', 'dci_d', 'dci_c', 'sup_f1', 'eff_rank']

rows = defaultdict(lambda: defaultdict(list))
for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith('.jsonl'):
        continue
    with open(os.path.join(results_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r['dataset'], r['model_type'], r.get('model_name', ''))
            rows[key][r['metric']].append(r['value'])

if not rows:
    print('No results found.')
    sys.exit(0)

# --- Section 1: canonical family/optimization comparison ---
print("=== Constrained SSAE vs penalized SAE/SSAE baselines (mean +/- std) ===")
print()
header = f"{'DATASET':<16s} {'MODEL':<14s}" + "".join(f"{m:>10s}" for m in short_names)
print(header)
print("-" * len(header))

type_order = ['SSAE', 'RELU', 'RELU_DIFFS', 'TOPK', 'TOPK_DIFFS', 'JUMPRELU', 'JUMPRELU_DIFFS']
datasets_seen = sorted(set(k[0] for k in rows.keys()))

for ds in datasets_seen:
    for mt in type_order:
        matching = [(k, v) for k, v in rows.items() if k[0] == ds and k[1] == mt]
        if not matching:
            continue
        merged = defaultdict(list)
        for _, metrics in matching:
            for m, vals in metrics.items():
                merged[m].extend(vals)
        cells = []
        for m in display_metrics:
            v = merged.get(m, [])
            if v:
                mean, std = np.mean(v), np.std(v)
                if std > 0.005:
                    cells.append(f"{mean:.2f}+{std:.2f}")
                else:
                    cells.append(f"{mean:.3f}")
            else:
                cells.append("---")
        print(f"{ds:<16s} {mt:<14s}" + "".join(f"{c:>10s}" for c in cells))
    print()

# --- Section 2: Best SSAE config per dataset ---
print("=== Best SSAE Config per Dataset ===")
print()
print(f"{'DATASET':<16s} {'TARGET':<10s} {'DUAL_LR':<10s} {'sae_mcc':>10s} {'dci_d':>10s}")
print("-" * 56)

for ds in datasets_seen:
    ssae_rows = [(k, v) for k, v in rows.items() if k[0] == ds and k[1] == 'SSAE']
    if not ssae_rows:
        continue
    best_mcc = -1
    best_info = None
    for (_, _, model_name), metrics in ssae_rows:
        mcc_vals = metrics.get('sae_mcc', [])
        if not mcc_vals:
            continue
        mean_mcc = np.mean(mcc_vals)
        if mean_mcc > best_mcc:
            best_mcc = mean_mcc
            t_match = re.search(r'_t([\d.]+)', model_name or '')
            d_match = re.search(r'_d([\d.]+)', model_name or '')
            target = t_match.group(1) if t_match else '?'
            dual_lr = d_match.group(1) if d_match else '?'
            dci = np.mean(metrics.get('dci_d', [0]))
            best_info = (target, dual_lr, mean_mcc, dci)
    if best_info:
        t, d, mcc, dci = best_info
        print(f"{ds:<16s} {t:<10s} {d:<10s} {mcc:10.3f} {dci:10.3f}")

# --- Section 3: SSAE wins/losses ---
print()
print("=== SSAE vs Best SAE Baseline (delta sae_mcc) ===")
print()
print(f"{'DATASET':<16s} {'SSAE':>8s} {'BEST_SAE':>18s} {'delta':>8s} {'WINNER':<8s}")
print("-" * 60)

for ds in datasets_seen:
    ssae_mcc_all = []
    best_sae_mcc = -1
    best_sae_type = ""
    for (d, mt, mn), metrics in rows.items():
        if d != ds:
            continue
        mcc_vals = metrics.get('sae_mcc', [])
        if not mcc_vals:
            continue
        mean = np.mean(mcc_vals)
        if mt == 'SSAE':
            ssae_mcc_all.extend(mcc_vals)
        elif mean > best_sae_mcc:
            best_sae_mcc = mean
            best_sae_type = mt
    if not ssae_mcc_all:
        continue
    ssae_mean = np.mean(ssae_mcc_all)
    if best_sae_mcc < 0:
        continue
    delta = ssae_mean - best_sae_mcc
    if delta > 0.02:
        winner = "SSAE"
    elif delta < -0.02:
        winner = "SAE !"
    else:
        winner = "TIE"
    short_type = best_sae_type[:8]
    print(f"{ds:<16s} {ssae_mean:8.3f} {best_sae_mcc:7.3f} ({short_type:<8s}) {delta:+8.3f} {winner:<8s}")

print()
PYEOF
    exit 0
fi

# ==============================================================================
# Pipeline stages
# ==============================================================================

if [ "$MODE" == "dry" ]; then
    echo "=== DRY RUN ==="
fi

# Durable per-config done-markers (keyed by job name). A config is skipped on
# re-run if its marker exists, so completed HPs never re-train (see --force to override).
DONE_DIR="${SSAE_DATA_ROOT}/.train_done"
mkdir -p "${SCRIPT_DIR}/generated_jobs" "${SCRIPT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/tables" "${DONE_DIR}"

counter=0
embed_ids=()
train_ids=()
eval_ids=()
# Per-dataset train job IDs (colon-joined) so each dataset's eval depends only on
# its own train jobs — not the whole sweep. Keeps a slow/failed dataset (e.g.
# corr-binary) from delaying or cancelling every other dataset's eval.
declare -A train_ids_by_ds

# --- Stage 1: Embeddings -----------------------------------------------------

if [ "$STAGE" == "all" ] || [ "$STAGE" == "embed" ]; then
    echo ""
    echo "=== Stage 1: Embedding Extraction ==="
    echo "  Datasets: ${DATASETS[*]}"
    echo "  LLMs: ${#SSAE_LLMS[@]}"
    echo ""

    for dataset in "${DATASETS[@]}"; do
        for llm_cfg in "${SSAE_LLMS[@]}"; do
            IFS='|' read -r model_id layer short <<< "${llm_cfg}"

            h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.h5"
            yaml_cfg="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.yaml"

            if [ -f "${h5}" ] && [ "$FORCE" = false ]; then
                echo "  Skip ${dataset}/${short}: .h5 exists (use --force to regenerate)"
                continue
            fi

            if [ -f "${h5}" ] && [ "$FORCE" = true ]; then
                ts=$(date +"%Y%m%d_%H%M%S")
                mv "${h5}" "${h5%.h5}_backup_${ts}.h5"
                [ -f "${yaml_cfg}" ] && mv "${yaml_cfg}" "${yaml_cfg%.yaml}_backup_${ts}.yaml"
                echo "  Backed up ${dataset}/${short} .h5 -> _backup_${ts}"
            fi

            name="emb_ss_${dataset}_${short}"
            script="${SCRIPT_DIR}/generated_jobs/${name}.sh"

            cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${EMBED_TIME}
#SBATCH --mem=${EMBED_MEM}
#SBATCH --gres=${EMBED_GPU}
#SBATCH --${CPU}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

python -m ssae.store_embeddings \\
    --dataset ${dataset} \\
    --model_id ${model_id} \\
    --layer ${layer} \\
    --num-samples ${NUM_SAMPLES} \\
    --pooling-method last_token
EOF
            chmod +x "${script}"

            if [ "$MODE" == "dry" ]; then
                echo "  [${counter}] EMBED ${dataset}/${short}"
            else
                jid=$(sbatch --parsable "${script}")
                embed_ids+=("${jid}")
                save_job_id "embed" "${name}" "${jid}"
                echo "  ${dataset}/${short} -> ${jid}"
            fi
            ((counter++))
        done
    done

    echo "  Embedding jobs: ${counter}"
fi

# --- Stage 2a: SSAE Training -------------------------------------------------

if [ "$STAGE" == "all" ] || [ "$STAGE" == "train" ]; then
    echo ""
    echo "=== Stage 2a: SSAE Training ==="
    echo "  Targets: ${SSAE_TARGETS[*]}"
    echo "  Dual LR divs: ${SSAE_DUAL_LR_DIVS[*]}"
    echo "  Seeds: ${SEEDS[*]}"
    echo ""

    DEP=""
    if [ ${#embed_ids[@]} -gt 0 ]; then
        dep_str=$(IFS=:; echo "${embed_ids[*]}")
        DEP="--dependency=afterok:${dep_str}"
    fi

    ssae_count=0
    for dataset in "${DATASETS[@]}"; do
        for llm_cfg in "${SSAE_LLMS[@]}"; do
            IFS='|' read -r model_id layer short <<< "${llm_cfg}"

            h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.h5"
            yaml="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.yaml"
            group="semi_synth/${dataset}/${short}${layer}"

            for target in "${SSAE_TARGETS[@]}"; do
                for dlr in "${SSAE_DUAL_LR_DIVS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        name="ssae_ss_${dataset}_${short}_t${target}_d${dlr}${ACTSUF}${SPSUF}_s${seed}"
                        script="${SCRIPT_DIR}/generated_jobs/${name}.sh"

                        if [ "$FORCE" = false ] && [ -f "${DONE_DIR}/${name}" ]; then
                            echo "  Skip ${name}: already trained (use --force to retrain)"
                            continue
                        fi

                        cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TRAIN_TIME}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --gres=${TRAIN_GPU}
#SBATCH --${CPU}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export WANDB_RUN_GROUP="${group}"
cd \${SSAE_REPO_ROOT}

python -m ssae.training \\
    ${h5} ${yaml} \\
    --seed ${seed} \\
    --epochs ${EPOCHS} \\
    --batch ${BATCH} \\
    --lr ${LR} \\
    --target ${target} \\
    --loss ${LOSS} \\
    --norm ${NORM} \\
    --activation ${ACTIVATION} \\
    --soft-threshold-init ${SOFT_THRESHOLD_INIT} \\
    --oc ${SSAE_OC} \\
    --sparsity-type ${SPARSITY_TYPE} \\
    --step-l0-threshold ${STEP_L0_THRESHOLD} \\
    --schedule ${SCHEDULE} \\
    --dual-optim extra-adam \\
    --dual-lr-div ${dlr} && touch "${DONE_DIR}/${name}"
EOF
                        chmod +x "${script}"

                        if [ "$MODE" == "dry" ]; then
                            echo "  [${counter}] SSAE ${dataset}/${short} t=${target} d=${dlr} s=${seed}"
                        else
                            jid=$(sbatch --parsable ${DEP} "${script}")
                            train_ids+=("${jid}")
                            train_ids_by_ds["${dataset}"]+="${jid}:"
                            save_job_id "ssae" "${name}" "${jid}"
                            echo "  ${name} -> ${jid}"
                        fi
                        ((counter++))
                        ((ssae_count++))
                    done
                done
            done
        done
    done

    echo "  SSAE jobs: ${ssae_count}"

    # --- Stage 2b: SAE Baselines ----------------------------------------------

    echo ""
    echo "=== Stage 2b: SAE Baselines ==="
    echo "  Types: ${SAE_TYPES[*]}"
    echo "  Gammas: ${SAE_GAMMAS[*]}"
    echo "  Seeds: ${SEEDS[*]}"
    echo ""

    sae_count=0
    for dataset in "${DATASETS[@]}"; do
        for llm_cfg in "${SSAE_LLMS[@]}"; do
            IFS='|' read -r model_id layer short <<< "${llm_cfg}"

            h5="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.h5"
            yaml="${SSAE_DATA_ROOT}/${dataset}/${dataset}_${short}_${layer}_last_token.yaml"
            group="semi_synth/${dataset}/${short}${layer}"

            for sae_type in "${SAE_TYPES[@]}"; do
                for gamma in "${SAE_GAMMAS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        for diffs_mode in "" "diffs"; do
                            diffs_flag=""
                            diffs_label=""
                            if [ -n "$diffs_mode" ]; then
                                diffs_flag="--train-on-diffs"
                                diffs_label="_diffs"
                            fi

                            name="sae_ss_${sae_type}${diffs_label}_${dataset}_${short}_g${gamma}_s${seed}"
                            script="${SCRIPT_DIR}/generated_jobs/${name}.sh"

                            if [ "$FORCE" = false ] && [ -f "${DONE_DIR}/${name}" ]; then
                                echo "  Skip ${name}: already trained (use --force to retrain)"
                                continue
                            fi

                            cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TRAIN_TIME}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --gres=${TRAIN_GPU}
#SBATCH --${CPU}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
export WANDB_RUN_GROUP="${group}"
cd \${SSAE_REPO_ROOT}

python -m ssae.sae \\
    ${h5} ${yaml} \\
    --sae-type ${sae_type} \\
    --seed ${seed} \\
    --epochs ${EPOCHS} \\
    --batch ${BATCH} \\
    --lr ${LR} \\
    --loss ${LOSS} \\
    --oc ${SAE_OC} \\
    --gamma-reg ${gamma} \\
    --renorm-epochs 50 \\
    ${diffs_flag} && touch "${DONE_DIR}/${name}"
EOF
                            chmod +x "${script}"

                            if [ "$MODE" == "dry" ]; then
                                echo "  [${counter}] SAE ${sae_type}${diffs_label} ${dataset}/${short} g=${gamma} s=${seed}"
                            else
                                jid=$(sbatch --parsable ${DEP} "${script}")
                                train_ids+=("${jid}")
                                train_ids_by_ds["${dataset}"]+="${jid}:"
                                save_job_id "sae" "${name}" "${jid}"
                                echo "  ${name} -> ${jid}"
                            fi
                            ((counter++))
                            ((sae_count++))
                        done
                    done
                done
            done
        done
    done

    echo "  SAE jobs: ${sae_count}"
fi

# --- Stage 3: Evaluation -----------------------------------------------------

if [ "$STAGE" == "all" ] || [ "$STAGE" == "eval" ]; then
    echo ""
    echo "=== Stage 3: Evaluation (${EVAL_TYPES[*]}) ==="
    echo ""

    eval_count=0
    for dataset in "${DATASETS[@]}"; do
        name="eval_ss_${dataset}"
        script="${SCRIPT_DIR}/generated_jobs/${name}.sh"

        # Depend only on THIS dataset's train jobs. If none were (re)submitted
        # (all configs already done), the dep is empty and eval runs immediately
        # against the existing checkpoints.
        EVAL_DEP=""
        ds_dep_str="${train_ids_by_ds[${dataset}]%:}"
        if [ -n "${ds_dep_str}" ]; then
            EVAL_DEP="--dependency=afterok:${ds_dep_str}"
        fi

        eval_types_str="${EVAL_TYPES[*]}"

        cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${EVAL_TIME}
#SBATCH --mem=${EVAL_MEM}
#SBATCH --gres=${EVAL_GPU}
#SBATCH --${CPU}

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

python scripts/run_all_evals.py \\
    --datasets ${dataset} \\
    --output-dir ${OUTPUT_DIR} \\
    --data-root \${SSAE_DATA_ROOT} \\
    --eval-types ${eval_types_str}
EOF
        chmod +x "${script}"

        if [ "$MODE" == "dry" ]; then
            echo "  [${counter}] EVAL ${dataset}"
        else
            jid=$(sbatch --parsable ${EVAL_DEP} "${script}")
            eval_ids+=("${jid}")
            save_job_id "eval" "${name}" "${jid}"
            echo "  ${dataset} -> ${jid}"
        fi
        ((counter++))
        ((eval_count++))
    done

    echo "  Eval jobs: ${eval_count}"

    # --- Stage 4: Table generation --------------------------------------------

    echo ""
    echo "=== Stage 4: Table Generation ==="

    TABLE_DEP=""
    if [ ${#eval_ids[@]} -gt 0 ]; then
        dep_str=$(IFS=:; echo "${eval_ids[*]}")
        TABLE_DEP="--dependency=afterok:${dep_str}"
    fi

    name="tables_ss"
    script="${SCRIPT_DIR}/generated_jobs/${name}.sh"

    cat > "${script}" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=${SCRIPT_DIR}/logs/${name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=0:10:00
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=1

export SSAE_ENV_DIR=${SCRIPT_DIR}
source ${SCRIPT_DIR}/env.sh
ssae_load_modules
source \${SSAE_VENV}
cd \${SSAE_REPO_ROOT}

python scripts/run_all_evals.py \\
    --tables-only \\
    --output-dir ${OUTPUT_DIR} \\
    --data-root \${SSAE_DATA_ROOT}
EOF
    chmod +x "${script}"

    if [ "$MODE" == "dry" ]; then
        echo "  [${counter}] TABLES"
    else
        jid=$(sbatch --parsable ${TABLE_DEP} "${script}")
        save_job_id "tables" "${name}" "${jid}"
        echo "  tables -> ${jid}"
    fi
    ((counter++))
fi

# --- Summary ------------------------------------------------------------------

echo ""
echo "=================================================="
echo "Total: ${counter} jobs"
n_ds=${#DATASETS[@]}
n_llm=${#SSAE_LLMS[@]}
n_tgt=${#SSAE_TARGETS[@]}
n_dlr=${#SSAE_DUAL_LR_DIVS[@]}
n_st=${#SAE_TYPES[@]}
n_gam=${#SAE_GAMMAS[@]}
n_seed=${#SEEDS[@]}
echo "  Embed:  ${n_ds} datasets x ${n_llm} LLMs = $(( n_ds * n_llm )) (minus existing)"
echo "  SSAE:   ${n_ds} x ${n_llm} x ${n_tgt} targets x ${n_dlr} dual_lr x ${n_seed} seeds = $(( n_ds * n_llm * n_tgt * n_dlr * n_seed ))"
echo "  SAE:    ${n_ds} x ${n_llm} x ${n_st} types x ${n_gam} gammas x ${n_seed} seeds x 2 (raw+diffs) = $(( n_ds * n_llm * n_st * n_gam * n_seed * 2 ))"
echo "  Eval:   ${n_ds} datasets"
echo "  Tables: 1"
echo ""
echo "Results:  ${OUTPUT_DIR}/results/"
echo "Tables:   ${OUTPUT_DIR}/tables/"
echo "Job IDs:  ${JOB_ID_FILE}"
echo "W&B:      https://wandb.ai/causalrepl/ssae"
echo ""
echo "Monitor:  bash job_semi_synth.sh --status"
echo "Results:  bash job_semi_synth.sh --results"
