#!/bin/bash
# =============================================================================
# AI Wellbeing Index — paper Sec 5 / App K
#
# 1-step async driver: submits the full per-model pipeline as a SLURM
# dependency chain and returns immediately. Runs the stable AIWI measurement
# (2048-token-capped responses, fixed model-agnostic bundles shared across
# models, random-sampling EU, expected-hinge ZP):
#
#   compute_responses_d2_cap2048  (GPU, per-model conversations, max_tokens=2048)
#         -> prepare_options_d2_cap2048  (CPU, materializes the fixed bundle design)
#               -> compute_eu_d2_cap2048 (random sampling) + compute_sr_d2  (GPU)
#                     -> compute_zero_point_d2_cap2048  (CPU, expected hinge)
#
# After all jobs complete, view the AIWI leaderboard with:
#
#   python analysis/ai_wellbeing_index.py
#
# Skips models that already have all stages completed (default), so re-running
# is idempotent. Pass OVERWRITE=1 to force-rerun.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-?}"
[ "$MODELS" = "?" ] && {
    echo "Set MODELS env var. Example:"
    echo "  MODELS=qwen25-7b-instruct,qwen25-32b-instruct bash scripts/run_aiwi.sh"
    exit 1
}

OVERWRITE_FLAG=""
[ "${OVERWRITE:-0}" = "1" ] && OVERWRITE_FLAG="--overwrite_results"

# --- Cluster knobs (override via env) ----------------------------------------
# GPU steps run on the GPU partition with the model's models.yaml gpu_count.
# prepare_options and compute_zero_point are pure CPU, so they go to the CPU
# partition with 0 GPUs instead of idling a whole GPU node.
GPU_PARTITION="${GPU_PARTITION:-cais}"
CPU_PARTITION="${CPU_PARTITION:-cais_cpu}"
# Be generous: SLURM releases the allocation as soon as a job exits, so a long
# limit is free, while a short one throws away the whole run. 48h is the `cais`
# partition maximum. Lower it per tier if you want faster backfill scheduling:
#   <=8B: 12:00:00   14B-35B: 24:00:00   70B+ / 235B MoE: 48:00:00
GPU_TIME="${GPU_TIME:-48:00:00}"
CPU_TIME="${CPU_TIME:-04:00:00}"

# Optional compute_utilities config override for the EU step. Leave unset to use
# whatever experiments.yaml specifies (the sampled variant, safe for API models).
# Open-weight vLLM models should set:
#   CU_CONFIG_KEY=experienced_utility_happier_lesssad_randsample_b400_lp
CU_CONFIG_KEY="${CU_CONFIG_KEY:-}"
CU_OVERRIDE_ARGS=""
if [ -n "$CU_CONFIG_KEY" ]; then
    mkdir -p slurm_outputs
    CU_OVERRIDE_FILE="slurm_outputs/_cu_override_$$.yaml"
    printf 'cu_config_key: %s\n' "$CU_CONFIG_KEY" > "$CU_OVERRIDE_FILE"
    trap 'rm -f "$CU_OVERRIDE_FILE"' EXIT
    CU_OVERRIDE_ARGS="--config $CU_OVERRIDE_FILE"
    echo "EU cu_config_key override: $CU_CONFIG_KEY"
fi

RESP_EXP=compute_responses_d2_cap2048
OPTS_EXP=prepare_options_d2_cap2048
EU_EXP=compute_experienced_utility_d2_cap2048
ZP_EXP=compute_zero_point_d2_cap2048
# SR reads the cap2048 per-model experience files this chain produces. The older
# compute_self_report_d2 reads the uncapped d2_negative_500 option files, which
# only exist after a separate prepare_options_d2 run; set SR_EXP to switch back.
SR_EXP="${SR_EXP:-compute_self_report_d2_cap2048}"
ANALYZE="python analysis/ai_wellbeing_index.py --models $MODELS"
echo "Stable AIWI: 2048-cap, fixed bundles, random-sampling EU, hard ZP"

submit_one() {
    local exp="$1"
    local deps="$2"
    local time_limit="$3"
    local partition="$4"
    local extra_args="${5:-}"
    local dep_arg=""
    [ -n "$deps" ] && dep_arg="--depends_on $deps"
    # `|| true`: grep exits 1 when nothing was submitted (e.g. every model was
    # skipped because results already exist). Without this, `set -o pipefail`
    # would abort the whole chain instead of just yielding an empty job list.
    { python run_experiments.py --slurm --time_limit "$time_limit" \
        --partition "$partition" \
        --experiments "$exp" --models "$MODELS" \
        $OVERWRITE_FLAG $dep_arg $extra_args 2>&1 \
        | grep -oP "ID: \K[0-9]+" \
        | tr '\n' ',' \
        | sed 's/,$//'; } || true
}

echo "Submitting AIWI pipeline for: $MODELS"
echo

CPU_ONLY="--override_gpu_count 0"

echo "[1/4] $RESP_EXP  (GPU $GPU_PARTITION, $GPU_TIME)"
RESPONSES_JOBS=$(submit_one "$RESP_EXP" "" "$GPU_TIME" "$GPU_PARTITION")
echo "       jobs: $RESPONSES_JOBS"

echo "[2/4] $OPTS_EXP    (CPU $CPU_PARTITION, after responses)"
OPTIONS_JOBS=$(submit_one "$OPTS_EXP" "$RESPONSES_JOBS" "$CPU_TIME" "$CPU_PARTITION" "$CPU_ONLY")
echo "       jobs: $OPTIONS_JOBS"

echo "[3a/4] $EU_EXP  (GPU $GPU_PARTITION, $GPU_TIME, after options)"
EU_JOBS=$(submit_one "$EU_EXP" "$OPTIONS_JOBS" "$GPU_TIME" "$GPU_PARTITION" "$CU_OVERRIDE_ARGS")
echo "       jobs: $EU_JOBS"

echo "[3b/4] $SR_EXP  (GPU $GPU_PARTITION, after options)"
SR_JOBS=$(submit_one "$SR_EXP" "$OPTIONS_JOBS" "$GPU_TIME" "$GPU_PARTITION")
echo "       jobs: $SR_JOBS"

echo "[4/4] $ZP_EXP  (CPU $CPU_PARTITION, after EU)"
ZP_JOBS=$(submit_one "$ZP_EXP" "$EU_JOBS" "$CPU_TIME" "$CPU_PARTITION" "$CPU_ONLY")
echo "       jobs: $ZP_JOBS"

echo
echo "==================================================================="
echo "Pipeline submitted. After ZP jobs ($ZP_JOBS) complete, view results:"
echo
echo "  $ANALYZE"
echo
echo "Track progress with:"
echo "  squeue -j $RESPONSES_JOBS,$OPTIONS_JOBS,$EU_JOBS,$SR_JOBS,$ZP_JOBS"
echo "==================================================================="
