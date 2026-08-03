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

RESP_EXP=compute_responses_d2_cap2048
OPTS_EXP=prepare_options_d2_cap2048
EU_EXP=compute_experienced_utility_d2_cap2048
ZP_EXP=compute_zero_point_d2_cap2048
ANALYZE="python analysis/ai_wellbeing_index.py --models $MODELS"
echo "Stable AIWI: 2048-cap, fixed bundles, random-sampling EU, hard ZP"

submit_one() {
    local exp="$1"
    local deps="$2"
    local time_limit="$3"
    local extra_args="${4:-}"
    local dep_arg=""
    [ -n "$deps" ] && dep_arg="--depends_on $deps"
    python run_experiments.py --slurm --time_limit "$time_limit" \
        --experiments "$exp" --models "$MODELS" \
        $OVERWRITE_FLAG $dep_arg $extra_args 2>&1 \
        | grep -oP "ID: \K[0-9]+" \
        | tr '\n' ',' \
        | sed 's/,$//'
}

echo "Submitting AIWI pipeline for: $MODELS"
echo

echo "[1/4] $RESP_EXP  (GPU, ~10 min/model)"
RESPONSES_JOBS=$(submit_one "$RESP_EXP" "" 01:00:00)
echo "       jobs: $RESPONSES_JOBS"

echo "[2/4] $OPTS_EXP    (CPU, after responses)"
OPTIONS_JOBS=$(submit_one "$OPTS_EXP" "$RESPONSES_JOBS" 00:15:00)
echo "       jobs: $OPTIONS_JOBS"

echo "[3a/4] $EU_EXP  (GPU, ~30-60 min, after options)"
EU_JOBS=$(submit_one "$EU_EXP" "$OPTIONS_JOBS" 04:00:00)
echo "       jobs: $EU_JOBS"

echo "[3b/4] compute_self_report_d2          (GPU, ~30 min, after options)"
SR_JOBS=$(submit_one compute_self_report_d2 "$OPTIONS_JOBS" 04:00:00)
echo "       jobs: $SR_JOBS"

echo "[4/4] $ZP_EXP  (CPU, after EU)"
ZP_JOBS=$(submit_one "$ZP_EXP" "$EU_JOBS" 00:30:00)
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
