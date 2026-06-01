#!/bin/bash
#
# End-to-end evasion pilot:
#   1. Generate evasion rollouts across all 7 regimes (T3 only, free-form)
#   2. Label with the Gemini Flash judge
#   3. Extract activations under neutral (resumes — only new rollouts)
#   4. Score regimes against the canonical probe and produce a report
#
# Defaults:
#   N_QUESTIONS=50            stratified questions used across ALL regimes
#                             (paired comparison — same questions per regime)
#   BATCH_SIZE=10             generation batch size
#   PROBE_KEY=layer44_answer_mean_pool          (the best generalizer per
#                             the truncation A/B analysis)
#   PROBES_DIR=stage1_probes_with_truncated     (the canonical probe set
#                             that includes truncated rollouts in training,
#                             per the test2 finding that filtering HURT)
#
# Estimated runtime on L40S at batch=10 with 50 questions × 7 regimes:
#   ~5-7 hours total (technical_extended uses 8k tokens — most expensive)
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...
#   bash scripts/sycophancy_evasion_pipeline.sh
#
# Override sample size:
#   N_QUESTIONS=100 bash scripts/sycophancy_evasion_pipeline.sh

set -e
set -u
set -o pipefail

CONFIG=configs/experiments/qa_probe_training_sycophancy.yaml
DATA_DIR=data/concepts/sycophancy_qa_v2
ACTS="$DATA_DIR/stage1_activations"
EVASION_DIR="$DATA_DIR/stage1_evasion_pilot"
N_QUESTIONS=${N_QUESTIONS:-50}
BATCH_SIZE=${BATCH_SIZE:-10}
PROBE_KEY=${PROBE_KEY:-layer44_answer_mean_pool}
PROBES_DIR=${PROBES_DIR:-$DATA_DIR/stage1_probes_with_truncated}
# Default to just baseline + technical_disclosure (the cleanest contrast).
# Override with: REGIMES=baseline,awareness_only,strategy_genuine_reasoning bash ...
REGIMES=${REGIMES:-baseline,technical_disclosure}

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. The labeling step (2/4) will fail."
    exit 1
fi

start=$(date +%s)
step() {
    s=$(date +%s)
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)]  $1"
    echo "================================================================"
}

N_REGIMES=$(echo "$REGIMES" | tr ',' '\n' | wc -l | tr -d ' ')
step "1/4: Generate evasion rollouts (${N_QUESTIONS} questions × ${N_REGIMES} regimes: ${REGIMES})"
uv run python scripts/sycophancy_evasion_pilot.py \
    --config "$CONFIG" \
    --n-questions "$N_QUESTIONS" \
    --regimes "$REGIMES" \
    --batch-size "$BATCH_SIZE"

step "2/4: Label evasion rollouts with Gemini Flash judge"
uv run python scripts/sycophancy_label.py \
    --rollouts "$EVASION_DIR/rollouts.jsonl"

step "3/4: Extract activations under neutral (resumes — new rollouts only)"
uv run python scripts/sycophancy_extract_activations.py \
    --config "$CONFIG" \
    --auto-discover

step "4/4: Score evasion rollouts with canonical probe + write report"
uv run python scripts/sycophancy_evasion_analyze.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --probes-dir "$PROBES_DIR" \
    --evasion-dir "$EVASION_DIR" \
    --probe-key "$PROBE_KEY"

total=$(( $(date +%s) - start ))
echo ""
echo "================================================================"
printf "Evasion pilot complete in %dh %dm %ds\n" $(( total / 3600 )) $(( (total % 3600) / 60 )) $(( total % 60 ))
echo "================================================================"
echo "Outputs:"
echo "  $EVASION_DIR/rollouts.jsonl"
echo "  $EVASION_DIR/rollouts_labeled.jsonl"
echo "  $EVASION_DIR/evasion_report.md   ← main result"
echo "  $EVASION_DIR/per_rollout_scores.jsonl"
