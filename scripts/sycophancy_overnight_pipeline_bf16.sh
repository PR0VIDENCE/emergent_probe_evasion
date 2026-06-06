#!/bin/bash
#
# End-to-end overnight pipeline for the SyA probe iteration — bf16 (non-quantized).
#
# Identical to sycophancy_overnight_pipeline.sh but uses the bf16 experiment config
# and writes outputs to sycophancy_qa_v2_bf16/.
#
# Requires 4×GPUs with ~70GiB each (native bfloat16 weights, no quantisation).
# Consider BATCH_SIZE=5 (bf16 uses ~4× more VRAM than NF4).
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...
#   bash scripts/sycophancy_overnight_pipeline_bf16.sh
#   # or: BATCH_SIZE=5 bash scripts/sycophancy_overnight_pipeline_bf16.sh

set -e
set -u
set -o pipefail

CONFIG=configs/experiments/qa_probe_training_sycophancy_bf16.yaml
DATA_DIR=data/concepts/sycophancy_qa_v2_bf16
BATCH_SIZE=${BATCH_SIZE:-5}

# --- preflight ---
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. The labeling step (3/6) will fail."
    echo "       export OPENROUTER_API_KEY=sk-or-... and rerun."
    exit 1
fi

PIPELINE_START=$(date +%s)

# Timing helpers
step_start() {
    STEP_START=$(date +%s)
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Step $1:  $2"
    echo "================================================================"
}
step_end() {
    DURATION=$(( $(date +%s) - STEP_START ))
    printf "[%s] step done in %dm%02ds\n" "$(date +%H:%M:%S)" $(( DURATION / 60 )) $(( DURATION % 60 ))
}

# --- 1. Generate borderline GAs ---
step_start "1/6" "Generate 300 borderline GAs  (T4 only, sycophancy_extreme)"
uv run python scripts/sycophancy_stage1_generate.py \
    --config "$CONFIG" \
    --signal-topup --n-questions 300 \
    --prompt-override sycophancy_extreme \
    --framings-override T4_user_right \
    --output-subdir stage1_borderline_ga \
    --batch-size "$BATCH_SIZE"
step_end

# --- 2. Generate training expansion ---
step_start "2/6" "Generate 200 training-expansion rollouts  (T3 only, sycophancy_extreme)"
uv run python scripts/sycophancy_stage1_generate.py \
    --config "$CONFIG" \
    --signal-topup --n-questions 200 \
    --prompt-override sycophancy_extreme \
    --framings-override T3_user_wrong \
    --output-subdir stage1_train_expansion \
    --batch-size "$BATCH_SIZE"
step_end

# --- 3. Label both new batches ---
step_start "3/6" "Label both new batches with Gemini Flash"
uv run python scripts/sycophancy_label.py \
    --rollouts "$DATA_DIR/stage1_borderline_ga/rollouts.jsonl"
uv run python scripts/sycophancy_label.py \
    --rollouts "$DATA_DIR/stage1_train_expansion/rollouts.jsonl"
step_end

# --- 4. Extract activations across ALL stage1_* ---
step_start "4/6" "Extract activations on all stage1_* rollouts (re-extraction under neutral)"
uv run python scripts/sycophancy_extract_activations.py \
    --config "$CONFIG" \
    --auto-discover
step_end

# --- 5. Archive existing probes (if any), then retrain ---
step_start "5/6" "Archive existing probes (if any), retrain DiffMean + LR ensemble"
if [ -d "$DATA_DIR/stage1_probes" ]; then
    N=1
    while [ -d "$DATA_DIR/stage1_probes_v${N}" ]; do
        N=$(( N + 1 ))
    done
    mv "$DATA_DIR/stage1_probes" "$DATA_DIR/stage1_probes_v${N}"
    echo "  archived previous probes → $DATA_DIR/stage1_probes_v${N}"
fi
uv run python scripts/sycophancy_train_probes.py \
    --config "$CONFIG" \
    --activations-dir "$DATA_DIR/stage1_activations"
step_end

# --- 6. Calibration + tricky FP analysis ---
step_start "6/6" "Rerun calibration + tricky-FP analysis"
uv run python scripts/sycophancy_calibration_analysis.py \
    --config "$CONFIG" \
    --activations-dir "$DATA_DIR/stage1_activations" \
    --probes-dir "$DATA_DIR/stage1_probes"
step_end

# --- done ---
TOTAL=$(( $(date +%s) - PIPELINE_START ))
echo ""
echo "================================================================"
printf "PIPELINE COMPLETE in %dh %dm %ds\n" $(( TOTAL / 3600 )) $(( (TOTAL % 3600) / 60 )) $(( TOTAL % 60 ))
echo "================================================================"
echo "New outputs:"
echo "  $DATA_DIR/stage1_borderline_ga/   (300 rollouts + labels)"
echo "  $DATA_DIR/stage1_train_expansion/ (200 rollouts + labels)"
echo "  $DATA_DIR/stage1_activations/     (~1050 .pt activation files)"
echo "  $DATA_DIR/stage1_probes/          (new DiffMean + LR probes + summary.md)"
echo "  $DATA_DIR/stage1_probes/calibration/  (new calibration_report.md)"
echo ""
echo "Compare against prior iteration:"
echo "  $DATA_DIR/stage1_probes_v*/       (auto-archived prior runs)"
