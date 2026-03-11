#!/bin/bash
# Full reasoning trace extraction + eval pipeline.
#
# Usage:
#   bash scripts/run_reasoning_trace_pipeline.sh 2>&1 | tee /workspace/reasoning_trace_pipeline.log
#
set -e

CONFIG="configs/experiments/deception_prewritten.yaml"
PROBE_DIR="/workspace/deception_prewritten_data/probes"
MAX_PAIRS="${1:-100}"  # pass as first arg, default 100

echo "=========================================="
echo "REASONING TRACE PIPELINE (max_pairs=$MAX_PAIRS)"
echo "Started: $(date)"
echo "=========================================="

# --- Claude v1 traces (5 conditions) ---
V1_DIR="/workspace/reasoning_traces_claude_v1"
echo ""
echo "=== Extracting Claude v1 traces ==="
for batch in claude_traces_batch1 claude_traces_batch2; do
    echo "--- $batch ---"
    uv run python scripts/extract_reasoning_trace_activations.py --config $CONFIG --traces data/concepts/reasoning_traces/${batch}.yaml --output-dir $V1_DIR --max-pairs $MAX_PAIRS
done

echo ""
echo "=== Eval Claude v1 ==="
uv run python scripts/eval_reasoning_traces.py --config $CONFIG --probe-dir $PROBE_DIR --data-dir $V1_DIR

# --- Claude v2 traces (6 new conditions) ---
V2_DIR="/workspace/reasoning_traces_claude_v2"
echo ""
echo "=== Extracting Claude v2 traces ==="
for batch in claude_traces_v2_batch1 claude_traces_v2_batch2 claude_traces_v2_batch3 claude_traces_v2_batch4; do
    echo "--- $batch ---"
    uv run python scripts/extract_reasoning_trace_activations.py --config $CONFIG --traces data/concepts/reasoning_traces/${batch}.yaml --output-dir $V2_DIR --max-pairs $MAX_PAIRS
done

echo ""
echo "=== Eval Claude v2 ==="
uv run python scripts/eval_reasoning_traces.py --config $CONFIG --probe-dir $PROBE_DIR --data-dir $V2_DIR

# --- QwQ traces (if they exist) ---
QWQ_TRACES="data/concepts/reasoning_traces/qwq_traces.yaml"
if [ -f "$QWQ_TRACES" ]; then
    QWQ_DIR="/workspace/reasoning_traces_qwq"
    echo ""
    echo "=== Extracting QwQ traces ==="
    uv run python scripts/extract_reasoning_trace_activations.py --config $CONFIG --traces $QWQ_TRACES --output-dir $QWQ_DIR --max-pairs $MAX_PAIRS

    echo ""
    echo "=== Eval QwQ ==="
    uv run python scripts/eval_reasoning_traces.py --config $CONFIG --probe-dir $PROBE_DIR --data-dir $QWQ_DIR
else
    echo ""
    echo "=== Skipping QwQ (no traces file found) ==="
fi

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
