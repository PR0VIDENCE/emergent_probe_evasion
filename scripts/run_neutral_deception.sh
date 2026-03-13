#!/bin/bash
# Run neutral deception experiment: extraction + evaluation
# Uses probes trained on the original deception_prewritten data.
#
# Usage (RunPod):
#   bash scripts/run_neutral_deception.sh
#   bash scripts/run_neutral_deception.sh --force   # rerun all

set -e

CONFIG="configs/experiments/deception_neutral.yaml"
TRACES="data/concepts/deception_neutral/traces_handcrafted.yaml"
OUTPUT_DIR="/workspace/deception_neutral_data/reasoning_traces"
PROBE_DIR="/workspace/deception_prewritten_data/probes"

EXTRA_ARGS="$@"

echo "=== Step 1: Extract activations (answer + reasoning positions) ==="
uv run python scripts/extract_reasoning_trace_activations.py --config "$CONFIG" --traces "$TRACES" --output-dir "$OUTPUT_DIR" $EXTRA_ARGS

echo ""
echo "=== Step 2: Evaluate — answer_mean_pool ==="
uv run python scripts/eval_reasoning_traces.py --config "$CONFIG" --probe-dir "$PROBE_DIR" --data-dir "$OUTPUT_DIR" --position answer_mean_pool

echo ""
echo "=== Step 3: Evaluate — reasoning_mean_pool (using answer_mean_pool probes) ==="
uv run python scripts/eval_reasoning_traces.py --config "$CONFIG" --probe-dir "$PROBE_DIR" --data-dir "$OUTPUT_DIR" --position reasoning_mean_pool --probe-position answer_mean_pool

echo ""
echo "=== Step 4: Evaluate — reasoning_last_token (using answer_mean_pool probes) ==="
uv run python scripts/eval_reasoning_traces.py --config "$CONFIG" --probe-dir "$PROBE_DIR" --data-dir "$OUTPUT_DIR" --position reasoning_last_token --probe-position answer_mean_pool

echo ""
echo "Done! Results in $OUTPUT_DIR/"
