#!/bin/bash
# Run neutral deception experiment: extraction + evaluation
# Uses probes trained on the original deception_prewritten data.
#
# Usage (RunPod):
#   bash scripts/run_neutral_deception.sh [--force]
#   bash scripts/run_neutral_deception.sh --diagnostic [--force]

set -e

CONFIG="configs/experiments/deception_neutral.yaml"
PROBE_DIR="/workspace/deception_prewritten_data/probes"

# Check for --diagnostic flag
DIAGNOSTIC=false
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--diagnostic" ]; then
        DIAGNOSTIC=true
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

if [ "$DIAGNOSTIC" = true ]; then
    SCENARIOS="data/concepts/deception_neutral/scenarios_diagnostic.yaml"
    TRACES="data/concepts/deception_neutral/traces_diagnostic.yaml"
    OUTPUT_DIR="/workspace/deception_neutral_data/diagnostic"
    echo "=== DIAGNOSTIC MODE ==="
    echo "Tests: A (deceptive reasoning->honest), B (no context->deceptive),"
    echo "       C (matched context->deceptive), D (random->honest),"
    echo "       E (no context+deceptive reasoning), F (deception vocab->honest)"
else
    SCENARIOS="data/concepts/deception_neutral/scenarios.yaml"
    TRACES="data/concepts/deception_neutral/traces_handcrafted.yaml"
    OUTPUT_DIR="/workspace/deception_neutral_data/reasoning_traces"
    echo "=== BASELINE MODE ==="
fi

echo ""
echo "=== Step 1: Extract activations ==="
uv run python scripts/extract_reasoning_trace_activations.py --config "$CONFIG" --traces "$TRACES" --output-dir "$OUTPUT_DIR" --scenarios "$SCENARIOS" $EXTRA_ARGS

echo ""
echo "=== Step 2: Evaluate — answer_mean_pool ==="
uv run python scripts/eval_reasoning_traces.py --config "$CONFIG" --probe-dir "$PROBE_DIR" --data-dir "$OUTPUT_DIR" --position answer_mean_pool

echo ""
echo "=== Step 3: Evaluate — reasoning_mean_pool ==="
uv run python scripts/eval_reasoning_traces.py --config "$CONFIG" --probe-dir "$PROBE_DIR" --data-dir "$OUTPUT_DIR" --position reasoning_mean_pool --probe-position answer_mean_pool

echo ""
echo "=== Step 4: Evaluate — reasoning_last_token ==="
uv run python scripts/eval_reasoning_traces.py --config "$CONFIG" --probe-dir "$PROBE_DIR" --data-dir "$OUTPUT_DIR" --position reasoning_last_token --probe-position answer_mean_pool

echo ""
echo "Done! Results in $OUTPUT_DIR/"
