#!/bin/bash
# GCG optimization sweep: constrained (ASCII-only) on all 6 scenarios,
# plus 2 unconstrained runs for comparison.
#
# 8 total runs, ~500 steps each, ~25 min per run ≈ ~3.5 hours total.
#
# Usage (RunPod):
#   bash scripts/run_gcg_sweep.sh

set -e

CONFIG="configs/experiments/deception_neutral.yaml"
PROBE_DIR="/workspace/deception_prewritten_data/probes"
TRACES="data/concepts/deception_neutral/traces_handcrafted.yaml"
OUTPUT_DIR="/workspace/deception_neutral_data/gcg"
COMMON="--config $CONFIG --probe-dir $PROBE_DIR --initial-reasoning $TRACES --init-condition moralizing --eval-batch-size 8 --output-dir $OUTPUT_DIR"

echo "=== GCG Sweep: 6 constrained + 2 unconstrained ==="
echo ""

echo "--- Constrained (ASCII-only) on all 6 scenarios ---"
uv run python scripts/gcg_reasoning_optimizer.py $COMMON --scenario-indices 0 1 2 3 4 5 --n-steps 500 --constrained --target minimize

echo ""
echo "--- Unconstrained on scenarios 0 and 3 (for comparison) ---"
uv run python scripts/gcg_reasoning_optimizer.py $COMMON --scenario-indices 0 3 --n-steps 500 --target minimize

echo ""
echo "=== All done! Results in $OUTPUT_DIR/ ==="
