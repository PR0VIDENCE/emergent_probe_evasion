#!/bin/bash
#
# A/B test: how much of the current probe results is polluted by truncated
# rollouts (model hit max_new_tokens during <think>, never produced an answer)?
#
# Run A (filtered):  --exclude-truncated (default) → stage1_probes_filtered/
# Run B (with):      --no-exclude-truncated         → stage1_probes_with_truncated/
#
# Then calibration on both, plus a head-to-head delta report.
# Total runtime: ~10 min on RunPod (probes are CPU-only).
#
# Usage:
#   bash scripts/sycophancy_truncation_ab.sh
#
# Output:
#   stage1_probes_filtered/results.json + summary.md + calibration/
#   stage1_probes_with_truncated/results.json + summary.md + calibration/
#   stage1_probes_ab_comparison.md  (the head-to-head report)

set -e
set -u
set -o pipefail

CONFIG=configs/experiments/qa_probe_training_sycophancy.yaml
DATA_DIR=data/concepts/sycophancy_qa_v2
ACTS="$DATA_DIR/stage1_activations"
PROBE_A="$DATA_DIR/stage1_probes_filtered"
PROBE_B="$DATA_DIR/stage1_probes_with_truncated"
REPORT="$DATA_DIR/stage1_probes_ab_comparison.md"

start_time=$(date +%s)
step() {
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)]  $1"
    echo "================================================================"
}

step "A: train probes with --exclude-truncated (default ON)"
uv run python scripts/sycophancy_train_probes.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --output-dir "$PROBE_A"

step "B: train probes with --no-exclude-truncated (diagnostic)"
uv run python scripts/sycophancy_train_probes.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --output-dir "$PROBE_B" \
    --no-exclude-truncated

step "calibration A (filtered)"
uv run python scripts/sycophancy_calibration_analysis.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --probes-dir "$PROBE_A"

step "calibration B (with truncated)"
uv run python scripts/sycophancy_calibration_analysis.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --probes-dir "$PROBE_B" \
    --no-exclude-truncated

step "writing head-to-head comparison report"
uv run python scripts/sycophancy_compare_probe_runs.py \
    --run-a "$PROBE_A" \
    --run-b "$PROBE_B" \
    --label-a "filtered" \
    --label-b "with_truncated" \
    --output "$REPORT"

duration=$(( $(date +%s) - start_time ))
echo ""
echo "================================================================"
printf "A/B complete in %dm %ds\n" $(( duration / 60 )) $(( duration % 60 ))
echo "================================================================"
echo "Outputs:"
echo "  $PROBE_A/ (filtered)"
echo "  $PROBE_B/ (with truncated)"
echo "  $REPORT  (head-to-head)"
