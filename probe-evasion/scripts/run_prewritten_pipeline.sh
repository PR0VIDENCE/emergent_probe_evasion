#!/bin/bash
# Full prewritten deception probe pipeline — run overnight on RunPod.
#
# Usage:
#   bash scripts/run_prewritten_pipeline.sh 2>&1 | tee /workspace/prewritten_pipeline.log
#
set -e

echo "=========================================="
echo "PREWRITTEN DECEPTION PROBE PIPELINE"
echo "Started: $(date)"
echo "=========================================="

# --- 1. Main pipeline: roleplaying ---
echo ""
echo "=== [1/8] Extract roleplaying activations ==="
uv run python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten.yaml --dataset roleplaying

echo ""
echo "=== [2/8] Train probes ==="
uv run python scripts/train_probes_qa.py --config configs/experiments/deception_prewritten.yaml

echo ""
echo "=== [3/8] Evaluate probes ==="
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/deception_prewritten.yaml

# --- 2. Cross-dataset transfer: AI Liar ---
echo ""
echo "=== [4/8] Extract AI Liar activations ==="
uv run python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten_ai_liar.yaml --dataset ai_liar

echo ""
echo "=== [5/8] Evaluate cross-dataset (roleplaying probes → AI Liar data) ==="
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/deception_prewritten_ai_liar.yaml --probe-dir /workspace/deception_prewritten_data/probes

# --- 3. Leave-one-category-out ---
echo ""
echo "=== [6/8] Leave-one-category-out ==="
for cat in student salesperson employee civilian medical tenant other; do
    echo ""
    echo "--- Holdout: $cat ---"
    uv run python scripts/train_probes_qa.py --config configs/experiments/deception_prewritten.yaml --split-file data/concepts/deception_prewritten/split_holdout_${cat}.yaml
    uv run python scripts/evaluate_probes_qa.py --config configs/experiments/deception_prewritten.yaml --split-file data/concepts/deception_prewritten/split_holdout_${cat}.yaml
done

# --- 4. Neutral control ---
echo ""
echo "=== [7/8] Extract neutral control activations ==="
uv run python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten.yaml --dataset roleplaying --neutral-control --output-dir /workspace/deception_prewritten_neutral

echo ""
echo "=== [8/8] Evaluate neutral control (expect ~0.5 AUROC) ==="
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/deception_prewritten.yaml --probe-dir /workspace/deception_prewritten_data/probes --data-dir /workspace/deception_prewritten_neutral

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
