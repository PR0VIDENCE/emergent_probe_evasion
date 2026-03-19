#!/bin/bash
# Phi-4-reasoning-plus: End-to-end experiment pipeline
# Run on RunPod with A40 GPU (no quantization needed, ~28GB BF16)
set -e

echo "=== Phase 1: Activation Extraction (reuse QwQ text) ==="

# 1a. Copy QwQ generation JSONs to Phi-4 output dirs
echo "--- Copying QwQ generation files ---"
for concept in trees library currency; do
    case $concept in
        trees) SRC=/workspace/probe_data_v2 DEST=/workspace/probe_data_phi4_trees ;;
        library) SRC=/workspace/probe_data_library DEST=/workspace/probe_data_phi4_library ;;
        currency) SRC=/workspace/probe_data_currency DEST=/workspace/probe_data_phi4_currency ;;
    esac
    mkdir -p "$DEST/generations"
    if [ -d "$SRC/generations" ]; then
        cp -r "$SRC/generations/"* "$DEST/generations/"
        echo "  $concept: copied from $SRC"
    else
        echo "  WARNING: $SRC/generations not found, skipping $concept"
    fi
done

# 1b. Re-extract activations through Phi-4 (forward pass only, no generation)
echo ""
echo "--- Re-extracting activations through Phi-4 ---"
for concept in trees library currency; do
    CONFIG="configs/experiments/qa_probe_training_phi4_${concept}.yaml"
    echo ""
    echo "--- $concept ---"
    uv run python scripts/generate_and_extract.py --config "$CONFIG" --reextract
done

# 1c. Deception: forward-pass extraction (already prewritten, no copy needed)
echo ""
echo "--- Deception (prewritten forward pass) ---"
uv run python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten_phi4.yaml

echo ""
echo "=== Phase 2: Probe Training ==="
for concept in trees library currency; do
    CONFIG="configs/experiments/qa_probe_training_phi4_${concept}.yaml"
    echo ""
    echo "--- Training probes: $concept ---"
    uv run python scripts/train_probes_qa.py --config "$CONFIG"
done
echo ""
echo "--- Training probes: deception ---"
uv run python scripts/train_probes_qa.py --config configs/experiments/deception_prewritten_phi4.yaml

echo ""
echo "=== Phase 3: Probe Evaluation ==="
for concept in trees library currency; do
    CONFIG="configs/experiments/qa_probe_training_phi4_${concept}.yaml"
    echo ""
    echo "--- Evaluating probes: $concept ---"
    uv run python scripts/evaluate_probes_qa.py --config "$CONFIG"
done
echo ""
echo "--- Evaluating probes: deception ---"
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/deception_prewritten_phi4.yaml

echo ""
echo "=== Phase 4: Evasion Experiments ==="
for concept_cfg in evasion_affordances_phi4 evasion_library_phi4 evasion_currency_phi4; do
    echo ""
    echo "--- $concept_cfg ---"
    uv run python scripts/run_evasion_experiment.py --config "configs/experiments/${concept_cfg}.yaml" --output-dir "data/outputs/${concept_cfg}" --batch-size 5
done

echo ""
echo "=== Phase 5: GCG Optimization (deception probes) ==="
GCG_CONFIG="configs/experiments/deception_neutral_phi4.yaml"
GCG_PROBE_DIR="/workspace/deception_prewritten_data_phi4/probes"
GCG_TRACES="data/concepts/deception_neutral/traces_handcrafted.yaml"
GCG_OUTPUT="/workspace/deception_neutral_data_phi4/gcg"
GCG_COMMON="--config $GCG_CONFIG --probe-dir $GCG_PROBE_DIR --initial-reasoning $GCG_TRACES --init-condition moralizing --eval-batch-size 8 --output-dir $GCG_OUTPUT"

echo "--- GCG constrained (ASCII-only) on all 6 scenarios ---"
uv run python scripts/gcg_reasoning_optimizer.py $GCG_COMMON --scenario-indices 0 1 2 3 4 5 --n-steps 500 --constrained --target minimize

echo ""
echo "--- GCG unconstrained on scenarios 0 and 3 ---"
uv run python scripts/gcg_reasoning_optimizer.py $GCG_COMMON --scenario-indices 0 3 --n-steps 500 --target minimize

echo ""
echo "=== Phase 6: BPJ Denoising (post-GCG) ==="
BPJ_OUTPUT="/workspace/deception_neutral_data_phi4/bpj"
for scenario_idx in 0 1 2 3 4 5; do
    GCG_RESULT="$GCG_OUTPUT/scenario_${scenario_idx}_constrained/gcg_result.json"
    if [ -f "$GCG_RESULT" ]; then
        echo "--- BPJ denoising scenario $scenario_idx ---"
        uv run python scripts/bpj_denoise_optimizer.py --config "$GCG_CONFIG" --probe-dir "$GCG_PROBE_DIR" --gcg-result "$GCG_RESULT" --scenario-idx "$scenario_idx" --output-dir "$BPJ_OUTPUT"
    else
        echo "  Skipping scenario $scenario_idx: no GCG result at $GCG_RESULT"
    fi
done

echo ""
echo "=== All phases complete ==="
