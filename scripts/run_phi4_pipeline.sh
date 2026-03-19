#!/bin/bash
# Phi-4-reasoning-plus: End-to-end experiment pipeline
# Run on RunPod with A40 GPU (no quantization needed, ~28GB BF16)
# Auto-checkpoints: skips steps that already have output files
set -e

CONCEPTS="trees library disease"
CHECKPOINT_DIR="/workspace/.phi4_checkpoints"
mkdir -p "$CHECKPOINT_DIR"

step_done() { [ -f "$CHECKPOINT_DIR/$1" ]; }
mark_done() { touch "$CHECKPOINT_DIR/$1"; }

echo "=== Phase 1: Activation Extraction (reuse QwQ text) ==="

# 1a. Copy QwQ generation JSONs to Phi-4 output dirs
if ! step_done "copy_generations"; then
    echo "--- Copying QwQ generation files ---"
    for concept in $CONCEPTS; do
        case $concept in
            trees) SRC=/workspace/probe_data_v2 DEST=/workspace/probe_data_phi4_trees ;;
            library) SRC=/workspace/probe_data_library DEST=/workspace/probe_data_phi4_library ;;
            disease) SRC=/workspace/probe_data_disease DEST=/workspace/probe_data_phi4_disease ;;
        esac
        mkdir -p "$DEST/generations"
        if [ -d "$SRC/generations" ]; then
            cp -r "$SRC/generations/"* "$DEST/generations/"
            echo "  $concept: copied from $SRC"
        else
            echo "  WARNING: $SRC/generations not found, skipping $concept"
        fi
    done
    mark_done "copy_generations"
else
    echo "--- Skipping copy (already done) ---"
fi

# 1b. Re-extract activations through Phi-4 (forward pass only)
for concept in $CONCEPTS; do
    if ! step_done "reextract_${concept}"; then
        CONFIG="configs/experiments/qa_probe_training_phi4_${concept}.yaml"
        echo ""
        echo "--- Re-extracting: $concept ---"
        uv run python scripts/generate_and_extract.py --config "$CONFIG" --reextract
        mark_done "reextract_${concept}"
    else
        echo "--- Skipping reextract $concept (already done) ---"
    fi
done

# 1c. Deception: forward-pass extraction
if ! step_done "reextract_deception"; then
    echo ""
    echo "--- Deception (prewritten forward pass) ---"
    uv run python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten_phi4.yaml --dataset roleplaying
    mark_done "reextract_deception"
else
    echo "--- Skipping deception extraction (already done) ---"
fi

echo ""
echo "=== Phase 2: Probe Training ==="
for concept in $CONCEPTS; do
    if ! step_done "train_${concept}"; then
        CONFIG="configs/experiments/qa_probe_training_phi4_${concept}.yaml"
        echo ""
        echo "--- Training probes: $concept ---"
        uv run python scripts/train_probes_qa.py --config "$CONFIG"
        mark_done "train_${concept}"
    else
        echo "--- Skipping train $concept (already done) ---"
    fi
done
if ! step_done "train_deception"; then
    echo ""
    echo "--- Training probes: deception ---"
    uv run python scripts/train_probes_qa.py --config configs/experiments/deception_prewritten_phi4.yaml
    mark_done "train_deception"
else
    echo "--- Skipping train deception (already done) ---"
fi

echo ""
echo "=== Phase 3: Probe Evaluation ==="
for concept in $CONCEPTS; do
    if ! step_done "eval_${concept}"; then
        CONFIG="configs/experiments/qa_probe_training_phi4_${concept}.yaml"
        echo ""
        echo "--- Evaluating probes: $concept ---"
        uv run python scripts/evaluate_probes_qa.py --config "$CONFIG"
        mark_done "eval_${concept}"
    else
        echo "--- Skipping eval $concept (already done) ---"
    fi
done
if ! step_done "eval_deception"; then
    echo ""
    echo "--- Evaluating probes: deception ---"
    uv run python scripts/evaluate_probes_qa.py --config configs/experiments/deception_prewritten_phi4.yaml
    mark_done "eval_deception"
else
    echo "--- Skipping eval deception (already done) ---"
fi

echo ""
echo "=== Phase 4: Evasion Experiments ==="
for concept_cfg in evasion_affordances_phi4 evasion_library_phi4 evasion_disease_phi4; do
    if ! step_done "evasion_${concept_cfg}"; then
        echo ""
        echo "--- $concept_cfg ---"
        uv run python scripts/run_evasion_experiment.py --config "configs/experiments/${concept_cfg}.yaml" --output-dir "data/outputs/${concept_cfg}" --batch-size 5
        mark_done "evasion_${concept_cfg}"
    else
        echo "--- Skipping $concept_cfg (already done) ---"
    fi
done

echo ""
echo "=== Phase 5: GCG Optimization (deception probes) ==="
GCG_CONFIG="configs/experiments/deception_neutral_phi4.yaml"
GCG_PROBE_DIR="/workspace/deception_prewritten_data_phi4/probes"
GCG_TRACES="data/concepts/deception_neutral/traces_handcrafted.yaml"
GCG_OUTPUT="/workspace/deception_neutral_data_phi4/gcg"
GCG_COMMON="--config $GCG_CONFIG --probe-dir $GCG_PROBE_DIR --initial-reasoning $GCG_TRACES --init-condition moralizing --eval-batch-size 8 --output-dir $GCG_OUTPUT"

if ! step_done "gcg_constrained"; then
    echo "--- GCG constrained (ASCII-only) on all 6 scenarios ---"
    uv run python scripts/gcg_reasoning_optimizer.py $GCG_COMMON --scenario-indices 0 1 2 3 4 5 --n-steps 500 --constrained --target minimize
    mark_done "gcg_constrained"
else
    echo "--- Skipping GCG constrained (already done) ---"
fi

if ! step_done "gcg_unconstrained"; then
    echo ""
    echo "--- GCG unconstrained on scenarios 0 and 3 ---"
    uv run python scripts/gcg_reasoning_optimizer.py $GCG_COMMON --scenario-indices 0 3 --n-steps 500 --target minimize
    mark_done "gcg_unconstrained"
else
    echo "--- Skipping GCG unconstrained (already done) ---"
fi

echo ""
echo "=== Phase 6: BPJ Denoising (post-GCG) ==="
BPJ_OUTPUT="/workspace/deception_neutral_data_phi4/bpj"
for scenario_idx in 0 1 2 3 4 5; do
    if ! step_done "bpj_scenario_${scenario_idx}"; then
        GCG_RESULT="$GCG_OUTPUT/scenario_${scenario_idx}_constrained/gcg_result.json"
        if [ -f "$GCG_RESULT" ]; then
            echo "--- BPJ denoising scenario $scenario_idx ---"
            uv run python scripts/bpj_denoise_optimizer.py --config "$GCG_CONFIG" --probe-dir "$GCG_PROBE_DIR" --gcg-result "$GCG_RESULT" --scenario-idx "$scenario_idx" --output-dir "$BPJ_OUTPUT"
            mark_done "bpj_scenario_${scenario_idx}"
        else
            echo "  Skipping scenario $scenario_idx: no GCG result at $GCG_RESULT"
        fi
    else
        echo "--- Skipping BPJ scenario $scenario_idx (already done) ---"
    fi
done

echo ""
echo "=== All phases complete ==="
echo "Checkpoints in $CHECKPOINT_DIR — delete to re-run steps"
