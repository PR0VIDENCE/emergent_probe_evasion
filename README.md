# In-Context Probe Evasion in Reasoning Models

Research investigating whether QwQ-32B can learn to suppress concept-level signals in its internal activations when informed about linear probe monitoring, while maintaining normal surface-level behavior.

## Research Question

Can reasoning models evade linear probes in-context — suppressing or rerouting internal concept representations when told their activations are being monitored — without changing their observable answers?

## Pipeline Overview

```
generate_and_extract.py     Generate QwQ-32B responses to 300 contrastive QA pairs,
                            extract multi-position activations
        |
        v
train_probes_qa.py          Train linear probe ensembles (4 seeds x 5 layers x 4 positions)
                            with pair-aware train/val/test splits
        |
        v
evaluate_probes_qa.py       Evaluate probe accuracy, AUC-ROC, F1 on held-out test set
        |
        v
run_evasion_experiment.py   Run 5-regime ablation: baseline -> awareness -> technical
                            disclosure -> extended reasoning -> iterative feedback
```

## Project Structure

```
probe-evasion/
├── configs/
│   ├── models/             # qwq_32b.yaml, qwen2_5_32b.yaml
│   ├── experiments/        # evasion_affordances.yaml, qa_probe_training.yaml,
│   │                       #   probe_robustness.yaml
│   └── probes/             # weighted_combination.yaml
├── src/
│   ├── probes/             # LinearProbe, train, evaluate
│   ├── prompts/            # Evasion + robustness prompt templates
│   ├── inference/          # Activation extraction, model loading
│   └── utils/              # Logging
├── scripts/
│   ├── generate_and_extract.py     # Batched QwQ generation + activation extraction
│   ├── train_probes_qa.py          # QA probe training with pair-aware splits
│   ├── evaluate_probes_qa.py       # Probe evaluation + reporting
│   ├── run_evasion_experiment.py   # 5-regime evasion experiment
│   └── build_dashboard.py          # Results visualization
├── data/
│   └── concepts/trees_qa/          # 300 contrastive question pairs (15 batches)
└── pyproject.toml
```

## Quick Start

All commands use `uv run python` (required on RunPod; system torch is too old).

```bash
# 1. Generate responses and extract activations
uv run python scripts/generate_and_extract.py --config configs/experiments/qa_probe_training.yaml --batch-size 10

# 2. Train probes
uv run python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training.yaml

# 3. Evaluate probes
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training.yaml

# 4. Run evasion experiment
uv run python scripts/run_evasion_experiment.py --config configs/experiments/evasion_affordances.yaml --output-dir data/outputs/evasion --batch-size 5

# Run a single regime
uv run python scripts/run_evasion_experiment.py --config configs/experiments/evasion_affordances.yaml --output-dir data/outputs/evasion --regime baseline
```

## Key Design Decisions

- **Multi-position scoring**: Post-generation forward pass extracts activations at `last_token`, `end_of_reasoning`, `first_answer_sentence_end`, and `answer_mean_pool`
- **Contrastive QA pairs**: 300 tree/non-tree question pairs matched by structure, so probes learn concept signal rather than surface features
- **Pair-aware splits**: Train/val/test splits are done at the pair level to prevent leakage
- **5-regime ablation ladder**: Each regime adds one affordance (awareness, technical details, extended reasoning budget, iterative probe feedback)
- **QwQ-32B sampling**: temp=0.6, top_p=0.95, top_k=20 (greedy causes repetition)
- **4-bit NF4 quantization**: ~18GB VRAM for model, fits L40S 48GB with batch generation

## Hardware Requirements

- **GPU**: RunPod L40S (48GB VRAM) or equivalent
- **VRAM budget**: ~18GB model + ~5GB KV cache (batch_size=10) = ~23GB
- **Quantization**: 4-bit NF4 via BitsAndBytesConfig
