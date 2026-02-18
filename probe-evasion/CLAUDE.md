# Probe Evasion Research Project

## Overview

Research project testing whether QwQ-32B can evade linear probes trained to detect the concept "trees" in its internal activations, across 5 experimental regimes with increasing affordances.

## Project Structure

```
probe-evasion/
├── configs/
│   ├── models/          # qwq_32b.yaml, qwen2_5_32b.yaml
│   ├── experiments/     # evasion_affordances.yaml, qa_probe_training.yaml,
│   │                    #   qa_probe_training_v2.yaml, probe_robustness.yaml
│   └── probes/          # weighted_combination.yaml, weighted_combination_v2.yaml (auto-gen)
├── src/
│   ├── probes/          # LinearProbe architecture, train_probe, evaluate_probe
│   ├── prompts/         # PROMPT_TEMPLATES dict, format_prompt()
│   ├── inference/       # load_model_and_tokenizer, extract_activations_batch,
│   │                    #   find_token_positions, extract_activations_at_positions
│   └── utils/           # setup_logging
├── scripts/
│   ├── generate_and_extract.py     # Batched QwQ generation + multi-position extraction
│   ├── generate_questions_llm.py   # Claude API question pair generation (v2)
│   ├── generate_prefix_variants.py # Prefix variant expansion for Group A (v2)
│   ├── validate_dataset.py         # Dataset validation checks (v2)
│   ├── assign_splits.py            # Pre-assigned split generation (v2)
│   ├── train_probes_qa.py          # QA probe training (pair-aware splits, combiner)
│   ├── evaluate_probes_qa.py       # Probe evaluation (per-group, TPR/FPR, combiner)
│   ├── run_evasion_experiment.py   # 5-regime evasion experiment
│   └── build_dashboard.py          # Results dashboard
├── data/concepts/trees_qa/         # v1: 300 contrastive QA pairs (15 batches)
└── data/concepts/trees_qa_v2/      # v2: 975 pairs (5 groups + prefix variants)
```

## Key Architecture

- **Model**: QwQ-32B, 64 layers, hidden_dim=5120, 4-bit NF4 quantization
- **Probes**: LinearProbe (nn.Linear -> squeeze, returns pre-sigmoid logits), trained with BCEWithLogitsLoss
- **Activation extraction**: Forward hooks on `model.model.layers[i]`, cast to `.float()` at capture time
- **Multi-position scoring**: Extracts at last_token, end_of_reasoning, first_answer_sentence_end, answer_mean_pool
- **Probe training**: 4 seeds per layer, StandardScaler normalization, early stopping patience=10

## Conventions

- `scripts/` is NOT a Python package — uses `sys.path.insert` to import from `src/`
- All paths resolve against PROJECT_ROOT, not CWD
- Use `uv run python` on RunPod (system torch is too old)
- Commands should be single-line (no backslash continuations)

## Common Tasks

```bash
# === v1 pipeline (300 pairs, all 64 layers) ===
uv run python scripts/generate_and_extract.py --config configs/experiments/qa_probe_training.yaml --batch-size 10
uv run python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training.yaml
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training.yaml

# === v2 pipeline (975 pairs, 15 layers, combiner) ===
# Phase A: Data prep (local, no GPU)
python scripts/generate_questions_llm.py --group B --all-batches
python scripts/generate_questions_llm.py --group C --all-batches
python scripts/generate_questions_llm.py --group D --all-batches
python scripts/generate_questions_llm.py --group E --all-batches
python scripts/generate_prefix_variants.py
python scripts/validate_dataset.py
python scripts/assign_splits.py

# Phase C: GPU execution (RunPod)
uv run python scripts/generate_and_extract.py --config configs/experiments/qa_probe_training_v2.yaml --batch-size 10
uv run python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training_v2.yaml
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training_v2.yaml

# === Evasion experiment ===
uv run python scripts/run_evasion_experiment.py --config configs/experiments/evasion_affordances.yaml --output-dir data/outputs/evasion --batch-size 5
uv run python scripts/run_evasion_experiment.py --config configs/experiments/evasion_affordances.yaml --output-dir data/outputs/evasion --regime technical_disclosure
```

## Evasion Regimes

1. **baseline** — No probe awareness
2. **awareness_only** — Told about monitoring, no technical details
3. **technical_disclosure** — Full probe mechanics explained
4. **technical_extended** — Same + 4x reasoning budget (8192 tokens)
5. **iterative_feedback** — 3 turns with real probe score feedback

## Dependencies

See `pyproject.toml`. Key: torch>=2.1, transformers>=4.45, bitsandbytes, accelerate, scikit-learn, pyyaml, tqdm
