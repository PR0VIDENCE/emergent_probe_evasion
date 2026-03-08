# Probe Evasion Research Project

## Overview

Research project testing whether QwQ-32B can evade linear probes trained to detect the concept "trees" in its internal activations, across 5 experimental regimes with increasing affordances.

## Project Structure

```
probe-evasion/
├── configs/
│   ├── models/          # qwq_32b.yaml, qwen2_5_32b.yaml
│   ├── concepts/        # trees.yaml, library.yaml, currency.yaml, disease.yaml
│   ├── experiments/     # qa_probe_training_{v2,library,currency,disease}.yaml, evasion_affordances.yaml
│   └── probes/          # weighted_combination*.yaml (auto-gen)
├── src/
│   ├── probes/          # LinearProbe architecture, train_probe, evaluate_probe
│   ├── prompts/         # PROMPT_TEMPLATES dict, format_prompt()
│   ├── inference/       # load_model_and_tokenizer, extract_activations_batch,
│   │                    #   find_token_positions, extract_activations_at_positions
│   └── utils/           # setup_logging, concept_config (generic concept support)
├── scripts/             # All scripts accept concept config for generic concepts
│   ├── generate_and_extract.py     # Batched QwQ generation + extraction
│   ├── generate_questions_llm.py   # Claude API question generation (--concept-config)
│   ├── generate_prefix_variants.py # Prefix variant expansion (--concept-config)
│   ├── validate_dataset.py         # Dataset validation (--concept-config)
│   ├── assign_splits.py            # Split generation (--concept-config)
│   ├── train_probes_qa.py          # Probe training (concept-generic)
│   ├── evaluate_probes_qa.py       # Probe evaluation (concept-generic)
│   ├── run_evasion_experiment.py   # 5-regime evasion experiment
│   └── build_dashboard.py          # Results dashboard
├── data/generation_prompt_templates/ # Generic prompt templates for new concepts
├── data/concepts/trees_qa_v2/      # v2 trees: 975 pairs
├── data/concepts/library_qa_v2/    # Library concept data
├── data/concepts/currency_qa_v2/   # Currency concept data
└── data/concepts/disease_qa_v2/    # Disease concept data
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
# === Trees v2 pipeline (backward-compatible) ===
python scripts/generate_prefix_variants.py
python scripts/validate_dataset.py
python scripts/assign_splits.py
uv run python scripts/generate_and_extract.py --config configs/experiments/qa_probe_training_v2.yaml --batch-size 10
uv run python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training_v2.yaml
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training_v2.yaml

# === New concept pipeline (e.g., library) ===
# Data prep (local):
python scripts/generate_prefix_variants.py --concept-config configs/concepts/library.yaml
python scripts/validate_dataset.py --concept-config configs/concepts/library.yaml
python scripts/assign_splits.py --concept-config configs/concepts/library.yaml
# GPU (RunPod):
uv run python scripts/generate_and_extract.py --config configs/experiments/qa_probe_training_library.yaml --batch-size 10
uv run python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training_library.yaml
uv run python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training_library.yaml

# === Evasion experiment ===
uv run python scripts/run_evasion_experiment.py --config configs/experiments/evasion_affordances.yaml --output-dir data/outputs/evasion --batch-size 5
```

## Evasion Regimes

1. **baseline** — No probe awareness
2. **awareness_only** — Told about monitoring, no technical details
3. **technical_disclosure** — Full probe mechanics explained
4. **technical_extended** — Same + 4x reasoning budget (8192 tokens)
5. **iterative_feedback** — 3 turns with real probe score feedback

## Dependencies

See `pyproject.toml`. Key: torch>=2.1, transformers>=4.45, bitsandbytes, accelerate, scikit-learn, pyyaml, tqdm
