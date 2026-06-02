# Probe Evasion Research Project

## Overview

Research project testing whether QwQ-32B can evade linear probes trained to detect the concept "trees" in its internal activations, across 5 experimental regimes with increasing affordances.

## Interaction Style

- When asked for a plan or conceptual discussion, do NOT make code edits unless explicitly asked. Present the plan/explanation first, then ask before implementing.
- When the user asks to work on a new concept or generalize, always base new work on the **proven trees pipeline** (qa_probe_training_v2.yaml / generate_and_extract.py), not untested experimental branches.

## File & Path Handling

- Before referencing file paths, verify they exist using Read or Bash(ls). Never assume local paths match remote (RunPod) paths.
- Data files may be gitignored and unavailable on remote machines. The `deception-detection/` repo is gitignored — use pre-converted files in `data/concepts/` instead.
- Local machine: `/Users/andrewwei/emergent_probe_evasion/`
- RunPod: `/workspace/emergent_probe_evasion/`

## Debugging

- When debugging, identify and fix the root cause before applying patches. If a fix doesn't work on the first attempt, step back and re-examine assumptions rather than trying variations of the same approach.
- Always check config files (especially target_layers, data paths, split files) for consistency before running experiments.

## Project Structure

```
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
│   ├── train_probes_qa.py          # Probe training (concept-generic, --split-file)
│   ├── evaluate_probes_qa.py       # Probe evaluation (--split-file, --probe-dir)
│   ├── run_evasion_experiment.py   # 5-regime evasion experiment
│   ├── extract_prewritten_activations.py  # Apollo-style forward-pass extraction
│   └── plot_midterm_figures.py     # Midterm report figures
├── data/concepts/trees_qa_v2/      # v2 trees: 975 pairs
├── data/concepts/library_qa_v2/    # Library concept data
├── data/concepts/currency_qa_v2/   # Currency concept data
├── data/concepts/disease_qa_v2/    # Disease concept data
├── data/concepts/deception_prewritten/  # Apollo-style roleplaying scenarios
└── data/concepts/deception_ai_liar/     # AI Liar cross-dataset eval
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

## Standardized eval pipeline (NEW — preferred for cross-concept runs)

The end-to-end probe-evasion eval consolidates all 6 concepts (trees, disease,
library, multiplication, deception, sycophancy) into one orchestrated run that
goes generation → labeling → extraction → probe training/validation → evasion →
scoring → dashboard. See `docs/EVAL_PIPELINE.md` for the full reference.

```bash
# Full run (RunPod L40S); fast smoke caps ~25 units/concept:
uv run python scripts/run_eval.py --config configs/eval/qwq32b_full.yaml
uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml
# No-GPU simulation to verify orchestration/schema/dashboard locally:
uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml --mock
```

- Code lives in `src/eval/` (handlers, judging, extraction, scoring, dashboard).
  Stage scripts are `scripts/eval_stage_*.py` + orchestrator `scripts/run_eval.py`.
- Per-concept config: `configs/eval/concepts/<concept>.yaml`. Universal regimes:
  `configs/eval/universal_regimes.yaml`. Eval-run config: `configs/eval/qwq32b_{fast,full}.yaml`.
- Output: `data/eval_runs/<run_id>/` (gitignored, NEVER overwrites existing data dirs).
- Resumable per concept + per stage; a probe that fails the 0.95 val-AUROC gate is
  logged and skipped at evasion without crashing the run.
- The original per-concept scripts below remain valid for single-concept work.

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
