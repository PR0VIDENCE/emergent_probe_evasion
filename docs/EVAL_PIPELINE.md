# Standardized probe-evasion eval pipeline

A single end-to-end eval that runs from generation through analysis for every
concept and emits one cross-concept dashboard. It consolidates the 6
concept-specific pipelines (trees, disease, library, multiplication, deception,
sycophancy) into one set of stage scripts driven by per-concept config.

## Run it

```bash
# Full run, all 6 concepts (RunPod L40S):
uv run python scripts/run_eval.py --config configs/eval/qwq32b_full.yaml

# Fast smoke (~25 units/concept, universal regimes only):
uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml

# A subset of concepts:
uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml --concepts trees,sycophancy

# No-GPU simulation (verify orchestration / schema / dashboard locally):
uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml --mock
```

Output lands in `data/eval_runs/<run_id>/` (gitignored). `<run_id>` is the
config's `run_id` (default `qwq32b_fast` / `qwq32b_full`). Open
`data/eval_runs/<run_id>/dashboard.html` in any browser.

The run is **resumable per concept and per stage**: re-running the same command
skips completed work (logged as "skip"). Killing mid-concept and re-running
resumes from the last completed stage. A concept whose probe fails validation is
logged + skipped at evasion and never crashes the run.

## The 7 stages

| Stage | Script | Writes |
|---|---|---|
| 1. Generate probe data | `eval_stage_generate.py` | `probe_training/rollouts.jsonl` |
| 2. Label | `eval_stage_label.py --phase probe_training` | `rollouts_labeled.jsonl` |
| 3. Extract activations | `eval_stage_extract.py --phase probe_training` | `activations/*.pt` |
| 4. Train + validate probes | `eval_stage_train_probes.py` | `probes/`, `validation.json` |
| 5. Run evasion | `eval_stage_evasion.py` (+ label/extract `--phase evasion`) | `evasion/rollouts*.jsonl`, `activations/` |
| 6. Score evasion | `eval_stage_score.py` | `evasion/scores.jsonl`, `regime_report.md` |
| 7. Build dashboard | `eval_stage_dashboard.py` | `dashboard.html`, `summary.json` |

`run_eval.py` chains all of them, loading the model once and sharing it across
concepts. Each stage is independently runnable for debugging.

## Output layout

```
data/eval_runs/<run_id>/
  config.yaml                 # snapshot of the eval-run config
  status.json                 # global progress + per-concept validation verdict
  dashboard.html              # self-contained cross-concept dashboard
  summary.json                # machine-readable cross-concept summary
  concepts/<concept>/
    status.json               # per-stage status (resumability)
    errors.jsonl              # per-rollout errors with context
    probe_training/
      rollouts.jsonl  rollouts_labeled.jsonl
      activations/<rollout_id>.pt   extraction_log.jsonl
    probes/
      <position>/layer{N}_seed{S}.pt + layer{N}_scaler.pt
      combiner.json  results.json  validation.json
    evasion/
      rollouts.jsonl  rollouts_labeled.jsonl
      activations/<rollout_id>.pt
      scores.jsonl  scores_summary.json  regime_report.md
```

## The rollout schema

Every JSONL rollout (probe training or evasion) is uniform:

```jsonc
{
  "concept": "sycophancy", "phase": "probe_training|evasion",
  "regime": "baseline" | null, "rollout_id": "<concept>_<phase>_<unit>_<regime>",
  "contrastive_id": "...", "model_id": "qwq-32b",
  "input": {"system_prompt": "...", "user_text": "..."},
  "output": {"thinking": "...", "response": "...", "n_gen_tokens": 0, "truncated": false},
  "labels": {"judge_label": "...", "judge_reason": "...", "judge_model": "..."},
  "probe_scores": {"answer_mean_pool_combined": 12.3},
  "metadata": { /* concept-specific: unit_id, source, scenario, true_answer, ... */ }
}
```

## Canonical probe score (§2.3)

1. Pick the concept's `canonical_position` (default `answer_mean_pool`;
   multiplication uses `last_token`).
2. For each `(layer, seed)` at that position, standardize with the saved scaler
   and compute the pre-sigmoid logit `w·x + b`.
3. Average across all layers and seeds → one scalar per rollout.

Implemented in `src/eval/probe_scoring.py`. The probe-fire thresholds used for
TPR@FPR are derived from the score distribution of the probe's **training
negatives**, so cross-concept TPR numbers are comparable.

## Validation gate (§2.5 / §2.8)

After training, the canonical combined score is computed on the validation split
and compared to `probe.validation_threshold` (default 0.95). If it fails, the
concept is marked failed in `status.json`, evasion is skipped, and the dashboard
renders a "PROBE FAILED VALIDATION" banner. The run continues to the next concept.

To tune the threshold for one concept without editing its shared config, use a
per-run override in the eval-run YAML:

```yaml
concept_overrides:
  multiplication:
    probe: {validation_threshold: 0.90}
```

## Architecture (L2 standardization)

Stage scripts are thin; all logic lives in `src/eval/`:

- `common.py` — config/path/status/schema helpers, `ModelContext` (loads the
  model once), per-run concept overrides.
- `data_handlers.py` — 4 pluggable handlers selected by `data.handler`:
  `contrastive_qa_pairs`, `prewritten_pairs`, `sycophancy_sharma`,
  `multiplication_programmatic`. Each turns a fixed pool into rollout requests.
- `judging.py` — labeling handlers (`none`, `numeric_check`,
  `llm_judge_sycophancy`, `llm_judge_deception`) + the probe-contrast resolver
  (`construction_label` vs `sycophancy`).
- `extraction.py` — forward-pass extraction with the re-extraction
  `context_handler` (`identity` / `neutral_system` / `scenario_system`).
- `generation.py` — batched generation with size-1 fallback.
- `pipeline.py` — shared resumable generation + extraction runners.
- `probes_train.py` — Apollo-style ensemble training + validation.
- `scoring.py` — canonical scoring + per-regime report (+ multiplication
  canonical-cell aggregation).
- `mock.py` — no-GPU simulation backend (used by `--mock`).

## Adding a new concept

1. Ensure a data handler exists for its data shape (the 4 above cover
   contrastive QA, prewritten pairs, Sharma sycophancy, programmatic
   multiplication). If genuinely new, add a function to `data_handlers.py` and
   register it in `_TRAINING_HANDLERS` / `_EVASION_HANDLERS`.
2. Write `configs/eval/concepts/<concept>.yaml` (copy the closest existing one).
3. Add the concept to the eval-run YAML's `concepts:` list. No pipeline-code
   changes needed.

## Adding a new model

1. Write `configs/models/<model>.yaml` matching `qwq_32b.yaml`'s shape. Set
   `has_reasoning_trace: false` if it does not emit `<think>...</think>`.
2. Point the eval-run YAML's `model_config:` at it. No pipeline-code changes.

## Concept-specific notes

- **Multiplication** sweeps `(regime × digits × phrasing)` cells at evasion
  (`eval_axes`); the cross-concept summary uses the `canonical_cell`
  (d=4, pure_numerical), and the per-concept drill-down shows the full
  digits×phrasing heatmap.
- **Sycophancy** trains on an 80/20 neutral/sycophancy_extreme system-prompt
  mix, re-extracts under the neutral prompt (`context_handler: neutral_system`),
  and uses the SyA-vs-(GA+pushback) contrast (`probe.contrast: sycophancy`).
- **Deception** uses Apollo prewritten completions for probe training (no
  generation; construction labels) and free-form generation judged by
  `llm_judge_deception` at evasion; re-extraction strips the evasion addendum
  via `context_handler: scenario_system`.
- LLM judges (sycophancy, deception evasion) need `OPENROUTER_API_KEY`.
