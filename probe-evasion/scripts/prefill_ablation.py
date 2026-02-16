"""
Pre-fill ablation: determine what drives the evasion effect.

Constructs synthetic sequences by mixing prompts, reasoning traces,
and answers from two regimes (e.g., baseline vs technical_disclosure),
then runs forward passes (no generation) to score probes.

This isolates whether the probe score change comes from:
  - The evasion PROMPT (instructions about probes)
  - The evasion REASONING (model's strategy in <think>)
  - The evasion ANSWER (potentially altered output text)

8 swap conditions (2^3 factorial over regime A vs B):
  AAA: all from regime A                     (control)
  AAB: A prompt + A reasoning + B answer     (answer only from B)
  ABA: A prompt + B reasoning + A answer     (reasoning only from B)
  BAA: B prompt + A reasoning + A answer     (prompt only from B)
  ABB: A prompt + B reasoning + B answer     (reasoning+answer from B)
  BAB: B prompt + A reasoning + B answer     (prompt+answer from B)
  BBA: B prompt + B reasoning + A answer     (prompt+reasoning from B)
  BBB: all from regime B                     (treatment)

3 ablation conditions:
  no_reasoning_B: B prompt + empty think + B answer
  no_prompt_B:    minimal prompt + B reasoning + B answer
  answer_only_B:  minimal prompt + empty think + B answer

Usage:
    uv run python scripts/prefill_ablation.py --config configs/experiments/evasion_affordances.yaml --data-dir data/outputs/evasion_v2 --output-dir data/outputs/prefill_ablation --regime-a baseline --regime-b technical_disclosure
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.extract_activations import (
    load_model_and_tokenizer,
    find_token_positions,
    extract_activations_at_positions,
)
from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_ensemble
from src.prompts.templates import format_prompt

from run_evasion_experiment import (
    load_probe_ensembles,
    score_probes_at_activation,
)


# ─── Condition definitions ────────────────────────────────────────────
# "A" = regime_a (control), "B" = regime_b (treatment)
# Each condition specifies which regime provides each component.

SWAP_CONDITIONS = {
    "AAA": {"prompt": "A", "reasoning": "A", "answer": "A"},
    "AAB": {"prompt": "A", "reasoning": "A", "answer": "B"},
    "ABA": {"prompt": "A", "reasoning": "B", "answer": "A"},
    "BAA": {"prompt": "B", "reasoning": "A", "answer": "A"},
    "ABB": {"prompt": "A", "reasoning": "B", "answer": "B"},
    "BAB": {"prompt": "B", "reasoning": "A", "answer": "B"},
    "BBA": {"prompt": "B", "reasoning": "B", "answer": "A"},
    "BBB": {"prompt": "B", "reasoning": "B", "answer": "B"},
}

ABLATION_CONDITIONS = {
    "no_reasoning_B": {"prompt": "B", "reasoning": "none", "answer": "B"},
    "no_prompt_B": {"prompt": "minimal", "reasoning": "B", "answer": "B"},
    "answer_only_B": {"prompt": "minimal", "reasoning": "none", "answer": "B"},
}

MINIMAL_PROMPT = "Please respond to the following."


# ─── Helpers ──────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p):
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_trial_data(data_dir, regime_name):
    """Load trials from {data_dir}/trials/{regime_name}.json."""
    path = os.path.join(data_dir, "trials", f"{regime_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_trial_index(trials):
    """Index trials by (question_id, rollout) -> trial dict."""
    index = {}
    for trial in trials:
        key = (trial["question_id"], trial["rollout"])
        index[key] = trial
    return index


def mean_probe_confidence(probe_results, position):
    """Compute mean confidence across all layers for a position."""
    pos_data = probe_results.get(position, {})
    if not pos_data:
        return None
    return sum(v["mean_confidence"] for v in pos_data.values()) / len(pos_data)


def find_regime_config(config, regime_name):
    """Look up a regime's config dict from the experiment config."""
    all_regimes = config.get("regimes", []) + config.get("control_regimes", [])
    for r in all_regimes:
        if r["name"] == regime_name:
            return r
    return None


def build_prompt_for_regime(regime_cfg, question_text, concept, target_layers,
                            num_probes):
    """Build the formatted prompt text for a given regime and question."""
    template_name = regime_cfg.get("template", "evasion_baseline")
    extra_kwargs = {}
    if regime_cfg.get("wrong_concept"):
        extra_kwargs["wrong_concept"] = regime_cfg["wrong_concept"]
    if regime_cfg.get("forbidden_word"):
        extra_kwargs["forbidden_word"] = regime_cfg["forbidden_word"]

    try:
        return format_prompt(
            template_name, question=question_text,
            concept=concept, num_probes=num_probes,
            layers=str(target_layers),
            **extra_kwargs,
        )
    except KeyError:
        return format_prompt(
            template_name, question=question_text,
            concept=concept,
            **extra_kwargs,
        )


# ─── Sequence assembly ────────────────────────────────────────────────

def assemble_prefill_sequence(
    prompt_text: str,
    thinking_trace: Optional[str],
    answer_text: str,
    tokenizer,
) -> Tuple[torch.Tensor, int]:
    """
    Construct a pre-filled chat sequence and return (output_ids, input_len).

    Builds the full token sequence as if the model had generated it:
        <chat_template with prompt>...<think>\\n
        {thinking_trace}\\n</think>\\n{answer_text}

    For no-reasoning conditions (thinking_trace=None), uses empty think block:
        <chat_template with prompt>...<think>\\n
        </think>\\n{answer_text}
    """
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_tokens = tokenizer(
        chat_prompt, return_tensors="pt", truncation=True, max_length=4096
    )
    input_len = prompt_tokens["input_ids"].shape[1]

    if thinking_trace is not None and thinking_trace.strip():
        generated_text = f"{thinking_trace}\n</think>\n{answer_text}"
    else:
        generated_text = f"</think>\n{answer_text}"

    full_text = chat_prompt + generated_text
    full_tokens = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=32768
    )
    output_ids = full_tokens["input_ids"]

    return output_ids, input_len


def run_prefill_probe(
    output_ids: torch.Tensor,
    input_len: int,
    model,
    tokenizer,
    target_layers: list,
    probe_ensembles: dict,
    scalers: dict,
    per_position: bool,
) -> dict:
    """Run forward pass on pre-filled sequence and score probes."""
    output_ids = output_ids.to(model.device)

    positions = find_token_positions(output_ids, input_len, tokenizer)

    activations = extract_activations_at_positions(
        output_ids, model, target_layers, positions,
        answer_start=positions.get("answer_start"),
        answer_end=positions.get("answer_end"),
    )

    probe_results = {}
    for pos_name, layer_acts in activations.items():
        pos_results = {}
        for layer_idx in target_layers:
            if layer_idx in layer_acts:
                activation = layer_acts[layer_idx].float()
                pos_results[layer_idx] = score_probes_at_activation(
                    activation, layer_idx, probe_ensembles, scalers,
                    position=pos_name, per_position=per_position,
                )
        probe_results[pos_name] = pos_results

    del activations, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "token_positions": {k: int(v) for k, v in positions.items()},
        "probe_results": probe_results,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-fill ablation: isolate what drives the evasion effect"
    )
    parser.add_argument("--config", required=True,
                        help="Evasion experiment config YAML")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with trial data (trials/*.json)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for ablation results")
    parser.add_argument("--regime-a", default="baseline",
                        help="Control regime (default: baseline)")
    parser.add_argument("--regime-b", default="technical_disclosure",
                        help="Treatment regime (default: technical_disclosure)")
    parser.add_argument("--num-rollouts", type=int, default=None,
                        help="Rollouts per question (default: all available)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip presence/absence ablation conditions")
    parser.add_argument("--skip-swap", action="store_true",
                        help="Skip 2^3 factorial swap conditions")
    args = parser.parse_args()

    # ─── Load config ───
    config = load_config(resolve_path(args.config))
    model_config = load_config(resolve_path(config["model_config"]))

    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))
    num_probes = config["num_probes_per_layer"]
    hidden_dim = model_config["hidden_dim"]
    concept = config["concept"]

    questions = {q["id"]: q["text"] for q in config["questions"]}
    question_ids = [q["id"] for q in config["questions"]]

    # Look up regime configs for prompt reconstruction
    regime_a_cfg = find_regime_config(config, args.regime_a)
    regime_b_cfg = find_regime_config(config, args.regime_b)
    if regime_a_cfg is None:
        print(f"ERROR: Regime '{args.regime_a}' not found in config")
        sys.exit(1)
    if regime_b_cfg is None:
        print(f"ERROR: Regime '{args.regime_b}' not found in config")
        sys.exit(1)

    print(f"Regime A (control):   {args.regime_a}")
    print(f"Regime B (treatment): {args.regime_b}")
    print(f"Questions: {len(question_ids)}")

    # ─── Load probes ───
    print("\nLoading probes...")
    probe_dir = resolve_path(config["probe_dir"])
    probe_positions = config.get("probe_positions", None)
    loaded = load_probe_ensembles(
        probe_dir, target_layers, num_probes, hidden_dim,
        positions=probe_positions,
    )
    probe_ensembles = loaded["probes"]
    scalers = loaded["scalers"]
    per_position = loaded["per_position"]

    # ─── Load model ───
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ─── Load trial data for both regimes ───
    data_dir = resolve_path(args.data_dir)

    print(f"\nLoading regime A data ({args.regime_a})...")
    trials_a = load_trial_data(data_dir, args.regime_a)
    if trials_a is None:
        print(f"ERROR: No data at {data_dir}/trials/{args.regime_a}.json")
        sys.exit(1)
    index_a = build_trial_index(trials_a)
    print(f"  Loaded {len(trials_a)} trials")

    print(f"Loading regime B data ({args.regime_b})...")
    trials_b = load_trial_data(data_dir, args.regime_b)
    if trials_b is None:
        print(f"ERROR: No data at {data_dir}/trials/{args.regime_b}.json")
        sys.exit(1)
    index_b = build_trial_index(trials_b)
    print(f"  Loaded {len(trials_b)} trials")

    # ─── Determine rollouts ───
    available_rollouts = set()
    for (qid, rollout) in index_a:
        if qid in questions and (qid, rollout) in index_b:
            available_rollouts.add(rollout)
    if not available_rollouts:
        print("ERROR: No overlapping (question, rollout) pairs between regimes")
        sys.exit(1)
    max_rollouts = min(args.num_rollouts or 999, max(available_rollouts) + 1)
    print(f"\nUsing {max_rollouts} rollouts per question")

    # ─── Output setup ───
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ─── Collect conditions ───
    conditions = {}
    if not args.skip_swap:
        conditions.update(SWAP_CONDITIONS)
    if not args.skip_ablation:
        conditions.update(ABLATION_CONDITIONS)

    if not conditions:
        print("ERROR: No conditions selected")
        sys.exit(1)

    total_trials = len(conditions) * len(question_ids) * max_rollouts
    print(f"\n{len(conditions)} conditions x {len(question_ids)} questions "
          f"x {max_rollouts} rollouts = {total_trials} forward passes")
    print(f"Estimated time: ~{total_trials * 2 / 60:.0f} minutes\n")

    # ─── Run ablation ───
    all_results = []
    trial_num = 0
    skipped = 0
    t_start = time.time()

    for cond_name, cond in conditions.items():
        print(f"{'═' * 60}")
        print(f"  {cond_name}: prompt={cond['prompt']}, "
              f"reasoning={cond['reasoning']}, answer={cond['answer']}")
        print(f"{'═' * 60}")

        for qid in question_ids:
            for rollout in range(max_rollouts):
                trial_num += 1

                trial_a = index_a.get((qid, rollout))
                trial_b = index_b.get((qid, rollout))

                needs_a = cond["prompt"] == "A" or cond["reasoning"] == "A" or cond["answer"] == "A"
                needs_b = cond["prompt"] == "B" or cond["reasoning"] == "B" or cond["answer"] == "B"
                if (needs_a and trial_a is None) or (needs_b and trial_b is None):
                    skipped += 1
                    continue

                # ── Select prompt (reconstructed from regime template) ──
                if cond["prompt"] == "A":
                    prompt_text = build_prompt_for_regime(
                        regime_a_cfg, questions[qid], concept,
                        target_layers, num_probes)
                elif cond["prompt"] == "B":
                    prompt_text = build_prompt_for_regime(
                        regime_b_cfg, questions[qid], concept,
                        target_layers, num_probes)
                else:  # "minimal"
                    prompt_text = MINIMAL_PROMPT

                # ── Select reasoning trace ──
                if cond["reasoning"] == "A":
                    thinking = trial_a["thinking_trace"]
                elif cond["reasoning"] == "B":
                    thinking = trial_b["thinking_trace"]
                else:  # "none"
                    thinking = None

                # ── Select answer ──
                if cond["answer"] == "A":
                    answer = trial_a["answer"]
                elif cond["answer"] == "B":
                    answer = trial_b["answer"]
                else:
                    answer = ""

                # ── Assemble and probe ──
                t0 = time.time()
                output_ids, input_len = assemble_prefill_sequence(
                    prompt_text, thinking, answer, tokenizer,
                )

                result = run_prefill_probe(
                    output_ids, input_len, model, tokenizer,
                    target_layers, probe_ensembles, scalers, per_position,
                )
                elapsed = time.time() - t0

                amp_mean = mean_probe_confidence(
                    result["probe_results"], "answer_mean_pool")
                eor_mean = mean_probe_confidence(
                    result["probe_results"], "end_of_reasoning")

                result.update({
                    "condition": cond_name,
                    "question_id": qid,
                    "rollout": rollout,
                    "regime_a": args.regime_a,
                    "regime_b": args.regime_b,
                    "prompt_source": cond["prompt"],
                    "reasoning_source": cond["reasoning"],
                    "answer_source": cond["answer"],
                    "seq_len": output_ids.shape[1],
                    "elapsed_seconds": elapsed,
                })
                all_results.append(result)

                # Progress
                elapsed_total = time.time() - t_start
                rate = trial_num / elapsed_total if elapsed_total > 0 else 0
                remaining = (total_trials - trial_num) / rate if rate > 0 else 0
                amp_s = f"AMP={amp_mean:.3f}" if amp_mean is not None else ""
                eor_s = f"EOR={eor_mean:.3f}" if eor_mean is not None else ""
                print(f"  [{trial_num}/{total_trials}] {cond_name} {qid} "
                      f"r{rollout}: {amp_s} {eor_s} "
                      f"({elapsed:.1f}s, ETA {remaining/60:.0f}m)")

                del output_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Condition summary
        cond_trials = [r for r in all_results if r["condition"] == cond_name]
        if cond_trials:
            amp_vals = [mean_probe_confidence(r["probe_results"], "answer_mean_pool")
                        for r in cond_trials]
            eor_vals = [mean_probe_confidence(r["probe_results"], "end_of_reasoning")
                        for r in cond_trials]
            amp_vals = [v for v in amp_vals if v is not None]
            eor_vals = [v for v in eor_vals if v is not None]
            amp_s = f"{sum(amp_vals)/len(amp_vals):.3f}" if amp_vals else "N/A"
            eor_s = f"{sum(eor_vals)/len(eor_vals):.3f}" if eor_vals else "N/A"
            print(f"\n  -> {cond_name}: AMP={amp_s}, EOR={eor_s} "
                  f"(n={len(cond_trials)})\n")

    # ─── Save results ───
    results_path = os.path.join(output_dir, "prefill_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {results_path}")
    if skipped:
        print(f"  ({skipped} trials skipped due to missing source data)")

    # ─── Summary table ───
    a_name = args.regime_a
    b_name = args.regime_b
    print(f"\n{'═' * 78}")
    print(f"  PREFILL ABLATION: {a_name} (A) vs {b_name} (B)")
    print(f"{'═' * 78}")
    print(f"  {'Condition':<16} {'Prompt':<8} {'Reason':<8} "
          f"{'Answer':<8} {'AMP':>7} {'EOR':>7} {'n':>4}")
    print(f"  {'─' * 72}")

    cond_order = list(SWAP_CONDITIONS) + list(ABLATION_CONDITIONS)
    for cond_name in cond_order:
        if cond_name not in conditions:
            continue
        cond = conditions[cond_name]
        cond_trials = [r for r in all_results if r["condition"] == cond_name]
        if not cond_trials:
            continue

        amp_vals = [mean_probe_confidence(r["probe_results"], "answer_mean_pool")
                    for r in cond_trials]
        eor_vals = [mean_probe_confidence(r["probe_results"], "end_of_reasoning")
                    for r in cond_trials]
        amp_vals = [v for v in amp_vals if v is not None]
        eor_vals = [v for v in eor_vals if v is not None]
        amp_s = f"{sum(amp_vals)/len(amp_vals):.3f}" if amp_vals else "  N/A"
        eor_s = f"{sum(eor_vals)/len(eor_vals):.3f}" if eor_vals else "  N/A"

        print(f"  {cond_name:<16} {cond['prompt']:<8} "
              f"{cond['reasoning']:<8} {cond['answer']:<8} "
              f"{amp_s:>7} {eor_s:>7} {len(cond_trials):>4}")

    # ─── Interpretation guide ───
    print(f"\n{'─' * 78}")
    print(f"  Key comparisons (A={a_name}, B={b_name}):")
    print(f"  - AAA vs BBB: full effect size between regimes")
    print(f"  - AAA vs BAA: does B's prompt alone change scores?")
    print(f"  - AAA vs ABA: does B's reasoning alone change scores?")
    print(f"  - AAA vs AAB: does B's answer alone change scores?")
    print(f"  - BBB vs ABB: does B's prompt matter (given B reasoning+answer)?")
    print(f"  - BBB vs BAB: does B's reasoning matter (given B prompt+answer)?")
    print(f"  - BBB vs BBA: does B's answer matter (given B prompt+reasoning)?")
    print(f"  - no_reasoning_B: can B evade without any reasoning?")
    print(f"  - no_prompt_B: can B reasoning+answer evade without B prompt?")

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time / 60:.1f} minutes")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
