"""
Interactive explorer: compare baseline vs any regime in real time.

Loads model + probes once, then enters an interactive loop where you can:
- Pick a question (from config or type custom)
- Pick a regime
- See generated response + probe scores side by side with baseline

Usage:
    uv run python scripts/interactive_explorer.py --config configs/experiments/evasion_affordances.yaml
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.extract_activations import (
    load_model_and_tokenizer,
    find_token_positions, extract_activations_at_positions,
)
from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_ensemble
from src.prompts.templates import format_prompt, PROMPT_TEMPLATES


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p):
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


# Reuse probe loading from experiment runner
from run_evasion_experiment import (
    load_probe_ensembles,
    generate_and_probe_single,
    generate_filler_text,
)


def format_probe_scores(probe_results, positions=("answer_mean_pool", "end_of_reasoning")):
    """Format probe scores as a compact table string."""
    lines = []
    for pos in positions:
        if pos not in probe_results:
            continue
        pos_data = probe_results[pos]
        pos_label = "AMP" if "mean_pool" in pos else "EOR"
        layer_strs = []
        for layer_idx in sorted(pos_data.keys(), key=lambda x: int(x)):
            conf = pos_data[layer_idx]["mean_confidence"]
            marker = color_conf(conf)
            layer_strs.append(f"  L{layer_idx:>2}: {marker}")
        mean_conf = sum(pos_data[l]["mean_confidence"] for l in pos_data) / len(pos_data) if pos_data else 0
        lines.append(f"  {pos_label} (mean {mean_conf:.3f}):")
        lines.extend(layer_strs)
    return "\n".join(lines)


def color_conf(v):
    """Return a colored confidence string for terminal."""
    bar_len = int(v * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    if v >= 0.85:
        color = "\033[92m"  # green
    elif v >= 0.65:
        color = "\033[93m"  # yellow
    elif v >= 0.45:
        color = "\033[33m"  # dark yellow
    else:
        color = "\033[91m"  # red
    return f"{color}{bar} {v:.3f}\033[0m"


def print_result(label, result, color_code="\033[0m"):
    """Print a generation result with probe scores."""
    print(f"\n{color_code}{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}\033[0m")
    print(f"\n\033[2m  Think: {result['thinking_tokens']} tok | Answer: {result['answer_tokens']} tok\033[0m")
    if result.get("think_truncated"):
        print(f"  \033[91m[THINK TRUNCATED]\033[0m")
    print(f"\n  \033[1mAnswer:\033[0m")
    # Word-wrap answer at ~80 chars
    answer = result["answer"]
    for i in range(0, len(answer), 76):
        print(f"  {answer[i:i+76]}")

    print(f"\n  \033[1mProbe Scores:\033[0m")
    print(format_probe_scores(result["probe_results"]))


def build_prompt(regime, question_text, concept, target_layers, num_probes):
    """Build the prompt text for a given regime and question."""
    template_name = regime.get("template", "evasion_baseline")
    extra_kwargs = {}
    if regime.get("filler_token_count"):
        extra_kwargs["filler_text"] = generate_filler_text(
            regime["filler_token_count"], seed=hash(question_text) & 0xFFFFFFFF
        )
    if regime.get("wrong_concept"):
        extra_kwargs["wrong_concept"] = regime["wrong_concept"]
    if regime.get("forbidden_word"):
        extra_kwargs["forbidden_word"] = regime["forbidden_word"]

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive regime explorer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    model_config = load_config(resolve_path(config["model_config"]))

    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))
    num_probes = config["num_probes_per_layer"]
    concept = config["concept"]
    hidden_dim = model_config["hidden_dim"]
    generation_config = config.get("generation", {})

    # Build regime lookup
    all_regimes = list(config["regimes"]) + config.get("control_regimes", [])
    regime_map = {r["name"]: r for r in all_regimes}
    # Add baseline explicitly
    baseline_regime = {"name": "baseline", "template": "evasion_baseline", "max_new_tokens": 2048}
    regime_map["baseline"] = baseline_regime

    # Build question lookup
    questions = {q["id"]: q["text"] for q in config["questions"]}
    neg_questions = {q["id"]: q["text"] for q in config.get("negative_control_questions", [])}

    # Load probes
    print("Loading probes...")
    probe_dir = resolve_path(config["probe_dir"])
    probe_positions = config.get("probe_positions", None)
    loaded = load_probe_ensembles(probe_dir, target_layers, num_probes, hidden_dim,
                                  positions=probe_positions)
    probe_ensembles = loaded["probes"]
    scalers = loaded["scalers"]
    per_position = loaded["per_position"]

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Cache for baseline results
    baseline_cache = {}

    print("\033[1m" + "=" * 60)
    print("  INTERACTIVE PROBE EVASION EXPLORER")
    print("=" * 60 + "\033[0m")
    print(f"  Model: {model_config['model_id']}")
    print(f"  Probes: {probe_dir}")
    print(f"  Layers: {len(target_layers)}")
    print(f"  Concept: {concept}")
    print()
    print("  Commands:")
    print("    [enter question ID or type custom question]")
    print("    'list'     - show available questions")
    print("    'regimes'  - show available regimes")
    print("    'quit'     - exit")
    print()

    while True:
        try:
            # Step 1: Get question
            print("\033[1m─── Question ───\033[0m")
            q_input = input("  Question (ID, custom text, or 'list'): ").strip()

            if q_input.lower() == "quit":
                break
            if q_input.lower() == "list":
                print("\n  Tree questions:")
                for qid, text in sorted(questions.items(), key=lambda x: int(x[0][1:])):
                    print(f"    {qid}: {text[:70]}")
                print("\n  Negative controls:")
                for qid, text in sorted(neg_questions.items(), key=lambda x: int(x[0][2:])):
                    print(f"    {qid}: {text[:70]}")
                print()
                continue
            if q_input.lower() == "regimes":
                print("\n  Available regimes:")
                for name, r in regime_map.items():
                    desc = r.get("description", "")
                    print(f"    {name:30s} {desc}")
                print()
                continue

            # Resolve question text
            if q_input in questions:
                question_text = questions[q_input]
                question_label = q_input
            elif q_input in neg_questions:
                question_text = neg_questions[q_input]
                question_label = q_input
            else:
                question_text = q_input
                question_label = "custom"

            print(f"  → {question_text[:80]}{'...' if len(question_text) > 80 else ''}")

            # Step 2: Get regime
            print("\n\033[1m─── Regime ───\033[0m")
            regime_names = list(regime_map.keys())
            for i, name in enumerate(regime_names):
                marker = "  "
                print(f"  {i:2d}. {name}")
            r_input = input("  Regime (name or number): ").strip()

            if r_input.isdigit() and int(r_input) < len(regime_names):
                regime_name = regime_names[int(r_input)]
            elif r_input in regime_map:
                regime_name = r_input
            else:
                print(f"  Unknown regime: {r_input}")
                continue

            regime = regime_map[regime_name]
            max_new_tokens = regime.get("max_new_tokens", 2048)
            print(f"  → {regime_name}")

            # Step 3: Generate baseline (use cache if available)
            cache_key = question_label + ":" + question_text[:50]
            if cache_key not in baseline_cache:
                print(f"\n  Generating baseline...", end="", flush=True)
                t0 = time.time()
                baseline_prompt = build_prompt(baseline_regime, question_text, concept, target_layers, num_probes)
                baseline_result = generate_and_probe_single(
                    baseline_prompt, model, tokenizer, target_layers,
                    probe_ensembles, scalers, 2048, generation_config, per_position,
                )
                print(f" done ({time.time() - t0:.1f}s)")
                baseline_cache[cache_key] = baseline_result
            else:
                print(f"\n  Using cached baseline")
                baseline_result = baseline_cache[cache_key]

            # Step 4: Generate regime response
            if regime_name == "baseline":
                regime_result = baseline_result
            else:
                print(f"  Generating {regime_name}...", end="", flush=True)
                t0 = time.time()
                regime_prompt = build_prompt(regime, question_text, concept, target_layers, num_probes)
                regime_result = generate_and_probe_single(
                    regime_prompt, model, tokenizer, target_layers,
                    probe_ensembles, scalers, max_new_tokens, generation_config, per_position,
                )
                print(f" done ({time.time() - t0:.1f}s)")

            # Step 5: Display side by side
            print_result("BASELINE", baseline_result, "\033[94m")
            print_result(regime_name.upper(), regime_result, "\033[93m")

            # Quick comparison
            def mean_score(result, pos):
                pd = result["probe_results"].get(pos, {})
                if not pd:
                    return 0
                return sum(pd[l]["mean_confidence"] for l in pd) / len(pd)

            base_amp = mean_score(baseline_result, "answer_mean_pool")
            base_eor = mean_score(baseline_result, "end_of_reasoning")
            reg_amp = mean_score(regime_result, "answer_mean_pool")
            reg_eor = mean_score(regime_result, "end_of_reasoning")

            print(f"\n\033[1m─── Comparison ───\033[0m")
            print(f"  AMP: baseline {base_amp:.3f} → {regime_name} {reg_amp:.3f} ({(reg_amp - base_amp)*100:+.1f}pp)")
            print(f"  EOR: baseline {base_eor:.3f} → {regime_name} {reg_eor:.3f} ({(reg_eor - base_eor)*100:+.1f}pp)")
            print()

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("Goodbye!")


if __name__ == "__main__":
    main()
