"""
Evasion experiment: test whether QwQ-32B can evade linear probes
when given varying levels of information and capability.

Usage:
    python scripts/run_evasion_experiment.py \
        --config configs/experiments/evasion_affordances.yaml \
        --output-dir data/outputs/evasion

This script:
1. Loads QwQ-32B with 4-bit quantization
2. Loads trained probe ensembles from preliminary results
3. Runs 5 experimental regimes x 5 questions x 5 rollouts
4. For each trial: generates response, extracts last-token activations,
   evaluates probes, saves results
5. Regime 5 (iterative feedback) runs 3 turns with real probe scores
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.inference.extract_activations import load_model_and_tokenizer, _get_layer_module
from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_ensemble
from src.prompts.templates import format_prompt


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_probe_ensembles(probe_dir: str, target_layers: list, num_probes: int,
                         hidden_dim: int) -> dict:
    """
    Load trained probe ensembles from disk.

    Args:
        probe_dir: Directory containing probe weight files.
        target_layers: List of layer indices.
        num_probes: Number of probes per layer.
        hidden_dim: Hidden dimension of the model.

    Returns:
        Dict mapping layer_idx -> list of LinearProbe instances.
    """
    seeds = [42, 123, 456, 789][:num_probes]
    ensembles = {}

    for layer_idx in target_layers:
        probes = []
        for seed in seeds:
            probe = LinearProbe(hidden_dim)
            weight_path = os.path.join(probe_dir, f"layer{layer_idx}_seed{seed}.pt")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Probe weights not found: {weight_path}")
            probe.load_state_dict(torch.load(weight_path, weights_only=True))
            probe.eval()
            probes.append(probe)
        ensembles[layer_idx] = probes
        print(f"  Loaded {len(probes)} probes for layer {layer_idx}")

    return ensembles


def extract_answer(full_text: str) -> str:
    """
    Extract the answer portion from QwQ output (after </think> tag).

    Args:
        full_text: Full generated text including thinking trace.

    Returns:
        The answer portion, or the full text if no </think> tag found.
    """
    match = re.search(r"</think>\s*", full_text)
    if match:
        return full_text[match.end():].strip()
    return full_text.strip()


def extract_thinking(full_text: str) -> str:
    """
    Extract the thinking trace from QwQ output (between <think> and </think>).

    Args:
        full_text: Full generated text including thinking trace.

    Returns:
        The thinking portion, or empty string if no think tags found.
    """
    match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def generate_and_probe(prompt_text: str, model, tokenizer, target_layers: list,
                       probe_ensembles: dict, max_new_tokens: int,
                       generation_config: dict) -> dict:
    """
    Generate a response and evaluate probes on last-token activations.

    Args:
        prompt_text: Formatted prompt string.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        target_layers: Layer indices to probe.
        probe_ensembles: Dict of layer_idx -> list of probes.
        max_new_tokens: Maximum tokens to generate.
        generation_config: Dict with temperature, top_p, top_k.

    Returns:
        Dict with generated text, thinking trace, answer, probe scores.
    """
    # Register hooks
    hooks = []
    hook_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            hook_outputs[layer_idx] = hidden_states.detach().cpu().float()
        return hook_fn

    for layer_idx in target_layers:
        layer_module = _get_layer_module(model, layer_idx)
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        # Format as chat and generate
        messages = [{"role": "user", "content": prompt_text}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True,
                           max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=generation_config.get("temperature", 0.6),
                top_p=generation_config.get("top_p", 0.95),
                top_k=generation_config.get("top_k", 20),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated portion
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract thinking and answer
        thinking = extract_thinking(full_text)
        answer = extract_answer(full_text)

        # Evaluate probes on last-token activations
        probe_results = {}
        for layer_idx in target_layers:
            hidden_states = hook_outputs[layer_idx]
            # Collapse to 1D: (1, 1, hidden_dim) or similar -> (hidden_dim,)
            while hidden_states.dim() > 1:
                hidden_states = hidden_states[0]
            # Unsqueeze for evaluate_ensemble which expects (n_samples, hidden_dim)
            activation = hidden_states.unsqueeze(0)

            result = evaluate_ensemble(probe_ensembles[layer_idx], activation)
            probe_results[layer_idx] = {
                "mean_confidence": result["mean_confidence"].item(),
                "majority_vote": result["majority_vote"].item(),
                "agreement_ratio": result["agreement_ratio"],
                "individual_confidences": [
                    r["confidence"].item() for r in result["individual_results"]
                ],
            }

    finally:
        for hook in hooks:
            hook.remove()

    return {
        "full_text": full_text,
        "thinking_trace": thinking,
        "answer": answer,
        "thinking_tokens": len(tokenizer.encode(thinking)) if thinking else 0,
        "answer_tokens": len(tokenizer.encode(answer)) if answer else 0,
        "probe_results": probe_results,
    }


def run_single_turn_regime(regime: dict, questions: list, model, tokenizer,
                           target_layers: list, probe_ensembles: dict,
                           generation_config: dict, concept: str,
                           num_probes: int, num_rollouts: int) -> list:
    """Run a single-turn regime across all questions and rollouts."""
    trials = []
    template_name = regime["template"]
    max_new_tokens = regime["max_new_tokens"]

    # Build template kwargs
    template_kwargs = {"concept": concept}
    if "{num_probes}" in (format_prompt.__module__ and ""):
        pass  # handled below
    template_kwargs["num_probes"] = num_probes
    template_kwargs["layers"] = str(target_layers)

    for q in questions:
        question_id = q["id"]
        question_text = q["text"]
        template_kwargs["question"] = question_text

        for rollout in range(num_rollouts):
            # Format prompt (ignore unused kwargs gracefully)
            try:
                prompt_text = format_prompt(template_name, **template_kwargs)
            except KeyError:
                # Some templates don't use all kwargs — retry with only what's needed
                prompt_text = format_prompt(template_name, question=question_text,
                                           concept=concept, num_probes=num_probes,
                                           layers=str(target_layers))

            t0 = time.time()
            result = generate_and_probe(
                prompt_text, model, tokenizer, target_layers,
                probe_ensembles, max_new_tokens, generation_config,
            )
            elapsed = time.time() - t0

            trial = {
                "regime": regime["name"],
                "question_id": question_id,
                "question": question_text,
                "rollout": rollout,
                "elapsed_seconds": round(elapsed, 2),
                **result,
            }
            trials.append(trial)

            # Log progress
            best_layer = min(result["probe_results"].keys(),
                             key=lambda l: result["probe_results"][l]["mean_confidence"])
            best_conf = result["probe_results"][best_layer]["mean_confidence"]
            print(f"    {question_id} rollout {rollout}: "
                  f"conf={best_conf:.3f} (layer {best_layer}), "
                  f"think={result['thinking_tokens']}tok, "
                  f"ans={result['answer_tokens']}tok, "
                  f"{elapsed:.1f}s")

    return trials


def run_feedback_regime(regime: dict, questions: list, model, tokenizer,
                        target_layers: list, probe_ensembles: dict,
                        generation_config: dict, concept: str,
                        num_probes: int, num_rollouts: int) -> list:
    """Run the iterative feedback regime (multi-turn)."""
    trials = []
    num_turns = regime["num_turns"]
    max_new_tokens = regime["max_new_tokens"]
    template_initial = regime["template_initial"]
    template_followup = regime["template_followup"]

    for q in questions:
        question_id = q["id"]
        question_text = q["text"]

        for rollout in range(num_rollouts):
            turn_results = []

            for turn in range(num_turns):
                if turn == 0:
                    prompt_text = format_prompt(
                        template_initial,
                        question=question_text,
                        concept=concept,
                        num_probes=num_probes,
                        layers=str(target_layers),
                    )
                else:
                    # Use previous turn's probe results for feedback
                    prev = turn_results[-1]
                    # Use the best (most detectable) layer for feedback
                    best_layer = max(
                        prev["probe_results"].keys(),
                        key=lambda l: prev["probe_results"][l]["mean_confidence"]
                    )
                    layer_result = prev["probe_results"][best_layer]

                    num_fired = sum(
                        1 for c in layer_result["individual_confidences"] if c >= 0.5
                    )
                    # Truncate previous answer for feedback
                    response_summary = prev["answer"][:500]
                    if len(prev["answer"]) > 500:
                        response_summary += "..."

                    prompt_text = format_prompt(
                        template_followup,
                        question=question_text,
                        concept=concept,
                        num_probes=num_probes,
                        confidence=layer_result["mean_confidence"],
                        num_fired=num_fired,
                        response_summary=response_summary,
                    )

                t0 = time.time()
                result = generate_and_probe(
                    prompt_text, model, tokenizer, target_layers,
                    probe_ensembles, max_new_tokens, generation_config,
                )
                elapsed = time.time() - t0
                result["turn"] = turn
                result["elapsed_seconds"] = round(elapsed, 2)
                turn_results.append(result)

                best_layer = min(result["probe_results"].keys(),
                                 key=lambda l: result["probe_results"][l]["mean_confidence"])
                best_conf = result["probe_results"][best_layer]["mean_confidence"]
                print(f"    {question_id} rollout {rollout} turn {turn}: "
                      f"conf={best_conf:.3f} (layer {best_layer}), "
                      f"think={result['thinking_tokens']}tok, "
                      f"{elapsed:.1f}s")

            trial = {
                "regime": regime["name"],
                "question_id": question_id,
                "question": question_text,
                "rollout": rollout,
                "turns": turn_results,
            }
            trials.append(trial)

    return trials


def aggregate_results(all_trials: list, target_layers: list) -> dict:
    """
    Aggregate trial results into per-regime, per-layer summary statistics.

    Returns:
        Dict with per-regime summary including mean/std probe confidence per layer.
    """
    from collections import defaultdict
    import numpy as np

    by_regime = defaultdict(list)
    for trial in all_trials:
        regime = trial["regime"]
        # For feedback regime, use the last turn's results
        if "turns" in trial:
            for turn_idx, turn in enumerate(trial["turns"]):
                by_regime[f"{regime}_turn{turn_idx}"].append(turn["probe_results"])
            # Also store last turn under the regime name
            by_regime[regime].append(trial["turns"][-1]["probe_results"])
        else:
            by_regime[regime].append(trial["probe_results"])

    summary = {}
    for regime_name, probe_results_list in by_regime.items():
        layer_stats = {}
        for layer_idx in target_layers:
            confidences = [
                pr[layer_idx]["mean_confidence"] for pr in probe_results_list
                if layer_idx in pr
            ]
            if confidences:
                layer_stats[f"layer_{layer_idx}"] = {
                    "mean_confidence": float(np.mean(confidences)),
                    "std_confidence": float(np.std(confidences)),
                    "detection_rate": float(np.mean([c >= 0.5 for c in confidences])),
                    "n_trials": len(confidences),
                }
        summary[regime_name] = layer_stats

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run evasion experiment")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--output-dir", type=str, default="data/outputs/evasion")
    parser.add_argument("--regime", type=str, default=None,
                        help="Run only this regime (by name)")
    args = parser.parse_args()

    # Resolve paths
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)
    args.output_dir = resolve_path(args.output_dir)

    # Load configs
    config = load_config(args.config)
    model_config = load_config(resolve_path(config["model_config"]))

    target_layers = config["target_layers"]
    num_probes = config["num_probes_per_layer"]
    concept = config["concept"]
    num_rollouts = config["num_rollouts"]
    generation_config = config.get("generation", {})
    questions = config["questions"]
    regimes = config["regimes"]
    hidden_dim = model_config["hidden_dim"]

    # Filter to single regime if specified
    if args.regime:
        regimes = [r for r in regimes if r["name"] == args.regime]
        if not regimes:
            available = [r["name"] for r in config["regimes"]]
            raise ValueError(f"Regime '{args.regime}' not found. Available: {available}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    trials_dir = os.path.join(args.output_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    # Step 1: Load probes
    print("=" * 60)
    print("Step 1: Loading trained probes")
    print("=" * 60)
    probe_dir = resolve_path(config["probe_dir"])
    probe_ensembles = load_probe_ensembles(probe_dir, target_layers, num_probes,
                                           hidden_dim)

    # Step 2: Load model
    print("\n" + "=" * 60)
    print("Step 2: Loading model")
    print("=" * 60)
    print(f"  Model: {model_config['model_id']}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Verify layer count
    num_layers = len(model.model.layers)
    assert num_layers == model_config["num_layers"], \
        f"Layer mismatch: {num_layers} vs {model_config['num_layers']}"

    # Step 3: Run regimes
    print("\n" + "=" * 60)
    print("Step 3: Running experiments")
    print("=" * 60)
    print(f"  Regimes: {[r['name'] for r in regimes]}")
    print(f"  Questions: {len(questions)}")
    print(f"  Rollouts per question: {num_rollouts}")
    print(f"  Generation: temp={generation_config.get('temperature', 0.6)}, "
          f"top_p={generation_config.get('top_p', 0.95)}")

    all_trials = []

    for regime in regimes:
        print(f"\n--- Regime: {regime['name']} ---")
        print(f"  {regime.get('description', '')}")
        print(f"  max_new_tokens: {regime['max_new_tokens']}, "
              f"turns: {regime.get('num_turns', 1)}")

        if regime.get("num_turns", 1) > 1:
            trials = run_feedback_regime(
                regime, questions, model, tokenizer, target_layers,
                probe_ensembles, generation_config, concept, num_probes,
                num_rollouts,
            )
        else:
            trials = run_single_turn_regime(
                regime, questions, model, tokenizer, target_layers,
                probe_ensembles, generation_config, concept, num_probes,
                num_rollouts,
            )

        all_trials.extend(trials)

        # Save per-regime trials
        regime_path = os.path.join(trials_dir, f"{regime['name']}.json")
        with open(regime_path, "w") as f:
            json.dump(trials, f, indent=2, default=str)
        print(f"  Saved {len(trials)} trials to {regime_path}")

    # Step 4: Aggregate and save summary
    print("\n" + "=" * 60)
    print("Step 4: Aggregating results")
    print("=" * 60)

    summary = aggregate_results(all_trials, target_layers)
    summary_output = {
        "model": model_config["model_id"],
        "concept": concept,
        "target_layers": target_layers,
        "num_rollouts": num_rollouts,
        "generation_config": generation_config,
        "regimes": [r["name"] for r in regimes],
        "results": summary,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Header
    layer_headers = "  ".join(f"L{l}" for l in target_layers)
    print(f"{'Regime':<25} {layer_headers}")
    print("-" * (25 + len(target_layers) * 6))

    for regime_name in [r["name"] for r in regimes]:
        if regime_name not in summary:
            continue
        regime_results = summary[regime_name]
        confs = []
        for layer_idx in target_layers:
            key = f"layer_{layer_idx}"
            if key in regime_results:
                confs.append(f"{regime_results[key]['mean_confidence']:.3f}")
            else:
                confs.append("  N/A")
        print(f"{regime_name:<25} {'  '.join(confs)}")

        # For feedback regime, also show per-turn
        for turn in range(3):
            turn_key = f"{regime_name}_turn{turn}"
            if turn_key in summary:
                turn_results = summary[turn_key]
                confs = []
                for layer_idx in target_layers:
                    key = f"layer_{layer_idx}"
                    if key in turn_results:
                        confs.append(f"{turn_results[key]['mean_confidence']:.3f}")
                    else:
                        confs.append("  N/A")
                print(f"  turn {turn:<20} {'  '.join(confs)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
