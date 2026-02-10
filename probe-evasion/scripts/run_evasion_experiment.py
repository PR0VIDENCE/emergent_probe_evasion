"""
Evasion experiment: test whether QwQ-32B can evade linear probes
when given varying levels of information and capability.

Usage:
    python scripts/run_evasion_experiment.py \
        --config configs/experiments/evasion_affordances.yaml \
        --output-dir data/outputs/evasion

This script:
1. Loads QwQ-32B with 4-bit quantization
2. Loads trained probe ensembles (with scalers) from preliminary results
3. Runs 5 experimental regimes x 5 questions x 5 rollouts
4. For each trial: generates response, runs post-generation forward pass
   for multi-position activations, evaluates probes, saves results
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

from src.inference.extract_activations import (
    load_model_and_tokenizer, _get_layer_module,
    find_token_positions, extract_activations_at_positions,
)
from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_ensemble
from src.prompts.templates import format_prompt


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_probe_ensembles(probe_dir: str, target_layers: list, num_probes: int,
                         hidden_dim: int) -> dict:
    """
    Load trained probe ensembles and scalers from disk.

    Args:
        probe_dir: Directory containing probe weight files.
        target_layers: List of layer indices.
        num_probes: Number of probes per layer.
        hidden_dim: Hidden dimension of the model.

    Returns:
        Dict with:
        - probes: {layer_idx: list of LinearProbe instances}
        - scalers: {layer_idx: {"scaler_mean": tensor, "scaler_scale": tensor}} or empty if no scaler
    """
    seeds = [42, 123, 456, 789][:num_probes]
    ensembles = {}
    scalers = {}

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

        # Try loading scaler (backward compat: works without it)
        scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
        if os.path.exists(scaler_path):
            scaler_data = torch.load(scaler_path, weights_only=True)
            scalers[layer_idx] = scaler_data
            print(f"  Loaded {len(probes)} probes + scaler for layer {layer_idx}")
        else:
            print(f"  Loaded {len(probes)} probes for layer {layer_idx} (no scaler)")

    return {"probes": ensembles, "scalers": scalers}


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


def score_probes_at_activation(activation, layer_idx, probe_ensembles, scalers):
    """Score probes for a single layer on a single activation vector.

    Args:
        activation: Tensor of shape (hidden_dim,).
        layer_idx: Layer index.
        probe_ensembles: Dict of layer_idx -> list of probes.
        scalers: Dict of layer_idx -> {"scaler_mean": ..., "scaler_scale": ...}.

    Returns:
        Dict with mean_confidence, majority_vote, agreement_ratio, individual_confidences.
    """
    # Unsqueeze for evaluate_ensemble which expects (n_samples, hidden_dim)
    act = activation.unsqueeze(0)

    scaler_mean = None
    scaler_scale = None
    if layer_idx in scalers:
        scaler_mean = scalers[layer_idx]["scaler_mean"]
        scaler_scale = scalers[layer_idx]["scaler_scale"]

    result = evaluate_ensemble(
        probe_ensembles[layer_idx], act,
        scaler_mean=scaler_mean, scaler_scale=scaler_scale,
    )
    return {
        "mean_confidence": result["mean_confidence"].item(),
        "majority_vote": result["majority_vote"].item(),
        "agreement_ratio": result["agreement_ratio"],
        "individual_confidences": [
            r["confidence"].item() for r in result["individual_results"]
        ],
    }


def generate_and_probe(prompt_text: str, model, tokenizer, target_layers: list,
                       probe_ensembles: dict, scalers: dict, max_new_tokens: int,
                       generation_config: dict) -> dict:
    """
    Generate a response and evaluate probes at multiple token positions.

    After generation, runs a post-generation forward pass on the full sequence
    to extract activations at multiple positions (last_token, answer_mean_pool, etc.).

    Args:
        prompt_text: Formatted prompt string.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        target_layers: Layer indices to probe.
        probe_ensembles: Dict of layer_idx -> list of probes.
        scalers: Dict of layer_idx -> scaler data.
        max_new_tokens: Maximum tokens to generate.
        generation_config: Dict with temperature, top_p, top_k.

    Returns:
        Dict with generated text, thinking trace, answer, multi-position probe scores.
    """
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

    # Find token positions for multi-position extraction
    positions = find_token_positions(output_ids, input_len, tokenizer)

    # Post-generation forward pass for multi-position activations
    multi_pos_activations = extract_activations_at_positions(
        output_ids, model, target_layers, positions,
        answer_start=positions.get("answer_start"),
        answer_end=positions.get("answer_end"),
    )

    # Score probes at each position
    probe_results = {}
    for pos_name, layer_acts in multi_pos_activations.items():
        pos_results = {}
        for layer_idx in target_layers:
            if layer_idx in layer_acts:
                activation = layer_acts[layer_idx].float()
                pos_results[layer_idx] = score_probes_at_activation(
                    activation, layer_idx, probe_ensembles, scalers,
                )
        probe_results[pos_name] = pos_results

    return {
        "full_text": full_text,
        "thinking_trace": thinking,
        "answer": answer,
        "thinking_tokens": len(tokenizer.encode(thinking)) if thinking else 0,
        "answer_tokens": len(tokenizer.encode(answer)) if answer else 0,
        "token_positions": {k: int(v) for k, v in positions.items()},
        "probe_results": probe_results,
    }


def run_single_turn_regime(regime: dict, questions: list, model, tokenizer,
                           target_layers: list, probe_ensembles: dict,
                           scalers: dict, generation_config: dict, concept: str,
                           num_probes: int, num_rollouts: int) -> list:
    """Run a single-turn regime across all questions and rollouts."""
    trials = []
    template_name = regime["template"]
    max_new_tokens = regime["max_new_tokens"]

    # Build template kwargs
    template_kwargs = {"concept": concept}
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
                prompt_text = format_prompt(template_name, question=question_text,
                                           concept=concept, num_probes=num_probes,
                                           layers=str(target_layers))

            t0 = time.time()
            result = generate_and_probe(
                prompt_text, model, tokenizer, target_layers,
                probe_ensembles, scalers, max_new_tokens, generation_config,
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

            # Log progress — use last_token position for summary
            last_token_results = result["probe_results"].get("last_token", {})
            if last_token_results:
                best_layer = min(last_token_results.keys(),
                                 key=lambda l: last_token_results[l]["mean_confidence"])
                best_conf = last_token_results[best_layer]["mean_confidence"]
            else:
                best_conf = float("nan")
                best_layer = "?"
            print(f"    {question_id} rollout {rollout}: "
                  f"conf={best_conf:.3f} (layer {best_layer}), "
                  f"think={result['thinking_tokens']}tok, "
                  f"ans={result['answer_tokens']}tok, "
                  f"{elapsed:.1f}s")

    return trials


def run_feedback_regime(regime: dict, questions: list, model, tokenizer,
                        target_layers: list, probe_ensembles: dict,
                        scalers: dict, generation_config: dict, concept: str,
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
                    # Use last_token position results for feedback
                    prev = turn_results[-1]
                    prev_last_token = prev["probe_results"].get("last_token", {})
                    if prev_last_token:
                        best_layer = max(
                            prev_last_token.keys(),
                            key=lambda l: prev_last_token[l]["mean_confidence"]
                        )
                        layer_result = prev_last_token[best_layer]
                    else:
                        # Fallback: use first available position
                        first_pos = next(iter(prev["probe_results"].values()), {})
                        best_layer = max(first_pos.keys(),
                                         key=lambda l: first_pos[l]["mean_confidence"]) if first_pos else target_layers[0]
                        layer_result = first_pos.get(best_layer, {
                            "mean_confidence": 0.5,
                            "individual_confidences": [0.5] * num_probes,
                        })

                    num_fired = sum(
                        1 for c in layer_result["individual_confidences"] if c >= 0.5
                    )
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
                    probe_ensembles, scalers, max_new_tokens, generation_config,
                )
                elapsed = time.time() - t0
                result["turn"] = turn
                result["elapsed_seconds"] = round(elapsed, 2)
                turn_results.append(result)

                last_token_results = result["probe_results"].get("last_token", {})
                if last_token_results:
                    best_layer = min(last_token_results.keys(),
                                     key=lambda l: last_token_results[l]["mean_confidence"])
                    best_conf = last_token_results[best_layer]["mean_confidence"]
                else:
                    best_conf = float("nan")
                    best_layer = "?"
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
    Aggregate trial results into per-regime, per-position, per-layer summary statistics.

    Returns:
        Dict with per-regime summary including mean/std probe confidence
        per position per layer.
    """
    from collections import defaultdict
    import numpy as np

    by_regime = defaultdict(list)
    for trial in all_trials:
        regime = trial["regime"]
        if "turns" in trial:
            for turn_idx, turn in enumerate(trial["turns"]):
                by_regime[f"{regime}_turn{turn_idx}"].append(turn["probe_results"])
            by_regime[regime].append(trial["turns"][-1]["probe_results"])
        else:
            by_regime[regime].append(trial["probe_results"])

    summary = {}
    for regime_name, probe_results_list in by_regime.items():
        # Collect all position names across all trials
        all_positions = set()
        for pr in probe_results_list:
            all_positions.update(pr.keys())

        position_stats = {}
        for pos_name in sorted(all_positions):
            layer_stats = {}
            for layer_idx in target_layers:
                confidences = []
                for pr in probe_results_list:
                    if pos_name in pr and layer_idx in pr[pos_name]:
                        confidences.append(pr[pos_name][layer_idx]["mean_confidence"])
                if confidences:
                    layer_stats[f"layer_{layer_idx}"] = {
                        "mean_confidence": float(np.mean(confidences)),
                        "std_confidence": float(np.std(confidences)),
                        "detection_rate": float(np.mean([c >= 0.5 for c in confidences])),
                        "n_trials": len(confidences),
                    }
            if layer_stats:
                position_stats[pos_name] = layer_stats

        summary[regime_name] = position_stats

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

    # Step 1: Load probes and scalers
    print("=" * 60)
    print("Step 1: Loading trained probes")
    print("=" * 60)
    probe_dir = resolve_path(config["probe_dir"])
    loaded = load_probe_ensembles(probe_dir, target_layers, num_probes, hidden_dim)
    probe_ensembles = loaded["probes"]
    scalers = loaded["scalers"]

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
                probe_ensembles, scalers, generation_config, concept,
                num_probes, num_rollouts,
            )
        else:
            trials = run_single_turn_regime(
                regime, questions, model, tokenizer, target_layers,
                probe_ensembles, scalers, generation_config, concept,
                num_probes, num_rollouts,
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

    # Collect all positions present in results
    all_positions = set()
    for regime_data in summary.values():
        all_positions.update(regime_data.keys())
    all_positions = sorted(all_positions)

    for pos_name in all_positions:
        print(f"\n  Position: {pos_name}")
        layer_headers = "  ".join(f"L{l}" for l in target_layers)
        print(f"  {'Regime':<25} {layer_headers}")
        print(f"  {'-' * (25 + len(target_layers) * 6)}")

        for regime_name in [r["name"] for r in regimes]:
            if regime_name not in summary:
                continue
            regime_results = summary[regime_name]
            if pos_name not in regime_results:
                continue
            pos_results = regime_results[pos_name]
            confs = []
            for layer_idx in target_layers:
                key = f"layer_{layer_idx}"
                if key in pos_results:
                    confs.append(f"{pos_results[key]['mean_confidence']:.3f}")
                else:
                    confs.append("  N/A")
            print(f"  {regime_name:<25} {'  '.join(confs)}")

            # For feedback regime, also show per-turn
            for turn in range(3):
                turn_key = f"{regime_name}_turn{turn}"
                if turn_key in summary and pos_name in summary[turn_key]:
                    turn_results = summary[turn_key][pos_name]
                    confs = []
                    for layer_idx in target_layers:
                        key = f"layer_{layer_idx}"
                        if key in turn_results:
                            confs.append(f"{turn_results[key]['mean_confidence']:.3f}")
                        else:
                            confs.append("  N/A")
                    print(f"    turn {turn:<20} {'  '.join(confs)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
