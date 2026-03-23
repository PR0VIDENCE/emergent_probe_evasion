"""
Probe-steered generation: token-level probe penalty during autoregressive decoding.

At each generation step, evaluates top-K candidate tokens by combining
language model log-probability with a probe confidence penalty:

    score = alpha * log_prob(token) - beta * probe_confidence(token)

This provides much richer search than sampling (genetic algorithm) — it can
select low-probability-but-fluent tokens that suppress the probe direction.

Two modes:
  - full: probe penalty throughout entire generation (reasoning + answer)
  - reasoning_only: penalty during <think> only, normal sampling after </think>

After generation, runs standard multi-position scoring for apples-to-apples
comparison with existing evasion experiment results.

Usage:
    uv run python scripts/run_probe_steered_generation.py --config configs/experiments/evasion_affordances.yaml --beta 2.0 --mode full --top-k-candidates 5 --output-dir data/outputs/probe_steered
"""

import argparse
import json
import os
import re
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
    _get_layer_module,
    find_token_positions,
    extract_activations_at_positions,
)
from src.probes.architectures import LinearProbe
from src.utils.logging import setup_logging

# Import post-generation scoring from evasion experiment
from scripts.run_evasion_experiment import (
    load_probe_ensembles,
    process_single_sequence,
    score_probes_at_activation,
    extract_answer,
    extract_thinking,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_steering_probes(
    probe_dir: str, position: str, target_layers: List[int],
    seeds: List[int], hidden_dim: int, device: torch.device,
) -> Dict[int, dict]:
    """Load probes + scalers directly onto GPU for fast per-token scoring.

    Returns:
        {layer_idx: {"probes": [LinearProbe, ...],
                     "scaler_mean": tensor|None,
                     "scaler_scale": tensor|None}}
    """
    result = {}
    pos_dir = os.path.join(probe_dir, position)
    for layer_idx in target_layers:
        probes = []
        for seed in seeds:
            p = LinearProbe(hidden_dim)
            path = os.path.join(pos_dir, f"layer{layer_idx}_seed{seed}.pt")
            p.load_state_dict(torch.load(path, weights_only=True))
            p.eval()
            p.to(device)
            probes.append(p)
        scaler_path = os.path.join(pos_dir, f"layer{layer_idx}_scaler.pt")
        scaler = torch.load(scaler_path, weights_only=True) if os.path.exists(scaler_path) else {}
        sm = scaler.get("scaler_mean")
        ss = scaler.get("scaler_scale")
        result[layer_idx] = {
            "probes": probes,
            "scaler_mean": sm.to(device).float() if sm is not None else None,
            "scaler_scale": ss.to(device).float() if ss is not None else None,
        }
        print(f"  Steering probe layer {layer_idx}: {len(probes)} probes" +
              (" + scaler" if sm is not None else ""))
    return result


def compute_steering_score(
    hidden_states: Dict[int, torch.Tensor],
    layer_probes: Dict[int, dict],
    target_layers: List[int],
) -> float:
    """Compute mean probe confidence from hook-captured hidden states.

    Args:
        hidden_states: {layer_idx: tensor(hidden_dim)} from hooks.
        layer_probes: Loaded steering probes dict.
        target_layers: Which layers to score.

    Returns:
        Mean probe confidence (float, 0-1).
    """
    confidences = []
    for layer_idx in target_layers:
        if layer_idx not in hidden_states:
            continue
        lp = layer_probes[layer_idx]
        act = hidden_states[layer_idx].unsqueeze(0)  # (1, hidden_dim)
        if lp["scaler_mean"] is not None:
            act = (act - lp["scaler_mean"]) / lp["scaler_scale"]
        logits = torch.stack([p(act) for p in lp["probes"]])  # (n_seeds, 1)
        mean_logit = logits.mean()
        conf = torch.sigmoid(mean_logit).item()
        confidences.append(conf)
    return sum(confidences) / len(confidences) if confidences else 0.5


def sample_token(logits: torch.Tensor, temperature: float = 0.6,
                 top_p: float = 0.95, top_k: int = 20) -> int:
    """Standard sampling for non-steered steps (replicates model.generate behavior)."""
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, top_k_indices, top_k_logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits.scatter_(-1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def _precompute_think_end_ids(tokenizer) -> List[int]:
    """Pre-compute the token IDs for '</think>' to detect mode switch."""
    return tokenizer.encode("</think>", add_special_tokens=False)


def _save_cache_state(past_key_values):
    """Save KV cache state for restoration between candidate evaluations.

    DynamicCache.update() does torch.cat to create NEW tensors, so saving
    list references (shallow copy) is sufficient — originals aren't modified.
    """
    if hasattr(past_key_values, "key_cache"):
        # DynamicCache (transformers >= 4.36)
        return {
            "key_cache": list(past_key_values.key_cache),
            "value_cache": list(past_key_values.value_cache),
            "_seen_tokens": past_key_values._seen_tokens,
        }
    else:
        # Legacy tuple-of-tuples format (shouldn't happen with Qwen2)
        return {"tuples": past_key_values}


def _restore_cache_state(past_key_values, saved):
    """Restore KV cache to a previously saved state."""
    if "key_cache" in saved:
        past_key_values.key_cache = list(saved["key_cache"])
        past_key_values.value_cache = list(saved["value_cache"])
        past_key_values._seen_tokens = saved["_seen_tokens"]
    return past_key_values


def probe_steered_generate(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    layer_probes: Dict[int, dict],
    target_layers: List[int],
    mode: str = "full",
    alpha: float = 1.0,
    beta: float = 2.0,
    top_k_candidates: int = 5,
    max_new_tokens: int = 2048,
    generation_config: dict = None,
) -> dict:
    """Probe-aware top-K greedy generation.

    At each step:
    1. Get top-K candidate tokens from current logits
    2. For each candidate, do a single-token forward pass (KV-cached)
    3. Score: alpha * log_prob - beta * probe_confidence
    4. Select best candidate

    Args:
        input_ids: Tokenized prompt (1, seq_len).
        model: Loaded model.
        tokenizer: Tokenizer.
        layer_probes: Steering probes on GPU.
        target_layers: Layers to probe.
        mode: "full" or "reasoning_only".
        alpha: Weight for log-probability term.
        beta: Probe penalty weight.
        top_k_candidates: Number of candidates to evaluate per step (K).
        max_new_tokens: Max tokens to generate.
        generation_config: Dict with temperature, top_p, top_k for sampling.

    Returns:
        Dict with output_ids, probe_trajectory, timing, mode_switch_token, etc.
    """
    if generation_config is None:
        generation_config = {}
    temperature = generation_config.get("temperature", 0.6)
    top_p = generation_config.get("top_p", 0.95)
    sampling_top_k = generation_config.get("top_k", 20)

    device = input_ids.device
    think_end_ids = _precompute_think_end_ids(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    # Register hooks for steering probe layers
    hook_outputs = {}
    hooks = []

    def make_hook(layer_idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Keep on GPU, capture last token only
            hook_outputs[layer_idx] = h[:, -1, :].detach().float()
        return fn

    for layer_idx in target_layers:
        layer_module = _get_layer_module(model, layer_idx)
        hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

    try:
        generated_ids = []
        probe_trajectory = []
        mode_switch_token = None
        in_thinking = True  # QwQ starts in <think> mode
        steering_active = True  # Whether probe penalty is applied

        # Initial forward pass to get first set of logits + KV cache
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_kv_values if hasattr(outputs, "past_kv_values") else outputs.past_key_values
        current_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        t_start = time.time()

        for step in range(max_new_tokens):
            if steering_active and beta > 0:
                # === Steered step: evaluate top-K candidates ===
                log_probs = torch.log_softmax(current_logits / temperature, dim=-1)
                _, top_k_indices = torch.topk(current_logits, min(top_k_candidates, current_logits.size(-1)))

                # Save cache state before evaluating candidates
                # (DynamicCache mutates in-place; torch.cat creates new tensors
                # so list reference copies suffice)
                saved_cache = _save_cache_state(past_key_values)

                best_score = float("-inf")
                best_token = top_k_indices[0].item()
                best_cache_state = None
                best_logits = None
                best_probe_conf = 0.5

                for k_idx in range(top_k_indices.size(0)):
                    # Restore cache to base state before each candidate
                    _restore_cache_state(past_key_values, saved_cache)

                    candidate_id = top_k_indices[k_idx].item()
                    candidate_tensor = torch.tensor([[candidate_id]], device=device)

                    hook_outputs.clear()
                    with torch.no_grad():
                        cand_outputs = model(
                            candidate_tensor,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )

                    # Probe confidence from hooks
                    probe_conf = compute_steering_score(
                        {li: hook_outputs[li][0] for li in target_layers if li in hook_outputs},
                        layer_probes, target_layers,
                    )

                    lp = log_probs[candidate_id].item()
                    score = alpha * lp - beta * probe_conf

                    if score > best_score:
                        best_score = score
                        best_token = candidate_id
                        # Save this candidate's extended cache state
                        best_cache_state = _save_cache_state(past_key_values)
                        best_logits = cand_outputs.logits[0, -1, :]
                        best_probe_conf = probe_conf

                # Restore cache to the winning candidate's state
                if best_cache_state is not None:
                    _restore_cache_state(past_key_values, best_cache_state)

                generated_ids.append(best_token)
                probe_trajectory.append(best_probe_conf)
                current_logits = best_logits

            else:
                # === Non-steered step: standard sampling ===
                token_id = sample_token(current_logits, temperature, top_p, sampling_top_k)
                generated_ids.append(token_id)

                # Forward pass to get next logits + update cache (in-place)
                token_tensor = torch.tensor([[token_id]], device=device)
                hook_outputs.clear()
                with torch.no_grad():
                    outputs = model(
                        token_tensor,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                # past_key_values is mutated in-place by DynamicCache
                current_logits = outputs.logits[0, -1, :]

                # Record probe score even on non-steered steps
                probe_conf = compute_steering_score(
                    {li: hook_outputs[li][0] for li in target_layers if li in hook_outputs},
                    layer_probes, target_layers,
                )
                probe_trajectory.append(probe_conf)

            # Check for EOS
            if generated_ids[-1] == eos_token_id:
                break

            # Check for </think> transition (for reasoning_only mode)
            if in_thinking and len(generated_ids) >= len(think_end_ids):
                recent = generated_ids[-len(think_end_ids):]
                if recent == think_end_ids:
                    in_thinking = False
                    mode_switch_token = len(generated_ids) - 1
                    if mode == "reasoning_only":
                        steering_active = False
                        print(f"      [mode_switch] </think> at token {mode_switch_token}, "
                              f"steering OFF")

            # Progress logging
            if (step + 1) % 100 == 0:
                elapsed = time.time() - t_start
                mean_probe = sum(probe_trajectory[-100:]) / min(100, len(probe_trajectory[-100:]))
                print(f"      Step {step + 1}/{max_new_tokens}: "
                      f"mean_probe_100={mean_probe:.3f}, "
                      f"{'steering' if steering_active else 'sampling'}, "
                      f"{elapsed:.1f}s")

        generation_time = time.time() - t_start

        # Build output_ids tensor
        output_ids = torch.cat([
            input_ids,
            torch.tensor([generated_ids], device=device),
        ], dim=1)

        return {
            "output_ids": output_ids,
            "input_len": input_ids.shape[1],
            "probe_trajectory": probe_trajectory,
            "mode_switch_token": mode_switch_token,
            "generation_time_seconds": round(generation_time, 2),
            "num_tokens": len(generated_ids),
            "steered_tokens": mode_switch_token if mode == "reasoning_only" and mode_switch_token else len(generated_ids),
        }

    finally:
        for hook in hooks:
            hook.remove()


def run_trial(
    question_id: str,
    question_text: str,
    model, tokenizer,
    layer_probes: Dict[int, dict],
    target_layers: List[int],
    probe_ensembles: dict,
    scalers: dict,
    per_position: bool,
    mode: str,
    alpha: float,
    beta: float,
    top_k_candidates: int,
    max_new_tokens: int,
    generation_config: dict,
    rollout: int = 0,
) -> dict:
    """Run a single probe-steered generation trial.

    Formats prompt, runs steered generation, then runs standard post-generation
    scoring for comparison with existing experiments.
    """
    # Format prompt (baseline template — no evasion instructions)
    messages = [{"role": "user", "content": f"Answer the following question.\n\nQuestion: {question_text}\n\nPlease provide your answer:"}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    print(f"    {question_id} rollout {rollout}: generating (beta={beta}, mode={mode}, K={top_k_candidates})...")
    t0 = time.time()

    # Steered generation
    gen_result = probe_steered_generate(
        input_ids, model, tokenizer, layer_probes, target_layers,
        mode=mode, alpha=alpha, beta=beta,
        top_k_candidates=top_k_candidates,
        max_new_tokens=max_new_tokens,
        generation_config=generation_config,
    )

    output_ids = gen_result["output_ids"]
    input_len = gen_result["input_len"]

    # Standard post-generation scoring
    scoring_result = process_single_sequence(
        output_ids, input_len, model, tokenizer,
        target_layers, probe_ensembles, scalers, per_position,
    )

    elapsed = time.time() - t0

    # Log key metrics
    best_conf = 0.0
    best_pos = "?"
    for pos_name, pos_results in scoring_result["probe_results"].items():
        for layer_idx, layer_result in pos_results.items():
            if layer_result["mean_confidence"] > best_conf:
                best_conf = layer_result["mean_confidence"]
                best_pos = pos_name

    mean_traj = sum(gen_result["probe_trajectory"]) / len(gen_result["probe_trajectory"]) if gen_result["probe_trajectory"] else 0
    print(f"    {question_id} rollout {rollout}: "
          f"post_score={best_conf:.3f} ({best_pos}), "
          f"mean_traj={mean_traj:.3f}, "
          f"tokens={gen_result['num_tokens']}, "
          f"{elapsed:.1f}s")

    trial = {
        "question_id": question_id,
        "question": question_text,
        "rollout": rollout,
        "beta": beta,
        "alpha": alpha,
        "mode": mode,
        "top_k_candidates": top_k_candidates,
        "full_text": scoring_result["full_text"],
        "thinking_trace": scoring_result["thinking_trace"],
        "answer": scoring_result["answer"],
        "thinking_tokens": scoring_result["thinking_tokens"],
        "answer_tokens": scoring_result["answer_tokens"],
        "probe_trajectory": gen_result["probe_trajectory"],
        "probe_trajectory_mean": mean_traj,
        "final_probe_results": scoring_result["probe_results"],
        "token_positions": scoring_result["token_positions"],
        "mode_switch_token": gen_result["mode_switch_token"],
        "generation_time_seconds": gen_result["generation_time_seconds"],
        "num_tokens": gen_result["num_tokens"],
        "steered_tokens": gen_result["steered_tokens"],
    }

    # Clean up
    del output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trial


def main():
    parser = argparse.ArgumentParser(description="Probe-steered generation experiment")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML (e.g., evasion_affordances.yaml)")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Probe penalty weight (default: 2.0)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Log-probability weight (default: 1.0)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "reasoning_only"],
                        help="Steering mode (default: full)")
    parser.add_argument("--top-k-candidates", type=int, default=5,
                        help="Branching factor K (default: 5)")
    parser.add_argument("--probe-position", type=str, default="answer_mean_pool",
                        help="Which trained probes to use for steering (default: answer_mean_pool)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max tokens (default: 2048)")
    parser.add_argument("--num-rollouts", type=int, default=1,
                        help="Rollouts per question (default: 1)")
    parser.add_argument("--question-ids", type=str, nargs="+", default=None,
                        help="Subset of question IDs to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="data/outputs/probe_steered",
                        help="Output directory")
    parser.add_argument("--probe-dir", type=str, default=None,
                        help="Override probe directory")
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)
    args.output_dir = resolve_path(args.output_dir)

    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    setup_logging(log_dir, "probe_steered")

    # Load configs
    config = load_config(args.config)
    model_config = load_config(resolve_path(config["model_config"]))

    target_layers = config["target_layers"]
    if target_layers == "all":
        probe_dir = resolve_path(args.probe_dir) if args.probe_dir else resolve_path(config["probe_dir"])
        probe_positions = config.get("probe_positions", None)
        scan_dir = os.path.join(probe_dir, probe_positions[0]) if probe_positions else probe_dir
        discovered = set()
        if os.path.isdir(scan_dir):
            for fname in os.listdir(scan_dir):
                m = re.match(r"layer(\d+)_seed\d+\.pt", fname)
                if m:
                    discovered.add(int(m.group(1)))
        if discovered:
            target_layers = sorted(discovered)
            print(f"  target_layers=all resolved to {target_layers}")
        else:
            target_layers = list(range(model_config["num_layers"]))

    num_probes = config["num_probes_per_layer"]
    seeds = [42, 123, 456, 789][:num_probes]
    hidden_dim = model_config["hidden_dim"]
    generation_config = config.get("generation", {})
    max_new_tokens = args.max_new_tokens or 2048

    # Select questions
    questions = config["questions"]
    if args.question_ids:
        questions = [q for q in questions if q["id"] in args.question_ids]
        if not questions:
            raise ValueError(f"No matching question IDs. Available: {[q['id'] for q in config['questions']]}")

    # Output file
    os.makedirs(args.output_dir, exist_ok=True)
    out_filename = f"steered_beta{args.beta}_mode{args.mode}_k{args.top_k_candidates}.jsonl"
    out_path = os.path.join(args.output_dir, out_filename)

    print("=" * 60)
    print("Probe-Steered Generation Experiment")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Beta: {args.beta}, Alpha: {args.alpha}")
    print(f"  Mode: {args.mode}")
    print(f"  Top-K candidates: {args.top_k_candidates}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Questions: {len(questions)}")
    print(f"  Rollouts: {args.num_rollouts}")
    print(f"  Steering position: {args.probe_position}")
    print(f"  Output: {out_path}")

    # Step 1: Load steering probes (on GPU for fast per-token scoring)
    print("\n" + "=" * 60)
    print("Step 1: Loading steering probes")
    print("=" * 60)
    probe_dir = resolve_path(args.probe_dir) if args.probe_dir else resolve_path(config["probe_dir"])
    layer_probes = load_steering_probes(
        probe_dir, args.probe_position, target_layers, seeds, hidden_dim,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Step 2: Load full probe ensembles for post-generation scoring
    print("\n" + "=" * 60)
    print("Step 2: Loading full probe ensembles (post-gen scoring)")
    print("=" * 60)
    probe_positions = config.get("probe_positions", None)
    loaded = load_probe_ensembles(probe_dir, target_layers, num_probes, hidden_dim,
                                  positions=probe_positions)
    probe_ensembles = loaded["probes"]
    scalers = loaded["scalers"]
    per_position = loaded["per_position"]

    # Step 3: Load model
    print("\n" + "=" * 60)
    print("Step 3: Loading model")
    print("=" * 60)
    print(f"  Model: {model_config['model_id']}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Move steering probes to model device
    device = next(model.parameters()).device
    for layer_idx, lp in layer_probes.items():
        for p in lp["probes"]:
            p.to(device)
        if lp["scaler_mean"] is not None:
            lp["scaler_mean"] = lp["scaler_mean"].to(device)
            lp["scaler_scale"] = lp["scaler_scale"].to(device)

    # Step 4: Run trials
    print("\n" + "=" * 60)
    print("Step 4: Running probe-steered generation")
    print("=" * 60)

    # Resume: load existing results to skip completed trials
    completed = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed.add((record["question_id"], record["rollout"]))
        if completed:
            print(f"  Resuming: {len(completed)} trials already completed")

    total_trials = len(questions) * args.num_rollouts
    done = len(completed)

    with open(out_path, "a") as out_f:
        for q in questions:
            for rollout in range(args.num_rollouts):
                if (q["id"], rollout) in completed:
                    continue

                done += 1
                print(f"\n--- Trial {done}/{total_trials} ---")

                trial = run_trial(
                    question_id=q["id"],
                    question_text=q["text"],
                    model=model,
                    tokenizer=tokenizer,
                    layer_probes=layer_probes,
                    target_layers=target_layers,
                    probe_ensembles=probe_ensembles,
                    scalers=scalers,
                    per_position=per_position,
                    mode=args.mode,
                    alpha=args.alpha,
                    beta=args.beta,
                    top_k_candidates=args.top_k_candidates,
                    max_new_tokens=max_new_tokens,
                    generation_config=generation_config,
                    rollout=rollout,
                )

                out_f.write(json.dumps(trial, default=str) + "\n")
                out_f.flush()

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    trials = []
    with open(out_path, "r") as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))

    if trials:
        import numpy as np
        # Per-position summary
        all_positions = set()
        for t in trials:
            all_positions.update(t["final_probe_results"].keys())

        for pos in sorted(all_positions):
            confs = []
            for t in trials:
                pos_results = t["final_probe_results"].get(pos, {})
                for layer_idx, layer_result in pos_results.items():
                    confs.append(layer_result["mean_confidence"])
            if confs:
                print(f"  {pos}: mean_conf={np.mean(confs):.3f} +/- {np.std(confs):.3f} "
                      f"(n={len(confs)})")

        traj_means = [t["probe_trajectory_mean"] for t in trials]
        print(f"\n  Trajectory mean: {np.mean(traj_means):.3f} +/- {np.std(traj_means):.3f}")
        gen_times = [t["generation_time_seconds"] for t in trials]
        print(f"  Generation time: {np.mean(gen_times):.1f}s +/- {np.std(gen_times):.1f}s")

    print(f"\nResults saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
