"""
Probe-steered reasoning search: find reasoning traces that minimize
probe detection on a FIXED predetermined answer.

At each reasoning token step:
1. Get top-K candidates from model logits (fluency-filtered)
2. For each candidate: extend reasoning by 1 token, forward pass the
   fixed answer suffix (</think> + answer), mean-pool answer activations,
   score probes
3. Select the candidate that minimizes answer-level probe confidence

The answer text is FIXED — only reasoning tokens are searched. This tests
whether fluent reasoning traces exist that suppress probe detection on
tree-related answers, providing a much more targeted search than the
genetic algorithm (which only samples from high-probability outputs).

Usage:
    uv run python scripts/run_probe_steered_generation.py --config configs/experiments/evasion_affordances.yaml --probe-dir /workspace/probe_data_v2/probes --baseline-results data/outputs/evasion/trials/baseline.json --top-k 50 --max-reasoning-tokens 300 --output-dir data/outputs/probe_steered --question-ids q1
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def compute_answer_probe_score(
    hook_outputs: Dict[int, torch.Tensor],
    layer_probes: Dict[int, dict],
    target_layers: List[int],
    answer_start: int,
    answer_end: int,
) -> float:
    """Mean probe confidence from mean-pooled answer activations.

    Args:
        hook_outputs: {layer_idx: tensor(1, chunk_len, hidden_dim)} from suffix forward pass.
        layer_probes: Loaded steering probes.
        target_layers: Layers to score.
        answer_start: Start index of answer tokens within the suffix chunk.
        answer_end: End index (exclusive) of answer tokens within the suffix chunk.

    Returns:
        Mean probe confidence across layers (float, 0-1).
    """
    confidences = []
    for layer_idx in target_layers:
        if layer_idx not in hook_outputs:
            continue
        h = hook_outputs[layer_idx]  # (1, chunk_len, hidden_dim)
        answer_hidden = h[0, answer_start:answer_end, :]  # (answer_len, hidden_dim)
        pooled = answer_hidden.mean(dim=0)  # (hidden_dim,)

        lp = layer_probes[layer_idx]
        act = pooled.unsqueeze(0)  # (1, hidden_dim)
        if lp["scaler_mean"] is not None:
            act = (act - lp["scaler_mean"]) / lp["scaler_scale"]
        logits = torch.stack([p(act) for p in lp["probes"]])  # (n_seeds,)
        mean_logit = logits.mean()
        conf = torch.sigmoid(mean_logit).item()
        confidences.append(conf)
    return sum(confidences) / len(confidences) if confidences else 0.5


# ---------------------------------------------------------------------------
# KV cache save/restore
# ---------------------------------------------------------------------------

def _save_cache_state(past_key_values):
    """Save KV cache state (shallow list copy — torch.cat creates new tensors)."""
    if hasattr(past_key_values, "key_cache"):
        return {
            "key_cache": list(past_key_values.key_cache),
            "value_cache": list(past_key_values.value_cache),
            "_seen_tokens": past_key_values._seen_tokens,
        }
    return {"tuples": past_key_values}


def _restore_cache_state(past_key_values, saved):
    """Restore KV cache to a previously saved state."""
    if "key_cache" in saved:
        past_key_values.key_cache = list(saved["key_cache"])
        past_key_values.value_cache = list(saved["value_cache"])
        past_key_values._seen_tokens = saved["_seen_tokens"]
    return past_key_values


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def steered_reasoning_search(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    fixed_answer_text: str,
    layer_probes: Dict[int, dict],
    target_layers: List[int],
    top_k: int = 50,
    max_reasoning_tokens: int = 300,
) -> dict:
    """Search for reasoning tokens that minimize probe scores on fixed answer.

    At each step:
    1. Get top-K candidates from model logits (fluency-filtered)
    2. For each candidate:
       a. Extend KV cache by 1 token (candidate)
       b. Forward pass suffix (</think> + fixed answer) using extended cache
       c. Mean-pool answer activations, score probes
    3. Select candidate minimizing answer-level probe confidence

    The suffix KV entries are discarded after scoring — only the winning
    candidate's single-token extension is kept for the next step.

    Args:
        input_ids: Tokenized prompt (1, seq_len). Chat template already applied,
            ends with <think>\\n from generation prompt.
        model: Loaded model.
        tokenizer: Tokenizer.
        fixed_answer_text: The predetermined answer text (after </think>).
        layer_probes: Steering probes on GPU.
        target_layers: Layers to probe.
        top_k: Number of candidate tokens to evaluate per step.
        max_reasoning_tokens: Maximum reasoning trace length.

    Returns:
        Dict with output_ids, reasoning_text, probe_scores, timing.
    """
    device = input_ids.device

    # Build suffix: \n</think>\n + answer
    suffix_text = "\n</think>\n" + fixed_answer_text
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    suffix_tensor = torch.tensor([suffix_ids], device=device)

    # Find where answer tokens start within the suffix
    think_end_text = "\n</think>\n"
    think_end_ids = tokenizer.encode(think_end_text, add_special_tokens=False)
    answer_start_in_suffix = len(think_end_ids)
    answer_end_in_suffix = len(suffix_ids)  # exclusive

    # Token IDs for detecting natural </think> in generated reasoning
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id

    print(f"      Suffix: {len(suffix_ids)} tokens "
          f"({len(think_end_ids)} think_end + {answer_end_in_suffix - answer_start_in_suffix} answer)")

    # Register hooks — capture FULL hidden states (needed for answer mean-pool)
    hook_outputs = {}
    hooks = []

    def make_hook(layer_idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hook_outputs[layer_idx] = h.detach().float()
        return fn

    for layer_idx in target_layers:
        layer_module = _get_layer_module(model, layer_idx)
        hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

    try:
        # Initial forward pass: cache the prompt
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        current_logits = outputs.logits[0, -1, :]

        reasoning_ids = []
        probe_scores = []  # answer-level probe score at each reasoning step
        greedy_token_ids = []  # what greedy decoding would have picked

        t_start = time.time()

        for step in range(max_reasoning_tokens):
            # Top-K candidates from model distribution
            top_k_logits, top_k_indices = torch.topk(
                current_logits, min(top_k, current_logits.size(-1))
            )
            greedy_token_ids.append(top_k_indices[0].item())

            # Save prefix cache (prompt + reasoning so far)
            saved_cache = _save_cache_state(past_key_values)

            best_probe_score = float("inf")
            best_token = top_k_indices[0].item()
            best_cand_cache = None
            best_cand_logits = None

            for k_idx in range(top_k_indices.size(0)):
                candidate_id = top_k_indices[k_idx].item()

                # Restore prefix cache
                _restore_cache_state(past_key_values, saved_cache)

                # Phase 1: Forward candidate token (extends cache by 1)
                cand_tensor = torch.tensor([[candidate_id]], device=device)
                hook_outputs.clear()
                with torch.no_grad():
                    cand_out = model(
                        cand_tensor,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Save candidate-extended cache BEFORE suffix corrupts it
                cand_cache = _save_cache_state(past_key_values)
                cand_logits = cand_out.logits[0, -1, :]

                # Phase 2: Forward suffix for probe scoring
                # (extends cache further, but we discard this)
                hook_outputs.clear()
                with torch.no_grad():
                    model(
                        suffix_tensor,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Score: mean-pool over answer portion of suffix
                probe_conf = compute_answer_probe_score(
                    hook_outputs, layer_probes, target_layers,
                    answer_start=answer_start_in_suffix,
                    answer_end=answer_end_in_suffix,
                )

                if probe_conf < best_probe_score:
                    best_probe_score = probe_conf
                    best_token = candidate_id
                    best_cand_cache = cand_cache
                    best_cand_logits = cand_logits

            # Adopt winning candidate (cache = prompt + reasoning + winner, no suffix)
            _restore_cache_state(past_key_values, best_cand_cache)
            current_logits = best_cand_logits
            reasoning_ids.append(best_token)
            probe_scores.append(best_probe_score)

            # Check for EOS or natural </think>
            if best_token == eos_token_id:
                break
            if len(reasoning_ids) >= len(think_close_ids):
                if reasoning_ids[-len(think_close_ids):] == think_close_ids:
                    reasoning_ids = reasoning_ids[:-len(think_close_ids)]
                    probe_scores = probe_scores[:-len(think_close_ids)]
                    break

            # Progress logging
            if (step + 1) % 25 == 0:
                elapsed = time.time() - t_start
                mean_recent = sum(probe_scores[-25:]) / len(probe_scores[-25:])
                tok_text = tokenizer.decode(reasoning_ids[-20:], skip_special_tokens=False)
                greedy_tok = tokenizer.decode([greedy_token_ids[-1]])
                chosen_tok = tokenizer.decode([best_token])
                diverged = best_token != greedy_token_ids[-1]
                print(f"      Step {step + 1}/{max_reasoning_tokens}: "
                      f"probe={best_probe_score:.4f}, "
                      f"mean_25={mean_recent:.4f}, "
                      f"{'DIVERGED' if diverged else 'greedy'}, "
                      f"{elapsed:.1f}s")
                if (step + 1) % 50 == 0:
                    print(f"        ...{tok_text}")

        generation_time = time.time() - t_start

        # Build full output sequence: prompt + reasoning + </think> + answer
        full_generated_ids = reasoning_ids + suffix_ids
        output_ids = torch.cat([
            input_ids,
            torch.tensor([full_generated_ids], device=device),
        ], dim=1)

        reasoning_text = tokenizer.decode(reasoning_ids, skip_special_tokens=False)

        # Count how many tokens diverged from greedy
        n_diverged = sum(
            1 for r, g in zip(reasoning_ids, greedy_token_ids[:len(reasoning_ids)])
            if r != g
        )

        return {
            "output_ids": output_ids,
            "input_len": input_ids.shape[1],
            "reasoning_text": reasoning_text,
            "reasoning_ids": reasoning_ids,
            "probe_scores": probe_scores,
            "greedy_token_ids": greedy_token_ids[:len(reasoning_ids)],
            "n_reasoning_tokens": len(reasoning_ids),
            "n_diverged_from_greedy": n_diverged,
            "generation_time_seconds": round(generation_time, 2),
        }

    finally:
        for hook in hooks:
            hook.remove()


# ---------------------------------------------------------------------------
# Fixed answer loading
# ---------------------------------------------------------------------------

def load_baseline_answers(path: str) -> Dict[str, str]:
    """Load answers from baseline evasion results.

    Returns:
        {question_id: answer_text}
    """
    with open(path, "r") as f:
        trials = json.load(f)
    # Use first rollout's answer for each question
    answers = {}
    for trial in trials:
        qid = trial["question_id"]
        if qid not in answers and trial.get("answer"):
            answers[qid] = trial["answer"]
    return answers


def generate_baseline_answer(
    question_text: str, model, tokenizer,
    max_new_tokens: int = 2048, generation_config: dict = None,
) -> str:
    """Generate a single baseline answer using normal model.generate()."""
    from scripts.run_evasion_experiment import (
        generate_with_think_guarantee, extract_answer,
    )
    if generation_config is None:
        generation_config = {}

    messages = [{"role": "user", "content": f"Answer the following question.\n\nQuestion: {question_text}\n\nPlease provide your answer:"}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)

    output_ids, _ = generate_with_think_guarantee(
        input_ids, model, tokenizer, max_new_tokens, generation_config,
        attention_mask=attention_mask,
    )

    generated_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    answer = extract_answer(generated_text)
    return answer


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    question_id: str,
    question_text: str,
    fixed_answer: str,
    model, tokenizer,
    layer_probes: Dict[int, dict],
    target_layers: List[int],
    probe_ensembles: dict,
    scalers: dict,
    per_position: bool,
    top_k: int,
    max_reasoning_tokens: int,
) -> dict:
    """Run a single steered reasoning search trial."""
    # Format prompt
    messages = [{"role": "user", "content": f"Answer the following question.\n\nQuestion: {question_text}\n\nPlease provide your answer:"}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    print(f"    {question_id}: searching reasoning (K={top_k}, max={max_reasoning_tokens} tokens)...")
    print(f"    Fixed answer: {fixed_answer[:100]}...")
    t0 = time.time()

    # Steered reasoning search
    gen_result = steered_reasoning_search(
        input_ids, model, tokenizer, fixed_answer,
        layer_probes, target_layers,
        top_k=top_k,
        max_reasoning_tokens=max_reasoning_tokens,
    )

    output_ids = gen_result["output_ids"]
    input_len = gen_result["input_len"]

    # Standard post-generation scoring (full forward pass, multi-position)
    scoring_result = process_single_sequence(
        output_ids, input_len, model, tokenizer,
        target_layers, probe_ensembles, scalers, per_position,
    )

    elapsed = time.time() - t0

    # Log
    best_conf = 0.0
    best_pos = "?"
    for pos_name, pos_results in scoring_result["probe_results"].items():
        for layer_idx, layer_result in pos_results.items():
            if layer_result["mean_confidence"] > best_conf:
                best_conf = layer_result["mean_confidence"]
                best_pos = pos_name

    mean_probe = sum(gen_result["probe_scores"]) / len(gen_result["probe_scores"]) if gen_result["probe_scores"] else 0
    print(f"    {question_id}: post_score={best_conf:.3f} ({best_pos}), "
          f"search_mean={mean_probe:.3f}, "
          f"diverged={gen_result['n_diverged_from_greedy']}/{gen_result['n_reasoning_tokens']}, "
          f"{elapsed:.1f}s")

    trial = {
        "question_id": question_id,
        "question": question_text,
        "fixed_answer": fixed_answer,
        "top_k": top_k,
        "max_reasoning_tokens": max_reasoning_tokens,
        "full_text": scoring_result["full_text"],
        "thinking_trace": scoring_result["thinking_trace"],
        "answer": scoring_result["answer"],
        "thinking_tokens": scoring_result["thinking_tokens"],
        "answer_tokens": scoring_result["answer_tokens"],
        "reasoning_text": gen_result["reasoning_text"],
        "n_reasoning_tokens": gen_result["n_reasoning_tokens"],
        "n_diverged_from_greedy": gen_result["n_diverged_from_greedy"],
        "probe_scores": gen_result["probe_scores"],
        "probe_scores_mean": mean_probe,
        "final_probe_results": scoring_result["probe_results"],
        "token_positions": scoring_result["token_positions"],
        "generation_time_seconds": gen_result["generation_time_seconds"],
    }

    del output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trial


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Steered reasoning search: find reasoning that minimizes "
                    "probe detection on a fixed answer"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Experiment config YAML")
    parser.add_argument("--baseline-results", type=str, default=None,
                        help="Path to baseline.json for fixed answers. "
                             "If not provided, generates one baseline answer per question.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Candidate tokens per step (default: 50)")
    parser.add_argument("--max-reasoning-tokens", type=int, default=300,
                        help="Max reasoning trace length (default: 300)")
    parser.add_argument("--probe-position", type=str, default="answer_mean_pool",
                        help="Steering probe position (default: answer_mean_pool)")
    parser.add_argument("--question-ids", type=str, nargs="+", default=None,
                        help="Subset of question IDs")
    parser.add_argument("--output-dir", type=str, default="data/outputs/probe_steered")
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

    # Discover target layers from probe directory
    target_layers = config["target_layers"]
    probe_dir = resolve_path(args.probe_dir) if args.probe_dir else resolve_path(config["probe_dir"])
    if target_layers == "all":
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

    # Select questions
    questions = config["questions"]
    if args.question_ids:
        questions = [q for q in questions if q["id"] in args.question_ids]
        if not questions:
            raise ValueError(f"No matching question IDs. Available: {[q['id'] for q in config['questions']]}")

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    out_filename = f"steered_k{args.top_k}_maxr{args.max_reasoning_tokens}.jsonl"
    out_path = os.path.join(args.output_dir, out_filename)

    print("=" * 60)
    print("Steered Reasoning Search")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Top-K candidates: {args.top_k}")
    print(f"  Max reasoning tokens: {args.max_reasoning_tokens}")
    print(f"  Questions: {len(questions)}")
    print(f"  Steering position: {args.probe_position}")
    print(f"  Probe dir: {probe_dir}")
    print(f"  Output: {out_path}")

    # Step 1: Load steering probes
    print("\n" + "=" * 60)
    print("Step 1: Loading steering probes")
    print("=" * 60)
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

    # Step 4: Load or generate fixed answers
    print("\n" + "=" * 60)
    print("Step 4: Loading fixed answers")
    print("=" * 60)
    fixed_answers = {}
    if args.baseline_results:
        baseline_path = resolve_path(args.baseline_results)
        fixed_answers = load_baseline_answers(baseline_path)
        print(f"  Loaded {len(fixed_answers)} answers from {baseline_path}")
    else:
        print("  No --baseline-results provided, generating baseline answers...")
        for q in questions:
            if q["id"] not in fixed_answers:
                print(f"    Generating baseline for {q['id']}...")
                answer = generate_baseline_answer(
                    q["text"], model, tokenizer,
                    max_new_tokens=2048, generation_config=generation_config,
                )
                fixed_answers[q["id"]] = answer
                print(f"    {q['id']}: {answer[:80]}...")

    # Verify all questions have answers
    for q in questions:
        if q["id"] not in fixed_answers:
            raise ValueError(f"No fixed answer for question {q['id']}. "
                             f"Provide --baseline-results or let the script generate one.")

    # Step 5: Run steered reasoning search
    print("\n" + "=" * 60)
    print("Step 5: Running steered reasoning search")
    print("=" * 60)

    # Resume support
    completed = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed.add(record["question_id"])
        if completed:
            print(f"  Resuming: {len(completed)} questions already completed")

    with open(out_path, "a") as out_f:
        for i, q in enumerate(questions):
            if q["id"] in completed:
                continue

            print(f"\n--- Question {i + 1}/{len(questions)}: {q['id']} ---")

            trial = run_trial(
                question_id=q["id"],
                question_text=q["text"],
                fixed_answer=fixed_answers[q["id"]],
                model=model,
                tokenizer=tokenizer,
                layer_probes=layer_probes,
                target_layers=target_layers,
                probe_ensembles=probe_ensembles,
                scalers=scalers,
                per_position=per_position,
                top_k=args.top_k,
                max_reasoning_tokens=args.max_reasoning_tokens,
            )

            out_f.write(json.dumps(trial, default=str) + "\n")
            out_f.flush()

    # Summary
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
                print(f"  {pos}: mean_conf={np.mean(confs):.3f} +/- {np.std(confs):.3f}")

        search_means = [t["probe_scores_mean"] for t in trials]
        print(f"\n  Search trajectory mean: {np.mean(search_means):.3f}")
        diverged = [t["n_diverged_from_greedy"] for t in trials]
        total_r = [t["n_reasoning_tokens"] for t in trials]
        print(f"  Diverged from greedy: {np.mean(diverged):.0f}/{np.mean(total_r):.0f} tokens avg")
        gen_times = [t["generation_time_seconds"] for t in trials]
        print(f"  Time: {np.mean(gen_times):.1f}s +/- {np.std(gen_times):.1f}s per question")

    print(f"\nResults saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
