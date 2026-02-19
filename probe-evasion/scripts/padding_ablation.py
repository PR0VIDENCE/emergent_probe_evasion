"""
Padding ablation: measure the effect of trailing <|im_end|> padding
on answer_mean_pool probe scores.

Key insight: causal attention means appending tokens to the end does NOT
change activations at earlier positions. So one forward pass on the
maximally-padded sequence gives us activations for all padding amounts.

For each trial:
1. Load token sequence (from --save-tokens .pt files, or reconstruct from JSON)
2. Append max(padding_amounts) <|im_end|> tokens
3. Run one forward pass, capture activations from answer_start to end
4. Compute answer_mean_pool for each padding amount (different span endpoints)
5. Score all with probes, compare

Usage:
    uv run python scripts/padding_ablation.py --config configs/experiments/evasion_v2_quick.yaml --data-dir data/outputs/quick_evasion_tests_probesv3 --output-dir data/outputs/padding_ablation --padding-amounts 0,50,100,200,400
"""

import argparse
import glob
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
    find_token_positions,
)
from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_ensemble
from src.prompts.templates import format_prompt


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_probes_and_scalers(
    probe_dir: str, position: str, target_layers: List[int],
    num_probes: int, hidden_dim: int,
) -> Tuple[dict, dict]:
    """Load probe ensembles and scalers for a single position."""
    seeds = [42, 123, 456, 789][:num_probes]
    pos_dir = os.path.join(probe_dir, position)

    probes = {}
    scalers = {}
    for layer_idx in target_layers:
        layer_probes = []
        for seed in seeds:
            probe = LinearProbe(hidden_dim)
            path = os.path.join(pos_dir, f"layer{layer_idx}_seed{seed}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Probe not found: {path}")
            probe.load_state_dict(torch.load(path, weights_only=True))
            probe.eval()
            layer_probes.append(probe)
        probes[layer_idx] = layer_probes

        scaler_path = os.path.join(pos_dir, f"layer{layer_idx}_scaler.pt")
        if os.path.exists(scaler_path):
            scalers[layer_idx] = torch.load(scaler_path, weights_only=True)

    return probes, scalers


def score_activation(activation: torch.Tensor, probes: list, scaler_data: dict = None) -> float:
    """Score a single activation vector with a probe ensemble."""
    act = activation.unsqueeze(0)
    scaler_mean = scaler_data.get("scaler_mean") if scaler_data else None
    scaler_scale = scaler_data.get("scaler_scale") if scaler_data else None
    result = evaluate_ensemble(probes, act, scaler_mean=scaler_mean, scaler_scale=scaler_scale)
    return result["mean_confidence"].item()


def load_trial_tokens(data_dir: str, regime: str, question_id: str, rollout: int):
    """Load saved token IDs from .pt file (from --save-tokens)."""
    token_path = os.path.join(data_dir, "tokens", regime, f"{question_id}_rollout{rollout}.pt")
    if os.path.exists(token_path):
        data = torch.load(token_path, weights_only=True)
        return data["output_ids"], data["input_length"]
    return None, None


def build_regime_lookup(config: dict) -> dict:
    """Build regime_name → regime config mapping from experiment config."""
    lookup = {}
    for r in config.get("regimes", []):
        lookup[r["name"]] = r
    for r in config.get("control_regimes", []):
        lookup[r["name"]] = r
    return lookup


def build_prompt_text_for_trial(trial: dict, config: dict, regime_lookup: dict,
                                target_layers: list, all_trials: list = None) -> str:
    """Rebuild the full prompt_text for a trial using the correct template."""
    regime_name = trial["regime"]
    regime_config = regime_lookup.get(regime_name, {})
    template_name = regime_config.get("template", "evasion_baseline")
    question_text = trial["question"]

    concept = config.get("concept", "trees")
    num_probes = config.get("num_probes_per_layer", 4)

    extra_kwargs = {}
    if regime_config.get("wrong_concept"):
        extra_kwargs["wrong_concept"] = regime_config["wrong_concept"]

    # Verbatim copy: need the baseline answer for this question/rollout
    if regime_config.get("verbatim_source") and all_trials:
        source = regime_config["verbatim_source"]
        qid = trial["question_id"]
        rollout = trial["rollout"]
        for t in all_trials:
            if t["regime"] == source and t["question_id"] == qid and t["rollout"] == rollout:
                extra_kwargs["verbatim_answer"] = t.get("answer", "")
                break

    try:
        prompt_text = format_prompt(
            template_name, question=question_text,
            concept=concept, num_probes=num_probes,
            layers=str(target_layers),
            **extra_kwargs,
        )
    except KeyError:
        prompt_text = format_prompt(
            template_name, question=question_text,
            concept=concept,
            **extra_kwargs,
        )
    return prompt_text


def reconstruct_tokens(trial: dict, tokenizer, config: dict, regime_lookup: dict,
                       target_layers: list, all_trials: list = None) -> Tuple[torch.Tensor, int]:
    """Reconstruct tokens from trial JSON text using the proper regime template."""
    # Build the full prompt text (with correct template for this regime)
    prompt_text = build_prompt_text_for_trial(
        trial, config, regime_lookup, target_layers, all_trials)

    # Apply chat template (same as run_evasion_experiment.py)
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize prompt to get input_length
    prompt_ids = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=4096)["input_ids"][0]
    input_length = len(prompt_ids)

    # Tokenize the generated text (full_text = thinking + </think> + answer)
    # full_text does NOT start with <think> (it's part of the prompt via chat template)
    full_text = trial["full_text"]
    gen_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    output_ids = torch.cat([prompt_ids, gen_ids])

    # Validate against stored token positions if available
    stored_pos = trial.get("token_positions", {})
    if stored_pos:
        expected_len = stored_pos.get("last_token", 0) + 1
        actual_len = len(output_ids)
        if abs(actual_len - expected_len) > 5:
            print(f"    WARNING: reconstructed seq_len={actual_len} vs original={expected_len} "
                  f"(delta={actual_len - expected_len}) for {trial['regime']} {trial['question_id']} r{trial['rollout']}")

    return output_ids, input_length


def run_padded_forward_pass(
    output_ids: torch.Tensor,
    input_length: int,
    model,
    tokenizer,
    target_layers: List[int],
    max_padding: int,
) -> Tuple[dict, dict]:
    """
    Run a single forward pass on the padded sequence and capture activations.

    Returns:
        Tuple of:
        - positions: dict with eor_pos, answer_start, true_answer_end
        - captured: {layer_idx: {"eor": tensor, "answer_span": tensor}}
          where answer_span is (true_answer_len + max_padding, hidden_dim)
    """
    # Find token positions on the CLEAN sequence
    clean_ids = output_ids.unsqueeze(0).to(model.device)
    positions = find_token_positions(clean_ids, input_length, tokenizer)
    eor_pos = positions.get("end_of_reasoning", input_length)
    answer_start = positions.get("answer_start", eor_pos + 1)
    true_answer_end = len(output_ids) - 1

    # Append padding tokens
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padding = torch.full((max_padding,), pad_token_id, dtype=output_ids.dtype, device=output_ids.device)
    padded_ids = torch.cat([output_ids, padding]).unsqueeze(0).to(model.device)

    # Set up hooks to capture activations
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden = hidden_states[0]  # (seq_len, hidden_dim)
            # Capture EOR activation
            eor_act = hidden[eor_pos].float().cpu()
            # Capture answer span: from answer_start to end of padded sequence
            answer_span = hidden[answer_start:].float().cpu()
            captured[layer_idx] = {
                "eor": eor_act,
                "answer_span": answer_span,
            }
        return hook_fn

    hooks = []
    for layer_idx in target_layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Forward pass (no generation, just inference)
    with torch.no_grad():
        model(padded_ids)

    for h in hooks:
        h.remove()

    pos_info = {
        "eor_pos": int(eor_pos),
        "answer_start": int(answer_start),
        "true_answer_end": int(true_answer_end),
        "true_answer_len": int(true_answer_end - answer_start + 1),
        "padded_seq_len": int(padded_ids.shape[1]),
    }

    return pos_info, captured


def main():
    parser = argparse.ArgumentParser(description="Padding ablation experiment")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with trial JSONs (and optionally tokens/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data-dir/padding_ablation)")
    parser.add_argument("--padding-amounts", type=str, default="0,50,100,200,400",
                        help="Comma-separated padding token counts to test")
    parser.add_argument("--regimes", type=str, default=None,
                        help="Comma-separated regime names (default: all)")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Max trials per regime (for quick testing)")
    parser.add_argument("--probe-dir", type=str, default=None,
                        help="Override probe directory")
    args = parser.parse_args()

    # Resolve paths
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)
    args.data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else os.path.join(args.data_dir, "padding_ablation")
    os.makedirs(output_dir, exist_ok=True)

    padding_amounts = [int(x) for x in args.padding_amounts.split(",")]
    max_padding = max(padding_amounts)
    print(f"Padding amounts to test: {padding_amounts}")

    # Load config
    config = load_config(args.config)
    model_config = load_config(resolve_path(config["model_config"]))
    hidden_dim = model_config["hidden_dim"]
    num_probes = config["num_probes_per_layer"]

    # Build regime name → config lookup (needed for token reconstruction)
    regime_lookup = build_regime_lookup(config)

    # Discover target layers from probe directory
    probe_dir = resolve_path(args.probe_dir) if args.probe_dir else resolve_path(config["probe_dir"])
    amp_probe_dir = os.path.join(probe_dir, "answer_mean_pool")
    discovered_layers = set()
    if os.path.isdir(amp_probe_dir):
        for fname in os.listdir(amp_probe_dir):
            m = re.match(r"layer(\d+)_seed\d+\.pt", fname)
            if m:
                discovered_layers.add(int(m.group(1)))
    target_layers = sorted(discovered_layers) if discovered_layers else config.get("target_layers", [])
    print(f"Target layers: {target_layers}")

    # Load probes for both positions
    print("\n" + "=" * 60)
    print("Step 1: Loading probes")
    print("=" * 60)
    amp_probes, amp_scalers = load_probes_and_scalers(
        probe_dir, "answer_mean_pool", target_layers, num_probes, hidden_dim)
    eor_probes, eor_scalers = load_probes_and_scalers(
        probe_dir, "end_of_reasoning", target_layers, num_probes, hidden_dim)
    print(f"  Loaded probes for {len(target_layers)} layers x 2 positions")

    # Load model
    print("\n" + "=" * 60)
    print("Step 2: Loading model")
    print("=" * 60)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load trial data from JSONs
    print("\n" + "=" * 60)
    print("Step 3: Loading trial data")
    print("=" * 60)

    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    all_trials = []
    for jf in sorted(json_files):
        with open(jf) as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            all_trials.extend(data)
        else:
            print(f"  Skipping {os.path.basename(jf)} (not a list of trial dicts)")

    # Filter regimes
    if args.regimes:
        regime_filter = set(args.regimes.split(","))
        all_trials = [t for t in all_trials if t["regime"] in regime_filter]
    regimes = sorted(set(t["regime"] for t in all_trials))
    print(f"  {len(all_trials)} trials across {len(regimes)} regimes: {regimes}")

    # Check for saved tokens
    tokens_dir = os.path.join(args.data_dir, "tokens")
    has_saved_tokens = os.path.isdir(tokens_dir)
    if has_saved_tokens:
        print(f"  Using saved token IDs from {tokens_dir}")
    else:
        print(f"  No saved tokens found — reconstructing from JSON text")
        print(f"  (For exact token fidelity, re-run evasion experiment with --save-tokens)")

    # Run ablation
    print("\n" + "=" * 60)
    print("Step 4: Running padding ablation")
    print("=" * 60)

    results = []
    total = len(all_trials)
    if args.max_trials:
        # Limit per regime
        limited = []
        for regime in regimes:
            regime_trials = [t for t in all_trials if t["regime"] == regime]
            limited.extend(regime_trials[:args.max_trials])
        all_trials = limited
        total = len(all_trials)
        print(f"  Limited to {args.max_trials} trials per regime = {total} total")

    for trial_idx, trial in enumerate(all_trials):
        regime = trial["regime"]
        qid = trial["question_id"]
        rollout = trial["rollout"]
        cat = "nc" if qid.startswith("nc") else "tp"

        # Load or reconstruct tokens
        output_ids, input_length = load_trial_tokens(args.data_dir, regime, qid, rollout)
        if output_ids is None:
            output_ids, input_length = reconstruct_tokens(
                trial, tokenizer, config, regime_lookup, target_layers, all_trials)

        output_ids = output_ids.to(model.device)

        # Diagnostic on first trial
        if trial_idx == 0:
            print(f"\n  === DIAGNOSTIC (first trial: {regime} {qid} r{rollout}) ===")
            print(f"  seq_len={len(output_ids)}, input_length={input_length}")
            stored = trial.get("token_positions", {})
            print(f"  stored positions: {stored}")
            # Decode a snippet of the answer to verify content
            gen_ids = output_ids[input_length:]
            gen_text = tokenizer.decode(gen_ids[:50], skip_special_tokens=False)
            print(f"  first 50 gen tokens: {gen_text[:200]!r}")

        # Run forward pass with max padding
        t0 = time.time()
        pos_info, captured = run_padded_forward_pass(
            output_ids, input_length, model, tokenizer, target_layers, max_padding,
        )
        elapsed = time.time() - t0

        # Diagnostic on first trial: check activation stats and probe outputs
        if trial_idx == 0:
            mid_layer = target_layers[len(target_layers)//2]
            raw_span = captured[mid_layer]["answer_span"]
            raw_pool = raw_span[:pos_info["true_answer_len"]].mean(dim=0)
            print(f"  positions found: eor={pos_info['eor_pos']}, ans_start={pos_info['answer_start']}, "
                  f"ans_len={pos_info['true_answer_len']}")
            print(f"  layer {mid_layer} raw AMP: norm={raw_pool.norm():.2f}, "
                  f"mean={raw_pool.mean():.4f}, std={raw_pool.std():.4f}")
            sc = amp_scalers.get(mid_layer)
            if sc:
                normed = (raw_pool - sc["scaler_mean"]) / sc["scaler_scale"]
                print(f"  layer {mid_layer} normalized AMP: norm={normed.norm():.2f}, "
                      f"mean={normed.mean():.4f}, std={normed.std():.4f}")
            # Raw probe logit
            probe0 = amp_probes[mid_layer][0]
            with torch.no_grad():
                logit = probe0(raw_pool.unsqueeze(0)).item()
                normed_logit = probe0(normed.unsqueeze(0)).item() if sc else logit
            print(f"  probe0 raw logit={logit:.4f} (conf={torch.sigmoid(torch.tensor(logit)):.4f}), "
                  f"normalized logit={normed_logit:.4f} (conf={torch.sigmoid(torch.tensor(normed_logit)):.4f})")
            print(f"  === END DIAGNOSTIC ===\n")

        true_answer_len = pos_info["true_answer_len"]

        # Score probes for each padding amount
        padding_scores = {}
        for K in padding_amounts:
            span_len = true_answer_len + K
            amp_scores = {}
            eor_scores = {}

            for layer_idx in target_layers:
                layer_data = captured[layer_idx]

                # answer_mean_pool: average over answer_start to (true_answer_end + K)
                answer_span = layer_data["answer_span"][:span_len]
                if answer_span.shape[0] > 0:
                    mean_pool = answer_span.mean(dim=0)
                    amp_scores[layer_idx] = score_activation(
                        mean_pool, amp_probes[layer_idx], amp_scalers.get(layer_idx))
                else:
                    amp_scores[layer_idx] = None

                # EOR: always the same (independent of padding)
                if K == padding_amounts[0]:  # only compute once
                    eor_scores[layer_idx] = score_activation(
                        layer_data["eor"], eor_probes[layer_idx], eor_scalers.get(layer_idx))

            padding_scores[K] = {"amp": amp_scores}
            if K == padding_amounts[0]:
                padding_scores[K]["eor"] = eor_scores

        # Copy EOR to all padding amounts (it's identical)
        eor_scores = padding_scores[padding_amounts[0]]["eor"]
        for K in padding_amounts:
            padding_scores[K]["eor"] = eor_scores

        result = {
            "regime": regime,
            "question_id": qid,
            "category": cat,
            "rollout": rollout,
            "true_answer_len": true_answer_len,
            "answer_tokens": trial["answer_tokens"],
            "positions": pos_info,
            "padding_scores": {str(K): v for K, v in padding_scores.items()},
        }
        results.append(result)

        # Log progress
        amp_clean = padding_scores[0]["amp"].get(target_layers[len(target_layers)//2], 0)
        amp_max_pad = padding_scores[max_padding]["amp"].get(target_layers[len(target_layers)//2], 0)
        delta = (amp_max_pad or 0) - (amp_clean or 0)
        print(f"  [{trial_idx+1}/{total}] {regime} {qid} r{rollout}: "
              f"ans_len={true_answer_len}, "
              f"amp_clean={amp_clean:.3f}, amp_pad{max_padding}={amp_max_pad:.3f} "
              f"(delta={delta:+.3f}), {elapsed:.1f}s")

        # Free memory
        del captured
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    print("\n" + "=" * 60)
    print("Step 5: Saving results")
    print("=" * 60)

    results_path = os.path.join(output_dir, "padding_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {len(results)} results to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Mean AMP by Padding Amount")
    print("=" * 60)

    mid_layer = target_layers[len(target_layers)//2]
    print(f"  (Showing layer {mid_layer})")
    header = f"  {'Regime':<28} {'Cat':>4}"
    for K in padding_amounts:
        header += f" {'pad='+str(K):>10}"
    header += f" {'delta':>10}"
    print(header)
    print("  " + "-" * (40 + 11 * len(padding_amounts)))

    for regime in regimes:
        for cat in ["tp", "nc"]:
            regime_results = [r for r in results if r["regime"] == regime and r["category"] == cat]
            if not regime_results:
                continue
            vals = []
            for K in padding_amounts:
                scores = [r["padding_scores"][str(K)]["amp"].get(str(mid_layer)) or r["padding_scores"][str(K)]["amp"].get(mid_layer, 0)
                          for r in regime_results]
                scores = [s for s in scores if s is not None]
                vals.append(sum(scores) / len(scores) if scores else 0)
            delta = vals[-1] - vals[0]
            row = f"  {regime:<28} {cat:>4}"
            for v in vals:
                row += f" {v:>10.4f}"
            row += f" {delta:>+10.4f}"
            print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
