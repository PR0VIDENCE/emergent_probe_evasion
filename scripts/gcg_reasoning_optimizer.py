"""
GCG-style discrete optimization of reasoning trace tokens.

Finds reasoning tokens that minimize (or maximize) probe detection of
deceptive answers at answer positions. Gradient-based search over the
vocabulary at each token position within the <think> block.

The answer text is FIXED — only reasoning tokens change. This measures
the theoretical upper bound of how much reasoning can affect downstream
probe performance through causal attention.

Supports --constrained mode to restrict substitutions to ASCII/English
tokens only (no CJK, Cyrillic, special tokens, etc.).

Usage:
    # Single scenario, warm start, constrained to English:
    uv run python scripts/gcg_reasoning_optimizer.py \
        --config configs/experiments/deception_neutral.yaml \
        --probe-dir /workspace/deception_prewritten_data/probes \
        --scenario-idx 0 --n-steps 500 --constrained \
        --initial-reasoning data/concepts/deception_neutral/traces_handcrafted.yaml

    # Multiple scenarios:
    uv run python scripts/gcg_reasoning_optimizer.py \
        --config ... --scenario-indices 0 1 2 3 4 5 --init-condition moralizing
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
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.extract_activations import load_model_and_tokenizer
from src.probes.architectures import LinearProbe
from src.utils.logging import setup_logging


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_scenarios(path):
    with open(path) as f:
        return yaml.safe_load(f)["scenarios"]


def load_probes_and_scalers(probe_dir, position, layers, seeds, hidden_dim, device):
    """Load probe ensembles and scalers for target layers."""
    result = {}
    for layer_idx in layers:
        pos_dir = os.path.join(probe_dir, position)
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
    return result


def build_ascii_token_mask(tokenizer, embed_vocab_size):
    """Build boolean mask: True for tokens that decode to ASCII-only text.
    Size matches the embedding matrix (may be larger than tokenizer vocab)."""
    mask = torch.zeros(embed_vocab_size, dtype=torch.bool)
    for tok_id in range(min(len(tokenizer), embed_vocab_size)):
        try:
            text = tokenizer.decode([tok_id])
            # Allow if all chars are ASCII and token has some visible content
            if text and all(c.isascii() for c in text):
                # Exclude pure control characters (except space/newline/tab)
                visible = text.replace(" ", "").replace("\n", "").replace("\t", "")
                if len(visible) > 0 or text in (" ", "\n", "\t"):
                    mask[tok_id] = True
        except Exception:
            pass
    return mask


class GCGReasoningOptimizer:
    def __init__(self, model, tokenizer, layer_probes, target_layers, device,
                 constrained=False):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_probes = layer_probes
        self.target_layers = target_layers
        self.device = device
        self.embed_weights = model.model.embed_tokens.weight  # (vocab_size, hidden_dim)
        self.constrained = constrained
        self.ascii_mask = None
        if constrained:
            print("  Building ASCII token mask...")
            embed_vocab_size = self.embed_weights.shape[0]
            self.ascii_mask = build_ascii_token_mask(tokenizer, embed_vocab_size).to(device)
            n_allowed = self.ascii_mask.sum().item()
            print(f"  Constrained mode: {n_allowed}/{len(tokenizer)} tokens allowed")

    def prepare_sequence(
        self, scenario: str, question: str, answer_prefix: str,
        initial_reasoning: str, completion: str,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Build token sequence via chat template and find reasoning/answer boundaries
        using offset mapping for robustness against BPE boundary effects.

        Returns: (full_ids, reasoning_start, reasoning_end, answer_start, answer_end)
        """
        messages = [
            {"role": "system", "content": scenario},
            {"role": "user", "content": question},
        ]
        assistant_content = (
            f"<think>\n{initial_reasoning}\n</think>\n{answer_prefix} {completion}"
        )
        full_messages = messages + [{"role": "assistant", "content": assistant_content}]
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        full_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        # Find character positions of reasoning and completion in full_text
        reason_char_start = full_text.index(initial_reasoning)
        reason_char_end = reason_char_start + len(initial_reasoning)

        completion_char_start = full_text.index(
            f"{answer_prefix} {completion}", reason_char_end
        )
        # Skip past answer_prefix + space to get actual completion start
        answer_char_start = completion_char_start
        answer_char_end = answer_char_start + len(f"{answer_prefix} {completion}")

        # Map character positions -> token positions
        reasoning_start = reasoning_end = answer_start = answer_end = None
        for i, (cs, ce) in enumerate(offsets):
            if reasoning_start is None and ce > reason_char_start:
                reasoning_start = i
            if ce > 0 and cs < reason_char_end:
                reasoning_end = i
            if answer_start is None and ce > answer_char_start:
                answer_start = i
            if ce > 0 and cs < answer_char_end:
                answer_end = i

        full_ids_t = torch.tensor([full_ids], device=self.device)

        print(f"  Sequence length: {len(full_ids)}")
        print(f"  Reasoning tokens: {reasoning_start}-{reasoning_end} "
              f"({reasoning_end - reasoning_start + 1} tokens)")
        print(f"  Answer tokens: {answer_start}-{answer_end} "
              f"({answer_end - answer_start + 1} tokens)")

        return full_ids_t, reasoning_start, reasoning_end, answer_start, answer_end

    def compute_probe_score(
        self, full_ids: torch.Tensor, answer_start: int, answer_end: int,
        grad_mode: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass + probe scoring. Returns mean confidence across layers.
        If grad_mode, returns a differentiable scalar. Otherwise detached.
        """
        hook_outputs = {}
        hooks = []

        def make_hook(layer_idx):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if not grad_mode:
                    h = h.detach()
                hook_outputs[layer_idx] = h
            return fn

        for layer_idx in self.target_layers:
            layer_module = self.model.model.layers[layer_idx]
            hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

        try:
            if grad_mode:
                self.model(inputs_embeds=full_ids)
            else:
                with torch.no_grad():
                    self.model(input_ids=full_ids)

            batch_size = full_ids.shape[0]
            layer_scores = []

            for layer_idx in self.target_layers:
                h = hook_outputs[layer_idx].float()  # (B, seq_len, hidden_dim)
                # Mean pool over answer tokens
                pooled = h[:, answer_start:answer_end + 1, :].mean(dim=1)  # (B, hidden_dim)

                lp = self.layer_probes[layer_idx]
                if lp["scaler_mean"] is not None:
                    pooled = (pooled - lp["scaler_mean"]) / lp["scaler_scale"]

                # Ensemble: mean logit across seed probes
                logits = torch.stack([p(pooled) for p in lp["probes"]])  # (n_seeds, B)
                mean_logit = logits.mean(dim=0)  # (B,)
                layer_scores.append(mean_logit)

            # Mean confidence across layers
            stacked = torch.stack(layer_scores)  # (n_layers, B)
            mean_logit = stacked.mean(dim=0)  # (B,)
            confidence = torch.sigmoid(mean_logit)  # (B,)
            return confidence

        finally:
            for hook in hooks:
                hook.remove()

    def get_token_gradients(
        self, full_ids: torch.Tensor,
        reasoning_start: int, reasoning_end: int,
        answer_start: int, answer_end: int,
    ) -> torch.Tensor:
        """
        Compute gradient of probe score w.r.t. reasoning token embeddings.
        Returns: gradient tensor (n_reasoning_tokens, hidden_dim).
        """
        # Get embeddings for full sequence
        embeds = self.model.model.embed_tokens(full_ids)  # (1, seq_len, hidden_dim)
        embeds = embeds.detach().clone().requires_grad_(True)

        # Forward pass through the model using embeddings
        confidence = self.compute_probe_score(
            embeds, answer_start, answer_end, grad_mode=True
        )
        confidence.sum().backward()

        # Extract gradient at reasoning positions
        grad = embeds.grad[0, reasoning_start:reasoning_end + 1, :].clone()
        return grad  # (n_reasoning_tokens, hidden_dim)

    def get_top_k_candidates(
        self, grad: torch.Tensor, top_k: int = 256,
        target: str = "minimize",
    ) -> torch.Tensor:
        """
        For each reasoning position, find top-k vocabulary tokens that would
        change the probe score in the desired direction.

        Returns: (n_reasoning_tokens, top_k) tensor of token ids.
        """
        # For minimization: we want tokens whose embeddings point in -gradient direction
        # score[pos, vocab] = -grad[pos] @ embed_weights[vocab]
        # For maximization: score = +grad @ embed_weights
        sign = -1.0 if target == "minimize" else 1.0

        # embed_weights: (vocab_size, hidden_dim), might be quantized
        # Dequantize if needed
        embed_w = self.embed_weights
        if hasattr(embed_w, "data"):
            embed_w = embed_w.data
        embed_w = embed_w.float()  # (vocab_size, hidden_dim)

        # score = sign * grad @ embed_w^T  -> (n_positions, vocab_size)
        scores = sign * (grad.float() @ embed_w.T)

        # Apply ASCII constraint: mask out non-English tokens
        if self.constrained and self.ascii_mask is not None:
            scores[:, ~self.ascii_mask] = -float("inf")

        # Get top-k per position
        _, top_indices = scores.topk(top_k, dim=-1)
        return top_indices  # (n_positions, top_k)

    def evaluate_candidates(
        self, base_ids: torch.Tensor,
        reasoning_start: int, reasoning_end: int,
        answer_start: int, answer_end: int,
        candidates: List[Tuple[int, int]],
        eval_batch_size: int = 16,
    ) -> List[Tuple[float, int, int]]:
        """
        Evaluate candidate single-token substitutions.
        candidates: list of (position_offset, token_id) where position_offset
                    is relative to reasoning_start.
        Returns: list of (score, position_offset, token_id) sorted by score.
        """
        results = []
        n_reasoning = reasoning_end - reasoning_start + 1

        # Process in mini-batches
        for batch_start in range(0, len(candidates), eval_batch_size):
            batch_cands = candidates[batch_start:batch_start + eval_batch_size]
            B = len(batch_cands)

            # Build batch: each element has one token substituted
            batch_ids = base_ids.expand(B, -1).clone()  # (B, seq_len)
            for i, (pos_offset, tok_id) in enumerate(batch_cands):
                batch_ids[i, reasoning_start + pos_offset] = tok_id

            # Forward pass
            scores = self.compute_probe_score(
                batch_ids, answer_start, answer_end, grad_mode=False
            )

            for i, (pos_offset, tok_id) in enumerate(batch_cands):
                results.append((scores[i].item(), pos_offset, tok_id))

        results.sort(key=lambda x: x[0])
        return results

    def optimize(
        self,
        scenario: str, question: str, answer_prefix: str,
        initial_reasoning: str, completion: str,
        n_steps: int = 200,
        batch_size: int = 64,
        top_k: int = 256,
        eval_batch_size: int = 16,
        target: str = "minimize",
        output_dir: str = None,
    ) -> dict:
        """Run GCG optimization loop."""
        print(f"\n{'='*60}")
        print(f"GCG Reasoning Optimizer — target={target}")
        print(f"{'='*60}")

        # Prepare sequence
        full_ids, r_start, r_end, a_start, a_end = self.prepare_sequence(
            scenario, question, answer_prefix, initial_reasoning, completion
        )
        n_reasoning = r_end - r_start + 1

        # Initial score
        initial_score = self.compute_probe_score(
            full_ids, a_start, a_end, grad_mode=False
        ).item()
        print(f"\n  Initial probe confidence: {initial_score:.4f}")

        # Track progress: score every step, full reasoning text every 50 steps
        history = [{
            "step": 0,
            "score": initial_score,
            "reasoning": self.tokenizer.decode(
                full_ids[0, r_start:r_end + 1].tolist(),
                skip_special_tokens=False
            ),
        }]
        score_curve = [{"step": 0, "score": initial_score}]

        best_score = initial_score
        best_ids = full_ids.clone()
        current_ids = full_ids.clone()

        print(f"\n  Starting optimization: {n_steps} steps, "
              f"batch_size={batch_size}, top_k={top_k}")
        print(f"  Optimizing {n_reasoning} reasoning tokens\n")

        t_start = time.time()

        for step in range(1, n_steps + 1):
            t_step = time.time()

            # Step 1: Get gradients
            grad = self.get_token_gradients(
                current_ids, r_start, r_end, a_start, a_end
            )

            # Step 2: Get top-k candidates per position
            top_k_tokens = self.get_top_k_candidates(grad, top_k, target)
            # (n_reasoning, top_k)

            # Step 3: Sample random (position, token) pairs
            candidates = []
            for _ in range(batch_size):
                pos = torch.randint(0, n_reasoning, (1,)).item()
                tok_idx = torch.randint(0, top_k, (1,)).item()
                tok_id = top_k_tokens[pos, tok_idx].item()
                candidates.append((pos, tok_id))

            # Step 4: Evaluate candidates
            scored = self.evaluate_candidates(
                current_ids, r_start, r_end, a_start, a_end,
                candidates, eval_batch_size,
            )

            # Step 5: Accept best if it improves
            if target == "minimize":
                best_cand_score, best_pos, best_tok = scored[0]
                improved = best_cand_score < best_score
            else:
                best_cand_score, best_pos, best_tok = scored[-1]
                improved = best_cand_score > best_score

            if improved:
                current_ids[0, r_start + best_pos] = best_tok
                best_score = best_cand_score
                best_ids = current_ids.clone()

            step_time = time.time() - t_step

            # Log
            current_score = self.compute_probe_score(
                current_ids, a_start, a_end, grad_mode=False
            ).item()

            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t_start
                reasoning_text = self.tokenizer.decode(
                    current_ids[0, r_start:r_end + 1].tolist(),
                    skip_special_tokens=False
                )
                # Show first 100 chars
                preview = reasoning_text[:100].replace("\n", "\\n")
                print(f"  Step {step:>4d} | score={current_score:.4f} | "
                      f"best={best_score:.4f} | "
                      f"{'improved' if improved else '        '} | "
                      f"{step_time:.1f}s | "
                      f"reasoning: {preview}...")

            score_curve.append({"step": step, "score": current_score})

            if step % 50 == 0:
                history.append({
                    "step": step,
                    "score": current_score,
                    "reasoning": self.tokenizer.decode(
                        current_ids[0, r_start:r_end + 1].tolist(),
                        skip_special_tokens=False
                    ),
                })

        # Final
        total_time = time.time() - t_start
        final_score = self.compute_probe_score(
            best_ids, a_start, a_end, grad_mode=False
        ).item()

        final_reasoning = self.tokenizer.decode(
            best_ids[0, r_start:r_end + 1].tolist(),
            skip_special_tokens=False
        )

        history.append({
            "step": n_steps,
            "score": final_score,
            "reasoning": final_reasoning,
        })

        result = {
            "initial_score": initial_score,
            "final_score": final_score,
            "delta": final_score - initial_score,
            "n_steps": n_steps,
            "target": target,
            "total_time_s": round(total_time, 1),
            "initial_reasoning": initial_reasoning,
            "final_reasoning": final_reasoning,
            "history": history,
            "score_curve": score_curve,
        }

        print(f"\n{'='*60}")
        print(f"DONE in {total_time:.0f}s")
        print(f"  Initial: {initial_score:.4f}")
        print(f"  Final:   {final_score:.4f}")
        print(f"  Delta:   {final_score - initial_score:+.4f}")
        print(f"\nFinal reasoning (first 200 chars):")
        print(f"  {final_reasoning[:200]}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "gcg_result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to {out_path}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="GCG discrete optimization of reasoning trace tokens"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--scenarios", default=None,
                        help="Scenarios YAML (default: config data_dir/scenarios.yaml)")
    parser.add_argument("--scenario-idx", type=int, default=None,
                        help="Single scenario index (use this OR --scenario-indices)")
    parser.add_argument("--scenario-indices", type=int, nargs="+", default=None,
                        help="Multiple scenario indices to optimize")
    parser.add_argument("--initial-reasoning", default=None,
                        help="Path to traces YAML for warm start")
    parser.add_argument("--init-condition", default="moralizing",
                        help="Which trace condition to use as warm start (default: moralizing)")
    parser.add_argument("--n-reasoning-tokens", type=int, default=100,
                        help="Number of reasoning tokens (only for random init)")
    parser.add_argument("--constrained", action="store_true",
                        help="Restrict to ASCII/English tokens only")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Candidates to evaluate per step")
    parser.add_argument("--top-k", type=int, default=256,
                        help="Top-k tokens per position from gradient")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="Forward pass batch size for candidate evaluation")
    parser.add_argument("--target", choices=["minimize", "maximize"], default="minimize",
                        help="Minimize (evade) or maximize (amplify) probe score")
    parser.add_argument("--position", default="answer_mean_pool")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    def resolve(p):
        path = Path(p)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    config = load_config(resolve(args.config))
    model_config = load_config(resolve(config["model_config"]))

    target_layers = config["target_layers"]
    seeds = config["probe_training"]["random_seeds"]
    hidden_dim = model_config["hidden_dim"]

    output_dir = args.output_dir or os.path.join(
        config["storage"]["base_dir"], "gcg_optimization"
    )
    log_dir = args.log_dir or os.path.join(output_dir, "logs")
    setup_logging(log_dir, "gcg_optimizer")

    # Load scenarios
    data_dir = resolve(config["data_dir"])
    scenarios_path = resolve(args.scenarios) if args.scenarios else os.path.join(data_dir, "scenarios.yaml")
    scenarios = load_scenarios(scenarios_path)

    # Determine which scenarios to run
    if args.scenario_indices is not None:
        run_indices = args.scenario_indices
    elif args.scenario_idx is not None:
        run_indices = [args.scenario_idx]
    else:
        run_indices = list(range(len(scenarios)))

    print(f"Scenarios to optimize: {run_indices}")
    print(f"Constrained (ASCII only): {args.constrained}")

    # Load traces for warm start
    traces_by_pair = {}
    if args.initial_reasoning:
        traces_path = resolve(args.initial_reasoning)
        with open(traces_path) as f:
            traces_list = yaml.safe_load(f)["traces"]
        traces_by_pair = {t["pair_id"]: t for t in traces_list}

    # Load model
    print(f"\nLoading model: {model_config['model_id']}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    device = next(model.parameters()).device
    print(f"  Loaded in {time.time() - t0:.1f}s on {device}")

    # Load probes
    print(f"\nLoading probes from {args.probe_dir}")
    layer_probes = load_probes_and_scalers(
        args.probe_dir, args.position, target_layers, seeds, hidden_dim, device
    )
    print(f"  Loaded probes for {len(layer_probes)} layers")

    # Build optimizer (once — reused across scenarios)
    optimizer = GCGReasoningOptimizer(
        model, tokenizer, layer_probes, target_layers, device,
        constrained=args.constrained,
    )

    # Run optimization for each scenario
    all_results = []
    for idx in run_indices:
        scenario = scenarios[idx]
        pair_id = scenario["pair_id"]
        category = scenario["category"]

        print(f"\n{'#'*60}")
        print(f"# Scenario {idx} (pair_id={pair_id}, {category})")
        print(f"#   Q: {scenario['question'][:70]}...")
        print(f"#   Deceptive: {scenario['deceptive_completion'][:70]}...")
        print(f"{'#'*60}")

        # Get initial reasoning
        if traces_by_pair and pair_id in traces_by_pair:
            trace = traces_by_pair[pair_id]
            cond = args.init_condition
            initial_reasoning = trace.get(cond, "")
            if not initial_reasoning:
                # Fallback: try first available condition
                available = [k for k in trace if k not in ("pair_id", "category")]
                cond = available[0] if available else None
                initial_reasoning = trace.get(cond, "") if cond else ""
            if initial_reasoning:
                print(f"  Warm start: {cond} ({len(initial_reasoning)} chars)")
            else:
                print(f"  WARNING: no trace found for pair_id={pair_id}, using random init")
        else:
            initial_reasoning = ""

        if not initial_reasoning:
            vocab_size = len(tokenizer)
            random_ids = torch.randint(100, vocab_size - 100, (args.n_reasoning_tokens,))
            initial_reasoning = tokenizer.decode(random_ids.tolist(), skip_special_tokens=False)
            print(f"  Cold start: {args.n_reasoning_tokens} random tokens")

        # Per-run output
        constraint_tag = "constrained" if args.constrained else "unconstrained"
        run_dir = os.path.join(output_dir, f"scenario_{pair_id}_{constraint_tag}")

        result = optimizer.optimize(
            scenario=scenario["scenario"],
            question=scenario["question"],
            answer_prefix=scenario["answer_prefix"],
            initial_reasoning=initial_reasoning,
            completion=scenario["deceptive_completion"],
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            top_k=args.top_k,
            eval_batch_size=args.eval_batch_size,
            target=args.target,
            output_dir=run_dir,
        )
        result["scenario_idx"] = idx
        result["pair_id"] = pair_id
        result["category"] = category
        result["constrained"] = args.constrained
        result["init_condition"] = args.init_condition
        all_results.append(result)

        # Clear GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} runs, {constraint_tag})")
    print(f"{'='*60}")
    print(f"{'Scenario':<12} {'Category':<12} {'Initial':>8} {'Final':>8} {'Delta':>8}")
    print("-" * 52)
    for r in all_results:
        print(f"pair_{r['pair_id']:<7} {r['category']:<12} "
              f"{r['initial_score']:>8.4f} {r['final_score']:>8.4f} "
              f"{r['delta']:>+8.4f}")

    # Save summary
    summary_path = os.path.join(output_dir, f"gcg_summary_{constraint_tag}.json")
    os.makedirs(output_dir, exist_ok=True)
    summary = [{
        "scenario_idx": r["scenario_idx"],
        "pair_id": r["pair_id"],
        "category": r["category"],
        "initial_score": r["initial_score"],
        "final_score": r["final_score"],
        "delta": r["delta"],
        "n_steps": r["n_steps"],
        "total_time_s": r["total_time_s"],
    } for r in all_results]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
