"""
BPJ-inspired denoising curriculum for reasoning trace optimization.

Starts from a GCG-optimized gibberish reasoning trace (low probe score)
and progressively replaces adversarial tokens with fluent alternatives.
Uses the model's own next-token predictions for fluency and greedy
denoising ordered by least probe-score increase.

Produces a Pareto frontier: (fraction fluent tokens) vs (probe score).
This shows exactly how much fluency you can recover while maintaining
probe evasion.

Usage:
    uv run python scripts/bpj_denoise_optimizer.py \
        --config configs/experiments/deception_neutral.yaml \
        --probe-dir /workspace/deception_prewritten_data/probes \
        --gcg-result /workspace/deception_neutral_data/gcg/scenario_0_constrained/gcg_result.json \
        --scenario-idx 0 \
        --output-dir /workspace/deception_neutral_data/bpj
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

from src.inference.extract_activations import load_model_and_tokenizer
from src.probes.architectures import LinearProbe


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_scenarios(path):
    with open(path) as f:
        return yaml.safe_load(f)["scenarios"]


def load_probes_and_scalers(probe_dir, position, layers, seeds, hidden_dim, device):
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


class BPJDenoiser:
    def __init__(self, model, tokenizer, layer_probes, target_layers, device):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_probes = layer_probes
        self.target_layers = target_layers
        self.device = device

    def build_sequence(
        self, scenario: str, question: str, answer_prefix: str,
        reasoning_text: str, completion: str,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """Build token sequence and find reasoning/answer boundaries."""
        messages = [
            {"role": "system", "content": scenario},
            {"role": "user", "content": question},
        ]
        assistant_content = (
            f"<think>\n{reasoning_text}\n</think>\n{answer_prefix} {completion}"
        )
        full_messages = messages + [{"role": "assistant", "content": assistant_content}]
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        encoding = self.tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        full_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        reason_char_start = full_text.index(reasoning_text)
        reason_char_end = reason_char_start + len(reasoning_text)
        answer_text = f"{answer_prefix} {completion}"
        answer_char_start = full_text.index(answer_text, reason_char_end)
        answer_char_end = answer_char_start + len(answer_text)

        r_start = r_end = a_start = a_end = None
        for i, (cs, ce) in enumerate(offsets):
            if r_start is None and ce > reason_char_start:
                r_start = i
            if ce > 0 and cs < reason_char_end:
                r_end = i
            if a_start is None and ce > answer_char_start:
                a_start = i
            if ce > 0 and cs < answer_char_end:
                a_end = i

        return torch.tensor([full_ids], device=self.device), r_start, r_end, a_start, a_end

    def compute_probe_score(self, full_ids, answer_start, answer_end) -> float:
        """Single forward pass, return probe confidence."""
        hook_outputs = {}
        hooks = []

        for layer_idx in self.target_layers:
            def make_hook(li):
                def fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hook_outputs[li] = h.detach()
                return fn
            hooks.append(
                self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            )

        try:
            with torch.no_grad():
                self.model(input_ids=full_ids)

            layer_scores = []
            for layer_idx in self.target_layers:
                h = hook_outputs[layer_idx].float()
                pooled = h[0, answer_start:answer_end + 1, :].mean(dim=0).unsqueeze(0)
                lp = self.layer_probes[layer_idx]
                if lp["scaler_mean"] is not None:
                    pooled = (pooled - lp["scaler_mean"]) / lp["scaler_scale"]
                logits = torch.stack([p(pooled).squeeze() for p in lp["probes"]])
                layer_scores.append(logits.mean())

            return torch.sigmoid(torch.stack(layer_scores).mean()).item()
        finally:
            for hook in hooks:
                hook.remove()

    def get_fluent_predictions(
        self, full_ids: torch.Tensor,
        reasoning_start: int, reasoning_end: int,
        top_p: float = 0.95, max_candidates: int = 50,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get model's own next-token predictions at each reasoning position.
        Returns: {position_offset: [(token_id, log_prob), ...]} sorted by prob.
        """
        with torch.no_grad():
            output = self.model(input_ids=full_ids)
        logits = output.logits[0].float()  # (seq_len, vocab_size)

        predictions = {}
        for pos in range(reasoning_start, reasoning_end + 1):
            # logits[pos-1] predicts token at pos
            pos_logits = logits[pos - 1]
            probs = torch.softmax(pos_logits, dim=-1)
            sorted_probs, sorted_ids = probs.sort(descending=True)

            # Collect top-p tokens
            cumsum = sorted_probs.cumsum(dim=0)
            cutoff = (cumsum <= top_p).sum().item() + 1
            cutoff = min(cutoff, max_candidates)

            candidates = []
            for j in range(cutoff):
                tok_id = sorted_ids[j].item()
                log_p = sorted_probs[j].log().item()
                candidates.append((tok_id, log_p))

            predictions[pos - reasoning_start] = candidates

        return predictions

    def classify_tokens(
        self, full_ids: torch.Tensor,
        reasoning_start: int, reasoning_end: int,
        fluent_predictions: Dict[int, List[Tuple[int, float]]],
        top_p_threshold: float = 0.95,
    ) -> Tuple[List[int], List[int]]:
        """
        Classify reasoning tokens as fluent or adversarial.
        A token is 'fluent' if it's in the model's top-p predictions for that position.
        Returns: (fluent_offsets, adversarial_offsets)
        """
        fluent = []
        adversarial = []

        for offset in range(reasoning_end - reasoning_start + 1):
            current_tok = full_ids[0, reasoning_start + offset].item()
            preds = fluent_predictions.get(offset, [])
            pred_ids = {tok_id for tok_id, _ in preds}

            if current_tok in pred_ids:
                fluent.append(offset)
            else:
                adversarial.append(offset)

        return fluent, adversarial

    def denoise(
        self,
        scenario: dict,
        gcg_reasoning: str,
        gcg_score: float,
        n_candidates_per_position: int = 10,
        score_budget: float = 0.85,
        output_dir: str = None,
    ) -> dict:
        """
        Greedy denoising curriculum.

        For each adversarial token (ordered by least score increase),
        replace with the best fluent alternative. Stop when probe score
        exceeds score_budget or all positions are denoised.
        """
        print(f"\n{'='*60}")
        print(f"BPJ Denoising Curriculum")
        print(f"{'='*60}")

        # Build initial sequence from GCG result
        full_ids, r_start, r_end, a_start, a_end = self.build_sequence(
            scenario["scenario"], scenario["question"],
            scenario["answer_prefix"], gcg_reasoning,
            scenario["deceptive_completion"],
        )
        n_reasoning = r_end - r_start + 1

        # Verify GCG score
        current_score = self.compute_probe_score(full_ids, a_start, a_end)
        print(f"  GCG starting score: {current_score:.4f} (reported: {gcg_score:.4f})")
        print(f"  Reasoning tokens: {n_reasoning}")
        print(f"  Score budget: {score_budget}")

        # Get model's fluent predictions for each position
        print(f"\n  Getting model predictions for fluency...")
        fluent_preds = self.get_fluent_predictions(full_ids, r_start, r_end)

        # Classify tokens
        fluent_offsets, adv_offsets = self.classify_tokens(
            full_ids, r_start, r_end, fluent_preds
        )
        print(f"  Fluent tokens: {len(fluent_offsets)}/{n_reasoning}")
        print(f"  Adversarial tokens: {len(adv_offsets)}/{n_reasoning}")

        # Phase 1: Score the cost of replacing each adversarial token
        # For each adversarial position, try top fluent candidates and record
        # the minimum probe score increase
        print(f"\n  Phase 1: Estimating replacement costs for {len(adv_offsets)} positions...")
        replacement_costs = []  # (offset, best_replacement_tok, score_after_replacement)

        current_ids = full_ids.clone()
        t0 = time.time()

        for i, offset in enumerate(adv_offsets):
            abs_pos = r_start + offset
            original_tok = current_ids[0, abs_pos].item()

            # Get top fluent candidates for this position
            candidates = fluent_preds.get(offset, [])[:n_candidates_per_position]
            if not candidates:
                continue

            best_replacement = None
            best_score = float("inf")

            for tok_id, log_p in candidates:
                # Temporarily substitute
                current_ids[0, abs_pos] = tok_id
                score = self.compute_probe_score(current_ids, a_start, a_end)
                if score < best_score:
                    best_score = score
                    best_replacement = tok_id
                # Restore
                current_ids[0, abs_pos] = original_tok

            if best_replacement is not None:
                replacement_costs.append((offset, best_replacement, best_score))

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"    Scored {i+1}/{len(adv_offsets)} positions ({elapsed:.0f}s)")

        # Sort by score increase (cheapest replacements first)
        replacement_costs.sort(key=lambda x: x[2])

        print(f"  Scored {len(replacement_costs)} replaceable positions")
        if replacement_costs:
            print(f"  Cheapest replacement score: {replacement_costs[0][2]:.4f}")
            print(f"  Most expensive: {replacement_costs[-1][2]:.4f}")

        # Phase 2: Greedy denoising — apply replacements in order of cost
        print(f"\n  Phase 2: Greedy denoising...")
        current_ids = full_ids.clone()
        current_score = self.compute_probe_score(current_ids, a_start, a_end)

        pareto_frontier = [{
            "step": 0,
            "n_denoised": 0,
            "n_adversarial": len(adv_offsets),
            "frac_fluent": len(fluent_offsets) / n_reasoning,
            "score": current_score,
            "reasoning": self.tokenizer.decode(
                current_ids[0, r_start:r_end + 1].tolist(),
                skip_special_tokens=False,
            ),
        }]

        n_denoised = 0
        for offset, replacement_tok, estimated_score in replacement_costs:
            abs_pos = r_start + offset
            old_tok = current_ids[0, abs_pos].item()

            # Apply replacement
            current_ids[0, abs_pos] = replacement_tok
            actual_score = self.compute_probe_score(current_ids, a_start, a_end)
            n_denoised += 1

            frac_fluent = (len(fluent_offsets) + n_denoised) / n_reasoning

            old_text = self.tokenizer.decode([old_tok])
            new_text = self.tokenizer.decode([replacement_tok])

            if n_denoised % 10 == 0 or n_denoised <= 5 or actual_score > score_budget:
                print(f"    Step {n_denoised:>3d} | score={actual_score:.4f} | "
                      f"fluent={frac_fluent:.1%} | "
                      f"'{old_text}' -> '{new_text}'")

            pareto_frontier.append({
                "step": n_denoised,
                "n_denoised": n_denoised,
                "n_adversarial": len(adv_offsets) - n_denoised,
                "frac_fluent": frac_fluent,
                "score": actual_score,
                "reasoning": self.tokenizer.decode(
                    current_ids[0, r_start:r_end + 1].tolist(),
                    skip_special_tokens=False,
                ) if n_denoised % 10 == 0 or actual_score > score_budget else None,
            })

            if actual_score > score_budget:
                print(f"\n  Score budget exceeded ({actual_score:.4f} > {score_budget}). "
                      f"Reverting last replacement.")
                current_ids[0, abs_pos] = old_tok
                n_denoised -= 1
                pareto_frontier[-1]["reverted"] = True
                break

        # Final state
        final_score = self.compute_probe_score(current_ids, a_start, a_end)
        final_reasoning = self.tokenizer.decode(
            current_ids[0, r_start:r_end + 1].tolist(),
            skip_special_tokens=False,
        )
        final_frac = (len(fluent_offsets) + n_denoised) / n_reasoning

        print(f"\n{'='*60}")
        print(f"DENOISING COMPLETE")
        print(f"  Starting score: {gcg_score:.4f}")
        print(f"  Final score:    {final_score:.4f}")
        print(f"  Fluent tokens:  {final_frac:.1%} ({len(fluent_offsets) + n_denoised}/{n_reasoning})")
        print(f"  Denoised:       {n_denoised}/{len(adv_offsets)} adversarial tokens")
        print(f"\nFinal reasoning (first 300 chars):")
        print(f"  {final_reasoning[:300]}")

        result = {
            "gcg_score": gcg_score,
            "final_score": final_score,
            "n_reasoning_tokens": n_reasoning,
            "n_initially_fluent": len(fluent_offsets),
            "n_adversarial": len(adv_offsets),
            "n_denoised": n_denoised,
            "final_frac_fluent": final_frac,
            "score_budget": score_budget,
            "gcg_reasoning": gcg_reasoning,
            "final_reasoning": final_reasoning,
            "pareto_frontier": pareto_frontier,
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "bpj_result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to {out_path}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="BPJ-inspired denoising of GCG reasoning traces"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--gcg-result", required=True,
                        help="Path to gcg_result.json from GCG optimizer")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--scenario-idx", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=10,
                        help="Fluent candidates to try per adversarial position")
    parser.add_argument("--score-budget", type=float, default=0.85,
                        help="Stop denoising when probe score exceeds this")
    parser.add_argument("--position", default="answer_mean_pool")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    def resolve(p):
        path = Path(p)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    config = load_config(resolve(args.config))
    model_config = load_config(resolve(config["model_config"]))
    target_layers = config["target_layers"]
    seeds = config["probe_training"]["random_seeds"]
    hidden_dim = model_config["hidden_dim"]

    # Load GCG result
    with open(resolve(args.gcg_result)) as f:
        gcg = json.load(f)
    gcg_reasoning = gcg["final_reasoning"]
    gcg_score = gcg["final_score"]
    print(f"GCG result: score={gcg_score:.4f}, {len(gcg_reasoning)} chars")

    # Load scenario
    data_dir = resolve(config["data_dir"])
    scenarios_path = resolve(args.scenarios) if args.scenarios else os.path.join(data_dir, "scenarios.yaml")
    scenarios = load_scenarios(scenarios_path)
    scenario = scenarios[args.scenario_idx]
    print(f"Scenario {args.scenario_idx}: {scenario['category']}")

    # Load model
    print(f"\nLoading model: {model_config['model_id']}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    device = next(model.parameters()).device
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load probes
    layer_probes = load_probes_and_scalers(
        args.probe_dir, args.position, target_layers, seeds, hidden_dim, device
    )
    print(f"  Loaded probes for {len(layer_probes)} layers")

    output_dir = args.output_dir or os.path.join(
        config["storage"]["base_dir"], "bpj_denoise"
    )

    denoiser = BPJDenoiser(model, tokenizer, layer_probes, target_layers, device)
    result = denoiser.denoise(
        scenario=scenario,
        gcg_reasoning=gcg_reasoning,
        gcg_score=gcg_score,
        n_candidates_per_position=args.n_candidates,
        score_budget=args.score_budget,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
