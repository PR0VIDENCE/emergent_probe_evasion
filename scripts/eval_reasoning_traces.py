"""
Evaluate probes on reasoning trace activations. Minimal script.

Loads activations from extract_reasoning_trace_activations.py output,
scores with trained probes, reports per-condition results.

Usage:
    uv run python scripts/eval_reasoning_traces.py --probe-dir /workspace/deception_prewritten_data/probes --data-dir /workspace/reasoning_traces_claude --config configs/experiments/deception_prewritten.yaml
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_ensemble


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_probes(probe_dir, position, layer_idx, seeds, hidden_dim):
    probes = []
    pos_dir = os.path.join(probe_dir, position)
    for seed in seeds:
        p = LinearProbe(hidden_dim)
        p.load_state_dict(torch.load(os.path.join(pos_dir, f"layer{layer_idx}_seed{seed}.pt"), weights_only=True))
        p.eval()
        probes.append(p)
    scaler_path = os.path.join(pos_dir, f"layer{layer_idx}_scaler.pt")
    scaler = torch.load(scaler_path, weights_only=True) if os.path.exists(scaler_path) else {}
    return probes, scaler.get("scaler_mean"), scaler.get("scaler_scale")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--position", default="answer_mean_pool",
                        help="Which activation position to evaluate")
    parser.add_argument("--probe-position", default=None,
                        help="Which probe directory to load from (default: same as --position)")
    args = parser.parse_args()
    if args.probe_position is None:
        args.probe_position = args.position

    def resolve(p):
        path = Path(p)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    config = load_config(resolve(args.config))
    model_config = load_config(resolve(config["model_config"]))
    seeds = config["probe_training"]["random_seeds"]
    hidden_dim = model_config["hidden_dim"]
    target_layers = config["target_layers"]

    # Scan generation JSONs to group prompt_ids by condition
    condition_prompts = defaultdict(list)  # condition -> [(prompt_id, label)]
    for label in ["tree", "non_tree"]:
        gen_dir = os.path.join(args.data_dir, "generations", label)
        if not os.path.isdir(gen_dir):
            continue
        for fn in sorted(os.listdir(gen_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(gen_dir, fn)) as f:
                gen = json.load(f)
            condition = gen.get("condition", "unknown")
            condition_prompts[condition].append((gen["prompt_id"], label))

    print(f"Conditions found: {list(condition_prompts.keys())}")
    for c, ps in condition_prompts.items():
        print(f"  {c}: {len(ps)} prompts")

    # Evaluate per condition x layer
    print(f"\nPosition: {args.position}")
    print(f"Layers: {target_layers}")
    print()

    # Load all generation JSONs for detailed logging
    all_gen_data = {}
    for label in ["tree", "non_tree"]:
        gen_dir = os.path.join(args.data_dir, "generations", label)
        if not os.path.isdir(gen_dir):
            continue
        for fn in sorted(os.listdir(gen_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(gen_dir, fn)) as f:
                gen = json.load(f)
            all_gen_data[gen["prompt_id"]] = gen

    # Pre-load all probes and scalers (once per layer, not per trial)
    print("Loading probes...")
    probe_cache = {}
    for layer_idx in target_layers:
        try:
            probes, sm, ss = load_probes(args.probe_dir, args.probe_position, layer_idx, seeds, hidden_dim)
            probe_cache[layer_idx] = (probes, sm, ss)
        except FileNotFoundError:
            pass
    print(f"  Loaded probes for {len(probe_cache)} layers")

    results = {}
    trial_details = []  # per-trial log

    for condition, prompts in sorted(condition_prompts.items()):
        layer_confs = defaultdict(list)
        layer_votes = defaultdict(list)

        for prompt_id, label in prompts:
            act_path = os.path.join(args.data_dir, "activations", label, args.position, f"{prompt_id}.pt")
            if not os.path.exists(act_path):
                continue
            layer_acts = torch.load(act_path, weights_only=True)

            trial_layer_confs = {}
            for layer_idx in target_layers:
                if layer_idx not in layer_acts or layer_idx not in probe_cache:
                    continue
                probes, sm, ss = probe_cache[layer_idx]
                act = layer_acts[layer_idx].float().unsqueeze(0)
                result = evaluate_ensemble(probes, act, None, scaler_mean=sm, scaler_scale=ss)
                conf = float(result["mean_confidence"].item())
                vote = int(result["majority_vote"].item())
                layer_confs[layer_idx].append(conf)
                layer_votes[layer_idx].append(vote)
                trial_layer_confs[layer_idx] = conf

            # Per-trial mean confidence across all layers
            trial_mean_conf = np.mean(list(trial_layer_confs.values())) if trial_layer_confs else 0.0

            # Build trial detail entry
            gen = all_gen_data.get(prompt_id, {})
            trial_details.append({
                "prompt_id": prompt_id,
                "condition": condition,
                "label": label,
                "pair_id": gen.get("global_pair_id"),
                "category": gen.get("category", ""),
                "question": gen.get("question", ""),
                "reasoning_trace": gen.get("reasoning_trace", ""),
                "answer": gen.get("answer", ""),
                "mean_confidence": round(trial_mean_conf, 4),
                "per_layer_confidence": {int(l): round(c, 4) for l, c in trial_layer_confs.items()},
            })

        # Aggregate: mean confidence and detection rate per layer, then average top-5
        layer_means = {l: np.mean(c) for l, c in layer_confs.items()}
        layer_det = {l: np.mean(v) for l, v in layer_votes.items()}
        top5 = sorted(layer_means.keys(), key=lambda l: layer_means[l], reverse=True)[:5]

        avg_conf = np.mean([layer_means[l] for l in top5]) if top5 else 0
        avg_det = np.mean([layer_det[l] for l in top5]) if top5 else 0
        n = len(prompts)

        results[condition] = {"confidence": avg_conf, "detection_rate": avg_det, "n": n, "top5": top5}

    # Print summary table
    print(f"\n{'Condition':<25} {'Confidence':>12} {'Det. Rate':>12} {'N':>6}")
    print("-" * 58)
    for condition, r in sorted(results.items()):
        print(f"{condition:<25} {r['confidence']:>12.3f} {r['detection_rate']:>12.3f} {r['n']:>6}")

    # Save summary JSON
    save_results = {
        condition: {
            "confidence": float(r["confidence"]),
            "detection_rate": float(r["detection_rate"]),
            "n": r["n"],
            "top5_layers": [int(l) for l in r["top5"]],
        }
        for condition, r in results.items()
    }
    report_path = os.path.join(args.data_dir, "reasoning_trace_eval.json")
    with open(report_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSummary saved to {report_path}")

    # Save detailed per-trial log
    detail_path = os.path.join(args.data_dir, "reasoning_trace_trials.jsonl")
    with open(detail_path, "w") as f:
        for trial in trial_details:
            f.write(json.dumps(trial) + "\n")
    print(f"Trial details saved to {detail_path} ({len(trial_details)} trials)")


if __name__ == "__main__":
    main()
