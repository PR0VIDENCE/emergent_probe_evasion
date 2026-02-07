"""
Evaluate trained probes on an adversarial test set.

Usage:
    python scripts/eval_adversarial.py \
        --model-config configs/models/qwen2_5_32b.yaml \
        --probe-config configs/probes/trees.yaml \
        --adversarial-data data/concepts/trees/adversarial_test.yaml \
        --probes-dir data/outputs/preliminary/probes \
        --output-dir data/outputs/adversarial

With pre-extracted activations (skip model loading):
    python scripts/eval_adversarial.py \
        --probe-config configs/probes/trees.yaml \
        --adversarial-data data/concepts/trees/adversarial_test.yaml \
        --probes-dir data/outputs/preliminary/probes \
        --output-dir data/outputs/adversarial \
        --skip-extraction

This script:
1. Loads the adversarial YAML test set
2. Loads the model and extracts activations (or loads cached)
3. Loads the saved probe ensembles at each layer
4. Evaluates per-example predictions and reports:
   - Overall accuracy by layer
   - Accuracy broken down by adversarial subcategory
   - Per-example results with confidence scores for error analysis
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.inference.extract_activations import load_model_and_tokenizer, extract_activations_batch
from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_probe, evaluate_ensemble


def load_adversarial_data(path: str) -> list:
    """Load adversarial examples from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["examples"]


def load_probe_ensemble(probes_dir: str, layer_idx: int, random_seeds: list, hidden_dim: int) -> list:
    """Load saved probe weights for a given layer."""
    probes = []
    for seed in random_seeds:
        probe_path = os.path.join(probes_dir, f"layer{layer_idx}_seed{seed}.pt")
        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"Probe not found: {probe_path}")
        probe = LinearProbe(hidden_dim)
        probe.load_state_dict(torch.load(probe_path, weights_only=True))
        probe.eval()
        probes.append(probe)
    return probes


def main():
    parser = argparse.ArgumentParser(description="Evaluate probes on adversarial test set")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Model config YAML (required unless --skip-extraction)")
    parser.add_argument("--probe-config", type=str, required=True)
    parser.add_argument("--adversarial-data", type=str, required=True)
    parser.add_argument("--probes-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Load cached activations from output-dir instead of extracting")
    args = parser.parse_args()

    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.probe_config = resolve_path(args.probe_config)
    args.adversarial_data = resolve_path(args.adversarial_data)
    args.probes_dir = resolve_path(args.probes_dir)
    args.output_dir = resolve_path(args.output_dir)
    if args.model_config:
        args.model_config = resolve_path(args.model_config)

    # Load configs
    with open(args.probe_config) as f:
        probe_config = yaml.safe_load(f)

    target_layers = probe_config["target_layers"]
    random_seeds = probe_config["random_seeds"]
    pooling = probe_config.get("pooling", "last_token")

    if args.model_config:
        with open(args.model_config) as f:
            model_config = yaml.safe_load(f)
        hidden_dim = model_config["hidden_dim"]
    else:
        # Infer hidden_dim from saved probe weights
        sample_path = os.path.join(args.probes_dir, f"layer{target_layers[0]}_seed{random_seeds[0]}.pt")
        state_dict = torch.load(sample_path, weights_only=True)
        hidden_dim = state_dict["linear.weight"].shape[1]
        model_config = None

    # Load adversarial data
    print("=" * 70)
    print("Loading adversarial test data")
    print("=" * 70)
    examples = load_adversarial_data(args.adversarial_data)
    texts = [ex["text"] for ex in examples]
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    print(f"  {len(examples)} examples ({n_pos} positive, {n_neg} negative)")

    subcategories = [ex.get("subcategory", "unknown") for ex in examples]
    unique_subcats = sorted(set(subcategories))
    print(f"  Subcategories: {unique_subcats}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    activations_dir = os.path.join(args.output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    # Extract or load activations
    if not args.skip_extraction:
        if model_config is None:
            parser.error("--model-config is required when not using --skip-extraction")

        print("\n" + "=" * 70)
        print("Extracting activations")
        print("=" * 70)
        import time
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(model_config)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        max_length = model_config.get("max_tokens", 512)
        acts = extract_activations_batch(texts, model, tokenizer, target_layers, pooling, max_length=max_length)

        for layer_idx, tensor in acts.items():
            save_path = os.path.join(activations_dir, f"adversarial_layer{layer_idx}.pt")
            torch.save(tensor, save_path)
            print(f"  Saved {save_path} (shape: {tensor.shape})")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  GPU memory freed.")
    else:
        print("\n  Loading cached activations from disk...")
        acts = {}
        for layer_idx in target_layers:
            path = os.path.join(activations_dir, f"adversarial_layer{layer_idx}.pt")
            acts[layer_idx] = torch.load(path, weights_only=True)
            print(f"  Loaded {path} (shape: {acts[layer_idx].shape})")

    # Evaluate probes
    print("\n" + "=" * 70)
    print("Evaluating probes on adversarial data")
    print("=" * 70)

    all_results = {}

    for layer_idx in target_layers:
        print(f"\n--- Layer {layer_idx} ---")
        probes = load_probe_ensemble(args.probes_dir, layer_idx, random_seeds, hidden_dim)
        layer_acts = acts[layer_idx]

        result = evaluate_ensemble(probes, layer_acts, labels)

        acc = result["ensemble_accuracy"]
        f1 = result["ensemble_f1"]
        auc = result.get("ensemble_auc_roc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  Ensemble accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc_str}")

        majority_vote = result["majority_vote"]
        mean_conf = result["mean_confidence"]

        # Per-subcategory breakdown
        subcat_correct = defaultdict(int)
        subcat_total = defaultdict(int)
        subcat_errors = defaultdict(list)

        for i, ex in enumerate(examples):
            subcat = ex.get("subcategory", "unknown")
            pred = majority_vote[i].item()
            true = ex["label"]
            conf = mean_conf[i].item()
            correct = pred == true
            subcat_total[subcat] += 1
            if correct:
                subcat_correct[subcat] += 1
            else:
                subcat_errors[subcat].append({
                    "text": ex["text"][:80] + "..." if len(ex["text"]) > 80 else ex["text"],
                    "true_label": true,
                    "predicted": pred,
                    "confidence": round(conf, 4),
                    "notes": ex.get("notes", ""),
                })

        print(f"\n  {'Subcategory':<35} {'Acc':>7} {'N':>4}")
        print("  " + "-" * 50)
        for subcat in unique_subcats:
            n = subcat_total[subcat]
            c = subcat_correct[subcat]
            subcat_acc = c / n if n > 0 else 0
            marker = "  <-- WEAK" if subcat_acc < 0.75 else ""
            print(f"  {subcat:<35} {subcat_acc:>6.1%} {n:>4}{marker}")

        # Print misclassified examples
        total_errors = sum(len(v) for v in subcat_errors.values())
        if total_errors > 0:
            print(f"\n  Misclassified examples ({total_errors} total):")
            for subcat in unique_subcats:
                for err in subcat_errors.get(subcat, []):
                    direction = "FP" if err["true_label"] == 0 else "FN"
                    print(f"    [{direction}] (conf={err['confidence']:.2f}) [{subcat}] {err['text']}")

        # Store results for JSON output
        per_example = []
        for i, ex in enumerate(examples):
            per_example.append({
                "text": ex["text"],
                "true_label": ex["label"],
                "predicted_label": majority_vote[i].item(),
                "mean_confidence": round(mean_conf[i].item(), 4),
                "correct": majority_vote[i].item() == ex["label"],
                "subcategory": ex.get("subcategory", "unknown"),
                "notes": ex.get("notes", ""),
            })

        all_results[f"layer_{layer_idx}"] = {
            "ensemble_accuracy": acc,
            "ensemble_f1": f1,
            "ensemble_auc_roc": auc,
            "subcategory_accuracy": {
                subcat: subcat_correct[subcat] / subcat_total[subcat]
                for subcat in unique_subcats if subcat_total[subcat] > 0
            },
            "per_example": per_example,
        }

    # Save full results
    results_path = os.path.join(args.output_dir, "adversarial_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nFull results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("ADVERSARIAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Layer':>8} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 40)
    for layer_idx in target_layers:
        r = all_results[f"layer_{layer_idx}"]
        auc = r["ensemble_auc_roc"]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{layer_idx:>8} {r['ensemble_accuracy']:>10.4f} {r['ensemble_f1']:>10.4f} {auc_str:>10}")

    print("\nDone!")


if __name__ == "__main__":
    main()
