"""
Preliminary probe training experiment for Qwen2.5 32B Instruct.

Usage:
    python scripts/run_preliminary.py \
        --model-config configs/models/qwen2_5_32b.yaml \
        --probe-config configs/probes/trees.yaml \
        --data-dir data/concepts/trees \
        --output-dir data/outputs/preliminary

This script:
1. Loads and splits the concept dataset (400 examples -> 320/40/40)
2. Loads Qwen2.5 32B with 4-bit quantization
3. Extracts last-token activations at 5 target layers
4. Trains 4 probes per layer (20 total)
5. Evaluates on held-out test set
6. Saves activations, probe weights, and results
"""

import argparse
import hashlib
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import torch
import yaml

# Add project root and scripts dir to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_concept_dataset import load_yaml_examples, validate_example, split_dataset
from src.inference.extract_activations import load_model_and_tokenizer, extract_activations_batch
from src.probes.train import train_probe_ensemble
from src.probes.evaluate import evaluate_ensemble


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_data_manifest(data_dir: str) -> str:
    """Compute a hash of all batch files to detect data changes."""
    pattern = os.path.join(data_dir, "generated_batch_*.yaml")
    batch_files = sorted(glob(pattern))
    hasher = hashlib.sha256()
    for f in batch_files:
        hasher.update(f.encode())
        hasher.update(open(f, "rb").read())
    return hasher.hexdigest()[:16]


def load_all_examples(data_dir: str) -> list:
    """Load all generated batch YAML files from the data directory."""
    pattern = os.path.join(data_dir, "generated_batch_*.yaml")
    batch_files = sorted(glob(pattern))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found matching {pattern}")

    all_examples = []
    for batch_file in batch_files:
        examples = load_yaml_examples(batch_file)
        for ex in examples:
            errors = validate_example(ex)
            if errors:
                print(f"WARNING: Skipping invalid example in {batch_file}: {errors}")
                continue
            all_examples.append(ex)

    print(f"Loaded {len(all_examples)} examples from {len(batch_files)} batch files")
    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Run preliminary probe training")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--probe-config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/concepts/trees")
    parser.add_argument("--output-dir", type=str, default="data/outputs/preliminary")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip activation extraction, load from disk")
    args = parser.parse_args()

    # Resolve relative paths against project root
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.model_config = resolve_path(args.model_config)
    args.probe_config = resolve_path(args.probe_config)
    args.data_dir = resolve_path(args.data_dir)
    args.output_dir = resolve_path(args.output_dir)

    # Load configs
    model_config = load_config(args.model_config)
    probe_config = load_config(args.probe_config)

    target_layers = probe_config["target_layers"]
    random_seeds = probe_config["random_seeds"]
    pooling = probe_config.get("pooling", "last_token")
    hidden_dim = model_config["hidden_dim"]
    training_cfg = probe_config.get("training", {})

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    activations_dir = os.path.join(args.output_dir, "activations")
    probes_dir = os.path.join(args.output_dir, "probes")
    os.makedirs(activations_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)

    # Step 1: Load and split dataset
    print("=" * 60)
    print("Step 1: Loading and splitting dataset")
    print("=" * 60)
    all_examples = load_all_examples(args.data_dir)
    split_cfg = probe_config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    splits = split_dataset(
        all_examples,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        seed=42,
    )
    for split_name, split_examples in splits.items():
        n_pos = sum(1 for ex in split_examples if ex["label"] == 1)
        n_neg = len(split_examples) - n_pos
        print(f"  {split_name}: {len(split_examples)} examples ({n_pos} pos, {n_neg} neg)")

    # Compute data manifest for consistency checking
    data_manifest = compute_data_manifest(args.data_dir)
    manifest_path = os.path.join(activations_dir, "data_manifest.txt")

    # Step 2: Extract activations (or load from disk)
    if not args.skip_extraction:
        print("\n" + "=" * 60)
        print("Step 2: Loading model and extracting activations")
        print("=" * 60)
        print(f"  Model: {model_config['model_id']}")
        print(f"  Layers: {target_layers}")
        print(f"  Pooling: {pooling}")

        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(model_config)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        # Verify model architecture and layer indices
        num_layers = len(model.model.layers)
        print(f"  Model has {num_layers} layers (expected {model_config['num_layers']})")
        assert num_layers == model_config["num_layers"], \
            f"Layer count mismatch: got {num_layers}, expected {model_config['num_layers']}"
        invalid_layers = [l for l in target_layers if l < 0 or l >= num_layers]
        assert not invalid_layers, \
            f"Layer indices {invalid_layers} out of range for {num_layers}-layer model"

        # Extract for each split
        max_length = model_config.get("max_tokens", 512)
        for split_name, split_examples in splits.items():
            print(f"\n  Extracting activations for {split_name} split ({len(split_examples)} examples)...")
            texts = [ex["text"] for ex in split_examples]

            t0 = time.time()
            acts = extract_activations_batch(
                texts, model, tokenizer, target_layers, pooling, max_length=max_length
            )
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s ({elapsed/len(texts):.2f}s/example)")

            # Save activations
            for layer_idx, tensor in acts.items():
                save_path = os.path.join(activations_dir, f"{split_name}_layer{layer_idx}.pt")
                torch.save(tensor, save_path)
                print(f"    Saved {save_path} (shape: {tensor.shape})")

            # Save labels
            labels = torch.tensor([ex["label"] for ex in split_examples], dtype=torch.long)
            labels_path = os.path.join(activations_dir, f"{split_name}_labels.pt")
            torch.save(labels, labels_path)

        # Save data manifest
        with open(manifest_path, "w") as f:
            f.write(data_manifest)

        # Free GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()
        print("\n  GPU memory freed.")

    else:
        print("\n  Skipping extraction, loading activations from disk...")
        # Verify cached activations match current data
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                cached_manifest = f.read().strip()
            if cached_manifest != data_manifest:
                raise RuntimeError(
                    f"Data files changed since activations were extracted "
                    f"(cached={cached_manifest}, current={data_manifest}). "
                    f"Re-run without --skip-extraction."
                )
        else:
            print("  WARNING: No data manifest found, cannot verify activation consistency.")

    # Step 3: Train probes
    print("\n" + "=" * 60)
    print("Step 3: Training probes")
    print("=" * 60)

    train_labels = torch.load(os.path.join(activations_dir, "train_labels.pt"),
                              weights_only=True)
    val_labels = torch.load(os.path.join(activations_dir, "val_labels.pt"),
                            weights_only=True)
    test_labels = torch.load(os.path.join(activations_dir, "test_labels.pt"),
                             weights_only=True)

    all_results = {}

    for layer_idx in target_layers:
        print(f"\n  Layer {layer_idx}:")
        train_acts = torch.load(os.path.join(activations_dir, f"train_layer{layer_idx}.pt"),
                                weights_only=True)
        val_acts = torch.load(os.path.join(activations_dir, f"val_layer{layer_idx}.pt"),
                              weights_only=True)
        test_acts = torch.load(os.path.join(activations_dir, f"test_layer{layer_idx}.pt"),
                               weights_only=True)

        # Train ensemble with early stopping using validation data
        ensemble_config = {
            **training_cfg,
            "random_seeds": random_seeds,
            "val_activations": val_acts,
            "val_labels": val_labels,
            "normalize": probe_config.get("normalize", True),
        }
        t0 = time.time()
        ensemble_result = train_probe_ensemble(train_acts, train_labels, ensemble_config)
        probes = ensemble_result["probes"]
        scaler_mean = ensemble_result["scaler_mean"]
        scaler_scale = ensemble_result["scaler_scale"]
        print(f"    Trained {len(probes)} probes in {time.time() - t0:.2f}s")
        if scaler_mean is not None:
            print(f"    Normalization: enabled (mean range [{scaler_mean.min():.2f}, {scaler_mean.max():.2f}])")

        # Evaluate on test set
        test_result = evaluate_ensemble(probes, test_acts, test_labels,
                                        scaler_mean=scaler_mean, scaler_scale=scaler_scale)
        print(f"    Ensemble test accuracy: {test_result['ensemble_accuracy']:.4f}")
        print(f"    Ensemble test AUC-ROC:  {test_result.get('ensemble_auc_roc', 'N/A')}")
        print(f"    Ensemble test F1:       {test_result['ensemble_f1']:.4f}")
        if "individual_accuracies" in test_result:
            accs = test_result["individual_accuracies"]
            print(f"    Individual accuracies:  {[f'{a:.4f}' for a in accs]}")

        # Evaluate on train set (check for overfitting)
        train_result = evaluate_ensemble(probes, train_acts, train_labels,
                                         scaler_mean=scaler_mean, scaler_scale=scaler_scale)
        print(f"    Ensemble train accuracy: {train_result['ensemble_accuracy']:.4f}")

        # Save probes
        for i, probe in enumerate(probes):
            probe_path = os.path.join(probes_dir, f"layer{layer_idx}_seed{random_seeds[i]}.pt")
            torch.save(probe.state_dict(), probe_path)

        # Save scaler
        if scaler_mean is not None:
            scaler_path = os.path.join(probes_dir, f"layer{layer_idx}_scaler.pt")
            torch.save({"scaler_mean": scaler_mean, "scaler_scale": scaler_scale}, scaler_path)
            print(f"    Saved scaler to {scaler_path}")

        # Store serializable results
        all_results[f"layer_{layer_idx}"] = {
            "layer_idx": layer_idx,
            "test_ensemble_accuracy": test_result["ensemble_accuracy"],
            "test_ensemble_auc_roc": test_result.get("ensemble_auc_roc"),
            "test_ensemble_f1": test_result["ensemble_f1"],
            "test_individual_accuracies": test_result.get("individual_accuracies", []),
            "test_individual_auc_rocs": test_result.get("individual_auc_rocs", []),
            "train_ensemble_accuracy": train_result["ensemble_accuracy"],
            "agreement_ratio": test_result["agreement_ratio"],
        }

    # Step 4: Save summary results
    print("\n" + "=" * 60)
    print("Step 4: Saving results")
    print("=" * 60)

    results_summary = {
        "model": model_config["model_id"],
        "concept": probe_config["concept"],
        "num_probes_per_layer": len(random_seeds),
        "target_layers": target_layers,
        "pooling": pooling,
        "training_config": training_cfg,
        "dataset_sizes": {k: len(v) for k, v in splits.items()},
        "results_by_layer": all_results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Layer':>8} {'Test Acc':>10} {'Test AUC':>10} {'Test F1':>10} "
          f"{'Train Acc':>10} {'Agreement':>10}")
    print("-" * 60)
    for layer_idx in target_layers:
        r = all_results[f"layer_{layer_idx}"]
        auc = r["test_ensemble_auc_roc"]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{layer_idx:>8} {r['test_ensemble_accuracy']:>10.4f} {auc_str:>10} "
              f"{r['test_ensemble_f1']:>10.4f} {r['train_ensemble_accuracy']:>10.4f} "
              f"{r['agreement_ratio']:>10.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
