"""
Train probes on generation-matched activations.

Loads activations extracted by generate_and_extract.py, splits by contrastive
pair (both members of a pair always in the same split to prevent leakage),
and trains probe ensembles at each token position x layer combination.

Usage:
    python scripts/train_probes_qa.py \
        --config configs/experiments/qa_probe_training.yaml

    # Use custom data directory:
    python scripts/train_probes_qa.py \
        --config configs/experiments/qa_probe_training.yaml \
        --data-dir /workspace/probe_data
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.probes.architectures import LinearProbe
from src.probes.train import train_probe_ensemble
from src.probes.evaluate import evaluate_ensemble
from src.utils.logging import setup_logging


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_generation_log(log_path: str) -> List[dict]:
    """Load completed generations from the JSONL log."""
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if "error" not in entry:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def get_pair_ids_from_log(log_entries: List[dict]) -> List[int]:
    """
    Get list of pair IDs that have BOTH tree and non_tree generations.

    Returns:
        Sorted list of global_pair_ids with complete pairs.
    """
    tree_pairs = set()
    non_tree_pairs = set()

    for entry in log_entries:
        pid = entry["global_pair_id"]
        if entry["label"] == "tree":
            tree_pairs.add(pid)
        else:
            non_tree_pairs.add(pid)

    complete_pairs = sorted(tree_pairs & non_tree_pairs)
    return complete_pairs


def pair_aware_split(
    pair_ids: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split pair IDs into train/val/test ensuring no pair leakage.

    Both members of a contrastive pair are always in the same split.

    Args:
        pair_ids: List of global pair IDs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to lists of pair IDs.
    """
    import random
    rng = random.Random(seed)

    shuffled = list(pair_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    return splits


def load_activations_for_position(
    data_dir: str,
    position: str,
    pair_ids: List[int],
    target_layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Load activations for a given position and set of pair IDs.

    Args:
        data_dir: Base data directory with activations/{label}/{position}/ structure.
        position: Token position name (e.g., "last_token").
        pair_ids: List of pair IDs to load.
        target_layers: List of layer indices.

    Returns:
        Tuple of (layer_activations, labels) where:
        - layer_activations: {layer_idx: tensor(n_samples, hidden_dim)}
        - labels: tensor(n_samples,) with 1=tree, 0=non_tree
    """
    collected = {layer_idx: [] for layer_idx in target_layers}
    labels = []
    loaded_count = 0

    for pair_id in pair_ids:
        for label_name, label_val in [("tree", 1), ("non_tree", 0)]:
            prompt_id = f"{label_name}_{pair_id:04d}"
            act_path = os.path.join(
                data_dir, "activations", label_name, position, f"{prompt_id}.pt"
            )

            if not os.path.exists(act_path):
                print(f"    WARNING: Missing activation file: {act_path}")
                continue

            layer_acts = torch.load(act_path, weights_only=True)

            for layer_idx in target_layers:
                if layer_idx in layer_acts:
                    collected[layer_idx].append(layer_acts[layer_idx].float())
                else:
                    print(f"    WARNING: Layer {layer_idx} missing in {act_path}")

            labels.append(label_val)
            loaded_count += 1

    # Stack into tensors
    result = {}
    for layer_idx in target_layers:
        if collected[layer_idx]:
            result[layer_idx] = torch.stack(collected[layer_idx])

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return result, labels_tensor


def load_supplementary_activations(
    data_dir: str,
    position: str,
    prompt_ids: List[str],
    labels: List[int],
    target_layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Load activations for supplementary prompt IDs (hw_val_*, adv_*).

    These are stored in the same activations/{tree,non_tree}/{position}/ structure
    as regular QA pairs, but with non-numeric prompt IDs.

    Args:
        data_dir: Base data directory.
        position: Token position name.
        prompt_ids: List of supplementary prompt IDs.
        labels: List of int labels (1=tree, 0=non_tree) matching prompt_ids.
        target_layers: Layer indices.

    Returns:
        Tuple of (layer_activations, labels_tensor).
    """
    collected = {layer_idx: [] for layer_idx in target_layers}
    valid_labels = []
    loaded = 0

    for pid, label_val in zip(prompt_ids, labels):
        label_name = "tree" if label_val == 1 else "non_tree"
        act_path = os.path.join(
            data_dir, "activations", label_name, position, f"{pid}.pt"
        )

        if not os.path.exists(act_path):
            continue

        layer_acts = torch.load(act_path, weights_only=True)
        for layer_idx in target_layers:
            if layer_idx in layer_acts:
                collected[layer_idx].append(layer_acts[layer_idx].float())
        valid_labels.append(label_val)
        loaded += 1

    result = {}
    for layer_idx in target_layers:
        if collected[layer_idx]:
            result[layer_idx] = torch.stack(collected[layer_idx])

    labels_tensor = torch.tensor(valid_labels, dtype=torch.float32)
    return result, labels_tensor


def parse_supplementary_yaml(yaml_path: str) -> Tuple[List[str], List[int]]:
    """Parse supplementary YAML to get prompt IDs and labels (no pyyaml)."""
    prompt_ids = []
    labels = []

    with open(yaml_path, "r") as f:
        text = f.read()

    current_id = None
    current_label = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current_id is not None:
                prompt_ids.append(current_id)
                labels.append(current_label if current_label is not None else 0)
            current_id = stripped.split(":", 1)[1].strip().strip('"')
            current_label = None
        elif stripped.startswith("label:") and current_id is not None:
            current_label = int(stripped.split(":", 1)[1].strip())

    if current_id is not None:
        prompt_ids.append(current_id)
        labels.append(current_label if current_label is not None else 0)

    return prompt_ids, labels


def main():
    parser = argparse.ArgumentParser(
        description="Train probes on generation-matched activations"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to qa_probe_training.yaml config")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: from config)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: data-dir/logs)")
    parser.add_argument("--probe-dir", type=str, default=None,
                        help="Override probe output directory (default: data-dir/probes)")
    args = parser.parse_args()

    # Resolve paths
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)
    config = load_config(args.config)
    model_config = load_config(resolve_path(config["model_config"]))

    data_dir = args.data_dir or config["storage"]["base_dir"]
    probe_base_dir = args.probe_dir or os.path.join(data_dir, "probes")

    # Set up logging
    log_dir = args.log_dir or os.path.join(data_dir, "logs")
    setup_logging(log_dir, "train_probes_qa")
    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))
    token_positions = config["token_positions"]
    probe_config = config["probe_training"]
    split_config = config["split"]
    supp_config = config.get("supplementary", {})

    # Parse handwritten validation file if configured
    hw_val_ids, hw_val_labels = [], []
    hw_val_path = supp_config.get("handwritten_validation")
    if hw_val_path:
        hw_val_path = resolve_path(hw_val_path)
        if os.path.exists(hw_val_path):
            hw_val_ids, hw_val_labels = parse_supplementary_yaml(hw_val_path)
            print(f"  Handwritten validation: {len(hw_val_ids)} examples from {hw_val_path}")

    # Load generation log to find complete pairs
    print("=" * 60)
    print("Step 1: Loading generation log")
    print("=" * 60)
    log_path = os.path.join(data_dir, "prompts", "generation_log.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Generation log not found: {log_path}")

    log_entries = load_generation_log(log_path)
    pair_ids = get_pair_ids_from_log(log_entries)
    print(f"  Total log entries: {len(log_entries)}")
    print(f"  Complete pairs: {len(pair_ids)}")

    # Split by pair
    print("\n" + "=" * 60)
    print("Step 2: Splitting data by contrastive pair")
    print("=" * 60)
    splits = pair_aware_split(
        pair_ids,
        train_ratio=split_config["train"],
        val_ratio=split_config["val"],
        test_ratio=split_config["test"],
    )
    for split_name, split_ids in splits.items():
        print(f"  {split_name}: {len(split_ids)} pairs ({len(split_ids) * 2} examples)")

    # Save split info
    split_info_path = os.path.join(data_dir, "split_info.json")
    with open(split_info_path, "w") as f:
        json.dump({
            "total_pairs": len(pair_ids),
            "splits": {k: v for k, v in splits.items()},
        }, f, indent=2)

    # Train probes for each position x layer
    print("\n" + "=" * 60)
    print("Step 3: Training probes")
    print("=" * 60)
    print(f"  Positions: {token_positions}")
    print(f"  Layers: {target_layers}")
    print(f"  Seeds: {probe_config['random_seeds']}")

    results = {}

    for position in token_positions:
        print(f"\n--- Position: {position} ---")

        # Load train activations
        print(f"  Loading train activations...")
        train_acts, train_labels = load_activations_for_position(
            data_dir, position, splits["train"], target_layers,
        )
        print(f"  Train: {len(train_labels)} examples "
              f"({int(train_labels.sum())} tree, {int(len(train_labels) - train_labels.sum())} non_tree)")

        # Load val activations
        print(f"  Loading val activations...")
        val_acts, val_labels = load_activations_for_position(
            data_dir, position, splits["val"], target_layers,
        )
        print(f"  Val: {len(val_labels)} examples (QA pairs)")

        # Inject handwritten validation examples into val set
        if hw_val_ids:
            hw_acts, hw_labels = load_supplementary_activations(
                data_dir, position, hw_val_ids, hw_val_labels, target_layers,
            )
            if len(hw_labels) > 0:
                for layer_idx in target_layers:
                    if layer_idx in hw_acts and layer_idx in val_acts:
                        val_acts[layer_idx] = torch.cat([val_acts[layer_idx], hw_acts[layer_idx]])
                    elif layer_idx in hw_acts:
                        val_acts[layer_idx] = hw_acts[layer_idx]
                val_labels = torch.cat([val_labels, hw_labels])
                print(f"  Val + handwritten: {len(val_labels)} examples total")

        # Load test activations
        print(f"  Loading test activations...")
        test_acts, test_labels = load_activations_for_position(
            data_dir, position, splits["test"], target_layers,
        )
        print(f"  Test: {len(test_labels)} examples")

        position_results = {}

        for layer_idx in target_layers:
            if layer_idx not in train_acts:
                print(f"  Layer {layer_idx}: no training data, skipping")
                continue

            print(f"  Training layer {layer_idx} ensemble...")

            # Build probe training config
            train_config = {
                "random_seeds": probe_config["random_seeds"],
                "learning_rate": probe_config["learning_rate"],
                "num_epochs": probe_config["num_epochs"],
                "weight_decay": probe_config["weight_decay"],
                "patience": probe_config["patience"],
                "val_activations": val_acts.get(layer_idx),
                "val_labels": val_labels if layer_idx in val_acts else None,
                "normalize": probe_config.get("normalize", True),
            }

            # Train ensemble
            ensemble_result = train_probe_ensemble(
                train_acts[layer_idx], train_labels, train_config
            )
            probes = ensemble_result["probes"]
            scaler_mean = ensemble_result["scaler_mean"]
            scaler_scale = ensemble_result["scaler_scale"]

            # Save probes
            probe_dir = os.path.join(probe_base_dir, position)
            os.makedirs(probe_dir, exist_ok=True)

            seeds = probe_config["random_seeds"]
            for probe, seed in zip(probes, seeds):
                probe_path = os.path.join(probe_dir, f"layer{layer_idx}_seed{seed}.pt")
                torch.save(probe.state_dict(), probe_path)

            # Save scaler
            if scaler_mean is not None:
                scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
                torch.save({"scaler_mean": scaler_mean, "scaler_scale": scaler_scale},
                           scaler_path)

            # Evaluate on test set
            if layer_idx in test_acts:
                test_result = evaluate_ensemble(
                    probes, test_acts[layer_idx], test_labels,
                    scaler_mean=scaler_mean, scaler_scale=scaler_scale,
                )
                layer_result = {
                    "ensemble_accuracy": test_result.get("ensemble_accuracy"),
                    "ensemble_f1": test_result.get("ensemble_f1"),
                    "ensemble_auc_roc": test_result.get("ensemble_auc_roc"),
                    "agreement_ratio": test_result.get("agreement_ratio"),
                    "individual_accuracies": test_result.get("individual_accuracies"),
                }
                position_results[layer_idx] = layer_result

                print(f"    Acc={layer_result['ensemble_accuracy']:.3f}, "
                      f"AUC={layer_result['ensemble_auc_roc']:.3f}, "
                      f"F1={layer_result['ensemble_f1']:.3f}")
            else:
                print(f"    No test data for layer {layer_idx}")

        results[position] = position_results

    # Save results
    print("\n" + "=" * 60)
    print("Step 4: Saving results")
    print("=" * 60)

    results_path = os.path.join(data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Test Set)")
    print("=" * 60)

    layer_headers = "  ".join(f"L{l:>3}" for l in target_layers)
    print(f"{'Position':<30} {layer_headers}")
    print("-" * (30 + len(target_layers) * 6))

    for position in token_positions:
        if position in results:
            accs = []
            for layer_idx in target_layers:
                if layer_idx in results[position]:
                    accs.append(f"{results[position][layer_idx]['ensemble_accuracy']:.3f}")
                else:
                    accs.append("  N/A")
            print(f"{position:<30} {'  '.join(accs)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
