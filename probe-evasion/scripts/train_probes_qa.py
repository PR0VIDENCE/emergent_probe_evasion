"""
Train probes on generation-matched activations.

Loads activations extracted by generate_and_extract.py, splits by contrastive
pair (both members of a pair always in the same split to prevent leakage),
and trains probe ensembles at each token position x layer combination.

v2 additions:
- Pre-assigned splits from split_assignment.yaml
- Top-K layer selection by validation AUROC
- Logistic regression combiner over top-K layers

Usage:
    python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training.yaml
    python scripts/train_probes_qa.py --config configs/experiments/qa_probe_training_v2.yaml --data-dir /workspace/probe_data_v2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def load_preassigned_splits(
    split_file: str,
    available_pair_ids: List[int],
) -> Dict[str, List[int]]:
    """
    Load pre-assigned splits from split_assignment.yaml.

    Args:
        split_file: Path to split_assignment.yaml.
        available_pair_ids: List of pair IDs that have activations available.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to lists of pair IDs.
    """
    with open(split_file, "r") as f:
        split_data = yaml.safe_load(f)

    assignments = split_data["assignments"]
    available_set = set(available_pair_ids)

    splits = {"train": [], "val": [], "test": []}
    for pair_id_str, split_name in assignments.items():
        pair_id = int(pair_id_str)
        if pair_id in available_set and split_name in splits:
            splits[split_name].append(pair_id)

    for split_name in splits:
        splits[split_name].sort()

    return splits


def load_activations_for_position(
    data_dir: str,
    position: str,
    pair_ids: List[int],
    target_layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Load activations for a given position and set of pair IDs.

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


def select_top_k_layers(
    position_results: Dict[int, dict],
    k: int = 5,
    metric: str = "ensemble_auc_roc",
) -> List[int]:
    """
    Select top-K layers by validation AUROC (or other metric).

    Args:
        position_results: {layer_idx: {metric_name: value, ...}}
        k: Number of layers to select.
        metric: Which metric to rank by.

    Returns:
        List of top-K layer indices, sorted by metric descending.
    """
    scored = []
    for layer_idx, metrics in position_results.items():
        score = metrics.get(metric)
        if score is not None:
            scored.append((layer_idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_layers = [layer_idx for layer_idx, _ in scored[:k]]
    return top_layers


def build_combiner_features(
    data_dir: str,
    position: str,
    pair_ids: List[int],
    top_layers: List[int],
    probes_by_layer: Dict[int, List[LinearProbe]],
    scalers_by_layer: Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix for combiner training.

    For each example, produces a K-dim feature vector where each feature is
    the mean confidence from the probe ensemble at one of the top-K layers.

    Returns:
        Tuple of (X, y) where X is (n_samples, K) and y is (n_samples,).
    """
    # Load activations for the position
    acts, labels = load_activations_for_position(
        data_dir, position, pair_ids, top_layers,
    )

    n_samples = len(labels)
    if n_samples == 0:
        return np.array([]).reshape(0, len(top_layers)), np.array([])

    features = np.zeros((n_samples, len(top_layers)))

    for i, layer_idx in enumerate(top_layers):
        if layer_idx not in acts:
            continue

        probes = probes_by_layer[layer_idx]
        scaler_mean, scaler_scale = scalers_by_layer[layer_idx]

        result = evaluate_ensemble(
            probes, acts[layer_idx], labels,
            scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        )
        # mean_confidence is a tensor of shape (n_samples,)
        features[:, i] = result["mean_confidence"].cpu().numpy()

    return features, labels.numpy()


def train_combiner(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    C: float = 1.0,
) -> dict:
    """
    Train logistic regression combiner over top-K layer features.

    Args:
        train_X: Training features (n_train, K).
        train_y: Training labels (n_train,).
        val_X: Validation features (n_val, K).
        val_y: Validation labels (n_val,).
        C: Regularization parameter.

    Returns:
        Dict with weights, intercept, scaler, operating_points, cross_validated_auc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    # Fit scaler on training data
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)

    # Train logistic regression
    lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    lr.fit(train_X_scaled, train_y)

    # Score on validation set
    val_probs = lr.predict_proba(val_X_scaled)[:, 1]
    try:
        val_auc = roc_auc_score(val_y, val_probs)
    except ValueError:
        val_auc = None

    # Compute operating points from validation ROC curve
    operating_points = {}
    if val_auc is not None:
        fpr_arr, tpr_arr, thresholds = roc_curve(val_y, val_probs)

        # zero_fpr: highest threshold where FPR == 0
        zero_fpr_mask = fpr_arr == 0.0
        if zero_fpr_mask.any():
            idx = np.where(zero_fpr_mask)[0][-1]
            operating_points["zero_fpr"] = {
                "threshold": round(float(thresholds[idx]), 4),
                "expected_tpr": round(float(tpr_arr[idx]), 4),
                "expected_fpr": 0.0,
            }

        # strict: FPR <= 2%
        strict_mask = fpr_arr <= 0.02
        if strict_mask.any():
            idx = np.where(strict_mask)[0][-1]
            operating_points["strict"] = {
                "threshold": round(float(thresholds[idx]), 4),
                "expected_tpr": round(float(tpr_arr[idx]), 4),
                "expected_fpr": round(float(fpr_arr[idx]), 4),
            }

        # low_fpr: FPR <= 3% (our target)
        low_fpr_mask = fpr_arr <= 0.03
        if low_fpr_mask.any():
            idx = np.where(low_fpr_mask)[0][-1]
            operating_points["low_fpr"] = {
                "threshold": round(float(thresholds[idx]), 4),
                "expected_tpr": round(float(tpr_arr[idx]), 4),
                "expected_fpr": round(float(fpr_arr[idx]), 4),
            }

        # balanced: closest to FPR=TPR diagonal (Youden's J)
        j_scores = tpr_arr - fpr_arr
        idx = int(np.argmax(j_scores))
        operating_points["balanced"] = {
            "threshold": round(float(thresholds[idx]), 4),
            "expected_tpr": round(float(tpr_arr[idx]), 4),
            "expected_fpr": round(float(fpr_arr[idx]), 4),
        }

        # high_recall: TPR >= 96%
        high_recall_mask = tpr_arr >= 0.96
        if high_recall_mask.any():
            idx = np.where(high_recall_mask)[0][0]  # first point reaching 96% TPR
            operating_points["high_recall"] = {
                "threshold": round(float(thresholds[idx]), 4),
                "expected_tpr": round(float(tpr_arr[idx]), 4),
                "expected_fpr": round(float(fpr_arr[idx]), 4),
            }

    return {
        "weights": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cross_validated_auc": round(val_auc, 4) if val_auc is not None else None,
        "operating_points": operating_points,
    }


def save_combiner_config(
    combiner_result: dict,
    top_layers: List[int],
    position: str,
    output_path: str,
    C: float = 1.0,
):
    """Save combiner as weighted_combination_v2.yaml."""
    config = {
        "description": (
            f"Learned weighted combination of probe scores. "
            f"Top-{len(top_layers)} layers by {position} validation AUROC. "
            f"Trained on v2 dataset with logistic regression combiner."
        ),
        "model": "logistic_regression",
        "regularization_C": C,
        "cross_validated_auc": combiner_result["cross_validated_auc"],
        "feature_order": {
            "positions": [position],
            "layers": top_layers,
            "note": (
                f"Features are {len(top_layers)} mean_confidence values, "
                f"one per top-{len(top_layers)} layer at {position} position."
            ),
        },
        "scaler": {
            "mean": combiner_result["scaler_mean"],
            "scale": combiner_result["scaler_scale"],
        },
        "weights": combiner_result["weights"],
        "intercept": combiner_result["intercept"],
        "operating_points": combiner_result["operating_points"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Combiner config saved to {output_path}")


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
    combiner_config = config.get("combiner", {})

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

    split_method = split_config.get("method", "random")

    if split_method == "preassigned":
        split_file = split_config.get("split_file")
        if split_file:
            split_file = resolve_path(split_file)
        if split_file and os.path.exists(split_file):
            print(f"  Using pre-assigned splits from {split_file}")
            splits = load_preassigned_splits(split_file, pair_ids)
        else:
            print(f"  WARNING: Pre-assigned split file not found: {split_file}")
            print(f"  Falling back to random pair-aware split")
            splits = pair_aware_split(
                pair_ids,
                train_ratio=split_config["train"],
                val_ratio=split_config["val"],
                test_ratio=split_config["test"],
            )
    else:
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
            "method": split_method,
        }, f, indent=2)

    # Train probes for each position x layer
    print("\n" + "=" * 60)
    print("Step 3: Training probes")
    print("=" * 60)
    print(f"  Positions: {token_positions}")
    print(f"  Layers: {target_layers}")
    print(f"  Seeds: {probe_config['random_seeds']}")

    results = {}
    val_results = {}  # Store val metrics for top-K selection
    all_probes = {}   # {(position, layer): probes}
    all_scalers = {}  # {(position, layer): (mean, scale)}

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
        position_val_results = {}

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

            # Store for combiner
            all_probes[(position, layer_idx)] = probes
            all_scalers[(position, layer_idx)] = (scaler_mean, scaler_scale)

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

            # Evaluate on validation set (for top-K selection)
            if layer_idx in val_acts:
                val_result = evaluate_ensemble(
                    probes, val_acts[layer_idx], val_labels,
                    scaler_mean=scaler_mean, scaler_scale=scaler_scale,
                )
                position_val_results[layer_idx] = {
                    "ensemble_auc_roc": val_result.get("ensemble_auc_roc"),
                    "ensemble_accuracy": val_result.get("ensemble_accuracy"),
                }

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
        val_results[position] = position_val_results

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

    # Step 5: Top-K layer selection and combiner training
    top_k = probe_config.get("top_k_layers", 5)
    primary_position = probe_config.get("primary_position", "answer_mean_pool")

    if combiner_config.get("enabled", False) and primary_position in val_results:
        print("\n" + "=" * 60)
        print(f"Step 5: Top-{top_k} layer selection & combiner training")
        print("=" * 60)

        # Select top-K layers by validation AUROC at primary position
        top_layers = select_top_k_layers(val_results[primary_position], k=top_k)
        print(f"  Primary position: {primary_position}")
        print(f"  Top-{top_k} layers by val AUROC: {top_layers}")

        for layer_idx in top_layers:
            auc = val_results[primary_position][layer_idx]["ensemble_auc_roc"]
            print(f"    Layer {layer_idx}: val AUROC = {auc:.4f}")

        # Save top-K selection
        top5_path = os.path.join(data_dir, "top5_layers.json")
        with open(top5_path, "w") as f:
            json.dump({
                "position": primary_position,
                "top_layers": top_layers,
                "val_aurocs": {
                    l: val_results[primary_position][l]["ensemble_auc_roc"]
                    for l in top_layers
                },
            }, f, indent=2)
        print(f"  Saved top-{top_k} selection to {top5_path}")

        # Build combiner features
        print(f"\n  Building combiner features...")

        # Prepare probes and scalers dicts for the primary position
        probes_by_layer = {l: all_probes[(primary_position, l)] for l in top_layers}
        scalers_by_layer = {l: all_scalers[(primary_position, l)] for l in top_layers}

        train_X, train_y = build_combiner_features(
            data_dir, primary_position, splits["train"],
            top_layers, probes_by_layer, scalers_by_layer,
        )
        val_X, val_y = build_combiner_features(
            data_dir, primary_position, splits["val"],
            top_layers, probes_by_layer, scalers_by_layer,
        )

        print(f"  Train: {train_X.shape}, Val: {val_X.shape}")

        # Train combiner
        reg_C = combiner_config.get("regularization_C", 1.0)
        combiner_result = train_combiner(train_X, train_y, val_X, val_y, C=reg_C)

        print(f"  Combiner val AUC: {combiner_result['cross_validated_auc']}")
        print(f"  Operating points:")
        for op_name, op in combiner_result["operating_points"].items():
            print(f"    {op_name}: threshold={op['threshold']}, "
                  f"TPR={op['expected_tpr']}, FPR={op['expected_fpr']}")

        # Save combiner config
        combiner_output_path = combiner_config.get(
            "output_path", "configs/probes/weighted_combination_v2.yaml"
        )
        combiner_output_path = resolve_path(combiner_output_path)
        save_combiner_config(
            combiner_result, top_layers, primary_position,
            combiner_output_path, C=reg_C,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
