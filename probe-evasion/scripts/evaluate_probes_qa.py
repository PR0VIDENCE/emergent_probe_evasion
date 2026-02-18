"""
Evaluate QA-trained probes and produce comprehensive report.

v2 additions:
- Pre-assigned splits from split_assignment.yaml
- Per-group evaluation (A, A_prefix, B, C, D, E)
- Explicit TPR/FPR metrics
- Combiner evaluation with operating points
- Prefix robustness table for handwritten test pairs

Usage:
    python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training.yaml
    python scripts/evaluate_probes_qa.py --config configs/experiments/qa_probe_training_v2.yaml --data-dir /workspace/probe_data_v2
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_probe, evaluate_ensemble, load_weighted_combination, score_weighted_combination
from src.utils.logging import setup_logging


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_available_pair_ids(log_path: str) -> List[int]:
    """Load generation log and return pair IDs with complete pairs."""
    if not os.path.exists(log_path):
        return []
    tree_pairs = set()
    non_tree_pairs = set()
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "error" in entry:
                    continue
                pid = entry["global_pair_id"]
                if entry["label"] == "tree":
                    tree_pairs.add(pid)
                else:
                    non_tree_pairs.add(pid)
            except (json.JSONDecodeError, KeyError):
                continue
    return sorted(tree_pairs & non_tree_pairs)


def load_probes(
    probe_dir: str,
    layer_idx: int,
    seeds: List[int],
    hidden_dim: int,
) -> List[LinearProbe]:
    """Load a probe ensemble from disk."""
    probes = []
    for seed in seeds:
        probe = LinearProbe(hidden_dim)
        path = os.path.join(probe_dir, f"layer{layer_idx}_seed{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Probe not found: {path}")
        probe.load_state_dict(torch.load(path, weights_only=True))
        probe.eval()
        probes.append(probe)
    return probes


def load_preassigned_splits(
    split_file: str,
    available_pair_ids: List[int],
) -> Dict[str, List[int]]:
    """Load pre-assigned splits from split_assignment.yaml."""
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
    """Load activations for a given position and set of pair IDs."""
    collected = {layer_idx: [] for layer_idx in target_layers}
    labels = []

    for pair_id in pair_ids:
        for label_name, label_val in [("tree", 1), ("non_tree", 0)]:
            prompt_id = f"{label_name}_{pair_id:04d}"
            act_path = os.path.join(
                data_dir, "activations", label_name, position, f"{prompt_id}.pt"
            )

            if not os.path.exists(act_path):
                continue

            layer_acts = torch.load(act_path, weights_only=True)

            for layer_idx in target_layers:
                if layer_idx in layer_acts:
                    collected[layer_idx].append(layer_acts[layer_idx].float())

            labels.append(label_val)

    result = {}
    for layer_idx in target_layers:
        if collected[layer_idx]:
            result[layer_idx] = torch.stack(collected[layer_idx])

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return result, labels_tensor


def compute_tpr_fpr(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute TPR and FPR from binary labels and predictions.

    Args:
        labels: Ground truth (1=positive/tree, 0=negative/non_tree).
        predictions: Binary predictions.

    Returns:
        Dict with tpr, fpr, tn, fp, fn, tp.
    """
    from sklearn.metrics import confusion_matrix

    # Handle edge cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        if unique_labels[0] == 1:
            tp = int((predictions == 1).sum())
            fn = int((predictions == 0).sum())
            return {"tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0, "fpr": 0.0, "tp": tp, "fp": 0, "fn": fn, "tn": 0}
        else:
            fp = int((predictions == 1).sum())
            tn = int((predictions == 0).sum())
            return {"tpr": 0.0, "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0, "tp": 0, "fp": fp, "fn": 0, "tn": tn}

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"tpr": tpr, "fpr": fpr, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def load_generation_metadata(data_dir: str) -> Dict[int, dict]:
    """
    Load group/style metadata from generation JSONs.

    Returns:
        Dict mapping global_pair_id -> {group, elicitation_style, tree_topic, ...}
    """
    from glob import glob

    metadata = {}
    gen_files = (
        glob(os.path.join(data_dir, "generations", "tree", "*.json"))
        + glob(os.path.join(data_dir, "generations", "non_tree", "*.json"))
    )

    for gf in gen_files:
        try:
            with open(gf, "r") as f:
                gen = json.load(f)
            pid = gen.get("global_pair_id")
            if pid is not None and pid not in metadata:
                metadata[pid] = {
                    "group": gen.get("group", ""),
                    "elicitation_style": gen.get("elicitation_style", "default"),
                    "tree_topic": gen.get("tree_topic", ""),
                    "domain": gen.get("domain", ""),
                    "base_pair_ref": gen.get("base_pair_ref"),
                }
        except (json.JSONDecodeError, KeyError):
            continue

    return metadata


def evaluate_all_probes(
    data_dir: str,
    probe_base_dir: str,
    target_layers: List[int],
    token_positions: List[str],
    seeds: List[int],
    hidden_dim: int,
    test_pair_ids: List[int],
) -> Dict:
    """Evaluate all QA-trained probes on the test set."""
    results = {}

    for position in token_positions:
        print(f"\n--- Position: {position} ---")

        # Load test activations
        test_acts, test_labels = load_activations_for_position(
            data_dir, position, test_pair_ids, target_layers,
        )
        print(f"  Test set: {len(test_labels)} examples "
              f"({int(test_labels.sum())} tree, {int(len(test_labels) - test_labels.sum())} non_tree)")

        position_results = {}

        for layer_idx in target_layers:
            if layer_idx not in test_acts:
                print(f"  Layer {layer_idx}: no test data")
                continue

            probe_dir = os.path.join(probe_base_dir, position)
            try:
                probes = load_probes(probe_dir, layer_idx, seeds, hidden_dim)
            except FileNotFoundError as e:
                print(f"  Layer {layer_idx}: {e}")
                continue

            # Load scaler if available
            scaler_mean, scaler_scale = None, None
            scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
            if os.path.exists(scaler_path):
                scaler_data = torch.load(scaler_path, weights_only=True)
                scaler_mean = scaler_data["scaler_mean"]
                scaler_scale = scaler_data["scaler_scale"]

            # Evaluate ensemble
            ensemble_result = evaluate_ensemble(
                probes, test_acts[layer_idx], test_labels,
                scaler_mean=scaler_mean, scaler_scale=scaler_scale,
            )

            # Compute TPR/FPR
            labels_np = test_labels.cpu().numpy()
            preds_np = ensemble_result["majority_vote"].cpu().numpy()
            tpr_fpr = compute_tpr_fpr(labels_np, preds_np)

            # Evaluate individual probes
            individual_metrics = []
            for i, probe in enumerate(probes):
                ind_result = evaluate_probe(
                    probe, test_acts[layer_idx], test_labels,
                    scaler_mean=scaler_mean, scaler_scale=scaler_scale,
                )
                individual_metrics.append({
                    "seed": seeds[i],
                    "accuracy": ind_result.get("accuracy"),
                    "auc_roc": ind_result.get("auc_roc"),
                    "f1": ind_result.get("f1"),
                })

            layer_result = {
                "ensemble_accuracy": ensemble_result.get("ensemble_accuracy"),
                "ensemble_f1": ensemble_result.get("ensemble_f1"),
                "ensemble_auc_roc": ensemble_result.get("ensemble_auc_roc"),
                "agreement_ratio": ensemble_result.get("agreement_ratio"),
                "tpr": tpr_fpr["tpr"],
                "fpr": tpr_fpr["fpr"],
                "tp": tpr_fpr["tp"],
                "fp": tpr_fpr["fp"],
                "fn": tpr_fpr["fn"],
                "tn": tpr_fpr["tn"],
                "individual_probes": individual_metrics,
                "n_test_examples": len(test_labels),
            }
            position_results[layer_idx] = layer_result

            print(f"  Layer {layer_idx}: "
                  f"Acc={layer_result['ensemble_accuracy']:.3f}, "
                  f"AUC={layer_result['ensemble_auc_roc']:.3f}, "
                  f"TPR={tpr_fpr['tpr']:.3f}, FPR={tpr_fpr['fpr']:.3f}")

        results[position] = position_results

    return results


def evaluate_per_group(
    data_dir: str,
    probe_base_dir: str,
    position: str,
    layer_idx: int,
    seeds: List[int],
    hidden_dim: int,
    test_pair_ids: List[int],
    pair_metadata: Dict[int, dict],
) -> Dict[str, dict]:
    """
    Evaluate probes broken down by data group.

    Returns:
        Dict mapping group_name -> {accuracy, tpr, fpr, n_examples}
    """
    # Group test pairs by group
    group_pairs = defaultdict(list)
    for pid in test_pair_ids:
        meta = pair_metadata.get(pid, {})
        group = meta.get("group", "unknown")
        group_pairs[group].append(pid)

    probe_dir = os.path.join(probe_base_dir, position)
    try:
        probes = load_probes(probe_dir, layer_idx, seeds, hidden_dim)
    except FileNotFoundError:
        return {}

    scaler_mean, scaler_scale = None, None
    scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
    if os.path.exists(scaler_path):
        scaler_data = torch.load(scaler_path, weights_only=True)
        scaler_mean = scaler_data["scaler_mean"]
        scaler_scale = scaler_data["scaler_scale"]

    group_results = {}

    for group_name, group_pids in sorted(group_pairs.items()):
        if not group_pids:
            continue

        acts, labels = load_activations_for_position(
            data_dir, position, group_pids, [layer_idx],
        )

        if layer_idx not in acts or len(labels) == 0:
            continue

        result = evaluate_ensemble(
            probes, acts[layer_idx], labels,
            scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        )

        labels_np = labels.cpu().numpy()
        preds_np = result["majority_vote"].cpu().numpy()
        tpr_fpr = compute_tpr_fpr(labels_np, preds_np)

        group_results[group_name] = {
            "accuracy": result.get("ensemble_accuracy"),
            "auc_roc": result.get("ensemble_auc_roc"),
            "tpr": tpr_fpr["tpr"],
            "fpr": tpr_fpr["fpr"],
            "n_examples": len(labels),
            "n_pairs": len(group_pids),
        }

    return group_results


def evaluate_prefix_robustness(
    data_dir: str,
    probe_base_dir: str,
    position: str,
    layer_idx: int,
    seeds: List[int],
    hidden_dim: int,
    test_pair_ids: List[int],
    pair_metadata: Dict[int, dict],
) -> Dict[str, dict]:
    """
    For handwritten test pairs, report per-prefix-style accuracy/TPR/FPR.

    Returns:
        Dict mapping style -> {accuracy, tpr, fpr, n_examples}
    """
    # Filter to handwritten + prefix variants in test set
    style_pairs = defaultdict(list)
    for pid in test_pair_ids:
        meta = pair_metadata.get(pid, {})
        group = meta.get("group", "")
        if group in ("A", "A_prefix"):
            style = meta.get("elicitation_style", "default")
            style_pairs[style].append(pid)

    if not style_pairs:
        return {}

    probe_dir = os.path.join(probe_base_dir, position)
    try:
        probes = load_probes(probe_dir, layer_idx, seeds, hidden_dim)
    except FileNotFoundError:
        return {}

    scaler_mean, scaler_scale = None, None
    scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
    if os.path.exists(scaler_path):
        scaler_data = torch.load(scaler_path, weights_only=True)
        scaler_mean = scaler_data["scaler_mean"]
        scaler_scale = scaler_data["scaler_scale"]

    style_results = {}

    for style, style_pids in sorted(style_pairs.items()):
        acts, labels = load_activations_for_position(
            data_dir, position, style_pids, [layer_idx],
        )

        if layer_idx not in acts or len(labels) == 0:
            continue

        result = evaluate_ensemble(
            probes, acts[layer_idx], labels,
            scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        )

        labels_np = labels.cpu().numpy()
        preds_np = result["majority_vote"].cpu().numpy()
        tpr_fpr = compute_tpr_fpr(labels_np, preds_np)

        style_results[style] = {
            "accuracy": result.get("ensemble_accuracy"),
            "tpr": tpr_fpr["tpr"],
            "fpr": tpr_fpr["fpr"],
            "n_examples": len(labels),
        }

    return style_results


def evaluate_combiner(
    data_dir: str,
    probe_base_dir: str,
    combo_config_path: str,
    test_pair_ids: List[int],
    seeds: List[int],
    hidden_dim: int,
) -> Dict:
    """
    Evaluate the learned combiner on the test set.

    Returns:
        Dict with combiner metrics and operating point performance.
    """
    combo = load_weighted_combination(combo_config_path)
    positions = combo["positions"]
    layers = combo["layers"]

    # Load probes and scalers for the combiner's layers
    # The v2 combiner uses a single position with multiple layers
    position = positions[0]

    probe_dir = os.path.join(probe_base_dir, position)
    probes_by_layer = {}
    scalers_by_layer = {}

    for layer_idx in layers:
        try:
            probes_by_layer[layer_idx] = load_probes(probe_dir, layer_idx, seeds, hidden_dim)
        except FileNotFoundError:
            print(f"  WARNING: Missing probes for layer {layer_idx}")
            return {}

        scaler_mean, scaler_scale = None, None
        scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
        if os.path.exists(scaler_path):
            scaler_data = torch.load(scaler_path, weights_only=True)
            scaler_mean = scaler_data["scaler_mean"]
            scaler_scale = scaler_data["scaler_scale"]
        scalers_by_layer[layer_idx] = (scaler_mean, scaler_scale)

    # Score each test example
    all_scores = []
    all_labels = []

    for pair_id in test_pair_ids:
        for label_name, label_val in [("tree", 1), ("non_tree", 0)]:
            prompt_id = f"{label_name}_{pair_id:04d}"

            # Build probe_results dict for this example
            probe_results = {}
            probe_results[position] = {}

            valid = True
            for layer_idx in layers:
                act_path = os.path.join(
                    data_dir, "activations", label_name, position, f"{prompt_id}.pt"
                )
                if not os.path.exists(act_path):
                    valid = False
                    break

                layer_acts_dict = torch.load(act_path, weights_only=True)
                if layer_idx not in layer_acts_dict:
                    valid = False
                    break

                act = layer_acts_dict[layer_idx].float().unsqueeze(0)
                probes = probes_by_layer[layer_idx]
                scaler_mean, scaler_scale = scalers_by_layer[layer_idx]

                result = evaluate_ensemble(
                    probes, act, None,
                    scaler_mean=scaler_mean, scaler_scale=scaler_scale,
                )
                probe_results[position][layer_idx] = {
                    "mean_confidence": float(result["mean_confidence"].item()),
                }

            if not valid:
                continue

            combo_result = score_weighted_combination(probe_results, combo)
            all_scores.append(combo_result["combined_score"])
            all_labels.append(label_val)

    if not all_scores:
        return {}

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = None

    # Evaluate at each operating point
    op_results = {}
    for op_name, op in combo["operating_points"].items():
        threshold = op["threshold"]
        preds = (scores >= threshold).astype(int)
        tpr_fpr = compute_tpr_fpr(labels, preds)
        op_results[op_name] = {
            "threshold": threshold,
            "tpr": tpr_fpr["tpr"],
            "fpr": tpr_fpr["fpr"],
            "accuracy": float(np.mean(preds == labels)),
            "tp": tpr_fpr["tp"],
            "fp": tpr_fpr["fp"],
            "fn": tpr_fpr["fn"],
            "tn": tpr_fpr["tn"],
        }

    return {
        "auc_roc": round(auc, 4) if auc is not None else None,
        "n_examples": len(labels),
        "operating_points": op_results,
    }


def parse_supplementary_yaml(yaml_path: str) -> List[dict]:
    """Parse supplementary YAML to get examples."""
    items = []
    current = {}

    with open(yaml_path, "r") as f:
        text = f.read()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current:
                items.append(current)
            current = {"id": stripped.split(":", 1)[1].strip().strip('"')}
        elif stripped.startswith("question:") and current:
            val = stripped.split(":", 1)[1].strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            current["question"] = val
        elif stripped.startswith("label:") and current:
            current["label"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("category:") and current:
            current["category"] = stripped.split(":", 1)[1].strip()

    if current:
        items.append(current)
    return items


def load_supplementary_activations(
    data_dir: str,
    position: str,
    examples: List[dict],
    target_layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[dict]]:
    """Load activations for supplementary examples."""
    collected = {layer_idx: [] for layer_idx in target_layers}
    labels = []
    loaded_examples = []

    for ex in examples:
        label_name = "tree" if ex["label"] == 1 else "non_tree"
        act_path = os.path.join(
            data_dir, "activations", label_name, position, f"{ex['id']}.pt"
        )

        if not os.path.exists(act_path):
            continue

        layer_acts = torch.load(act_path, weights_only=True)
        for layer_idx in target_layers:
            if layer_idx in layer_acts:
                collected[layer_idx].append(layer_acts[layer_idx].float())

        labels.append(ex["label"])
        loaded_examples.append(ex)

    result = {}
    for layer_idx in target_layers:
        if collected[layer_idx]:
            result[layer_idx] = torch.stack(collected[layer_idx])

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return result, labels_tensor, loaded_examples


def evaluate_adversarial_set(
    data_dir: str,
    adversarial_path: str,
    probe_base_dir: str,
    target_layers: List[int],
    token_positions: List[str],
    seeds: List[int],
    hidden_dim: int,
) -> Dict:
    """Evaluate probes on the adversarial test set with per-category metrics."""
    examples = parse_supplementary_yaml(adversarial_path)
    if not examples:
        print("  No adversarial examples found.")
        return {}

    categories = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ex)

    print(f"  Adversarial set: {len(examples)} examples across {len(categories)} categories")
    for cat, exs in sorted(categories.items()):
        n_pos = sum(1 for e in exs if e["label"] == 1)
        n_neg = sum(1 for e in exs if e["label"] == 0)
        print(f"    {cat}: {len(exs)} ({n_pos} tree, {n_neg} non-tree)")

    results = {}

    for position in token_positions:
        print(f"\n  Position: {position}")
        adv_acts, adv_labels, loaded = load_supplementary_activations(
            data_dir, position, examples, target_layers,
        )

        if len(adv_labels) == 0:
            print(f"    No activations found for position {position}")
            continue

        print(f"    Loaded activations for {len(adv_labels)} examples")

        position_results = {}
        for layer_idx in target_layers:
            if layer_idx not in adv_acts:
                continue

            probe_dir = os.path.join(probe_base_dir, position)
            try:
                probes = load_probes(probe_dir, layer_idx, seeds, hidden_dim)
            except FileNotFoundError:
                continue

            scaler_mean, scaler_scale = None, None
            scaler_path = os.path.join(probe_dir, f"layer{layer_idx}_scaler.pt")
            if os.path.exists(scaler_path):
                scaler_data = torch.load(scaler_path, weights_only=True)
                scaler_mean = scaler_data["scaler_mean"]
                scaler_scale = scaler_data["scaler_scale"]

            ensemble_result = evaluate_ensemble(
                probes, adv_acts[layer_idx], adv_labels,
                scaler_mean=scaler_mean, scaler_scale=scaler_scale,
            )

            # Per-category
            cat_metrics = {}
            for cat, cat_examples in categories.items():
                cat_indices = [i for i, ex in enumerate(loaded) if ex.get("category") == cat]
                if not cat_indices:
                    continue

                cat_acts_list = [adv_acts[layer_idx][i] for i in cat_indices]
                cat_labels_list = [adv_labels[i].item() for i in cat_indices]

                if len(cat_acts_list) == 0:
                    continue

                cat_acts_tensor = torch.stack(cat_acts_list)
                cat_labels_tensor = torch.tensor(cat_labels_list, dtype=torch.float32)

                cat_result = evaluate_ensemble(
                    probes, cat_acts_tensor, cat_labels_tensor,
                    scaler_mean=scaler_mean, scaler_scale=scaler_scale,
                )
                cat_metrics[cat] = {
                    "accuracy": cat_result.get("ensemble_accuracy"),
                    "n_examples": len(cat_indices),
                    "mean_confidence": cat_result.get("mean_confidence"),
                }

            layer_result = {
                "overall_accuracy": ensemble_result.get("ensemble_accuracy"),
                "overall_auc_roc": ensemble_result.get("ensemble_auc_roc"),
                "per_category": cat_metrics,
            }
            position_results[layer_idx] = layer_result

            print(f"    Layer {layer_idx}: Acc={layer_result['overall_accuracy']:.3f}")
            for cat, cm in cat_metrics.items():
                print(f"      {cat}: Acc={cm['accuracy']:.3f} (n={cm['n_examples']})")

        results[position] = position_results

    return results


def print_results_table(
    qa_results: Dict,
    target_layers: List[int],
    token_positions: List[str],
):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("QA-TRAINED PROBES — Test Set Performance")
    print("=" * 70)

    layer_headers = "  ".join(f"L{l:>3}" for l in target_layers)
    print(f"\n{'Position (Accuracy)':<30} {layer_headers}")
    print("-" * (30 + len(target_layers) * 6))

    for position in token_positions:
        if position not in qa_results:
            continue
        accs = []
        for layer_idx in target_layers:
            if layer_idx in qa_results[position]:
                accs.append(f"{qa_results[position][layer_idx]['ensemble_accuracy']:.3f}")
            else:
                accs.append("  N/A")
        print(f"{position:<30} {'  '.join(accs)}")

    # AUC table
    print(f"\n{'Position (AUC-ROC)':<30} {layer_headers}")
    print("-" * (30 + len(target_layers) * 6))

    for position in token_positions:
        if position not in qa_results:
            continue
        aucs = []
        for layer_idx in target_layers:
            if layer_idx in qa_results[position]:
                auc = qa_results[position][layer_idx].get('ensemble_auc_roc')
                aucs.append(f"{auc:.3f}" if auc is not None else "  N/A")
            else:
                aucs.append("  N/A")
        print(f"{position:<30} {'  '.join(aucs)}")

    # TPR/FPR table
    print(f"\n{'Position (TPR/FPR)':<30} {layer_headers}")
    print("-" * (30 + len(target_layers) * 6))

    for position in token_positions:
        if position not in qa_results:
            continue
        vals = []
        for layer_idx in target_layers:
            r = qa_results[position].get(layer_idx, {})
            if "tpr" in r:
                vals.append(f"{r['tpr']:.2f}/{r['fpr']:.2f}")
            else:
                vals.append("   N/A")
        # Wider spacing for TPR/FPR
        print(f"{position:<30} {'  '.join(vals)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA-trained probes and produce report"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to qa_probe_training.yaml config")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: data-dir/logs)")
    parser.add_argument("--adversarial-set", type=str, default=None,
                        help="Path to adversarial_test_set.yaml for adversarial evaluation")
    parser.add_argument("--probe-dir", type=str, default=None,
                        help="Override probe directory (default: data-dir/probes)")
    args = parser.parse_args()

    # Resolve paths
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)
    config = load_config(args.config)

    model_config_path = resolve_path(config["model_config"])
    model_config = load_config(model_config_path)

    data_dir = args.data_dir or config["storage"]["base_dir"]
    probe_base_dir = args.probe_dir or os.path.join(data_dir, "probes")

    # Set up logging
    log_dir = args.log_dir or os.path.join(data_dir, "logs")
    setup_logging(log_dir, "evaluate_probes_qa")
    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))
    token_positions = config["token_positions"]
    seeds = config["probe_training"]["random_seeds"]
    hidden_dim = model_config["hidden_dim"]
    split_config = config["split"]
    combiner_config = config.get("combiner", {})

    # Load split info
    print("=" * 60)
    print("Step 1: Loading split info")
    print("=" * 60)

    split_method = split_config.get("method", "random")

    if split_method == "preassigned":
        split_file = split_config.get("split_file")
        if split_file:
            split_file = resolve_path(split_file)

        if split_file and os.path.exists(split_file):
            print(f"  Using pre-assigned splits from {split_file}")
            # Load generation log to get available pair IDs
            log_path = os.path.join(data_dir, "prompts", "generation_log.jsonl")
            available_ids = _get_available_pair_ids(log_path)
            splits = load_preassigned_splits(split_file, available_ids)
            test_pair_ids = splits["test"]
        else:
            # Fall back to split_info.json
            split_info_path = os.path.join(data_dir, "split_info.json")
            if not os.path.exists(split_info_path):
                raise FileNotFoundError(f"Split info not found: {split_info_path}. Run train_probes_qa.py first.")
            with open(split_info_path, "r") as f:
                split_info = json.load(f)
            test_pair_ids = split_info["splits"]["test"]
    else:
        split_info_path = os.path.join(data_dir, "split_info.json")
        if not os.path.exists(split_info_path):
            raise FileNotFoundError(f"Split info not found: {split_info_path}. Run train_probes_qa.py first.")
        with open(split_info_path, "r") as f:
            split_info = json.load(f)
        test_pair_ids = split_info["splits"]["test"]

    print(f"  Test pairs: {len(test_pair_ids)} ({len(test_pair_ids) * 2} examples)")

    # Load generation metadata for per-group evaluation
    pair_metadata = load_generation_metadata(data_dir)
    print(f"  Loaded metadata for {len(pair_metadata)} pairs")

    # Evaluate QA-trained probes
    print("\n" + "=" * 60)
    print("Step 2: Evaluating QA-trained probes")
    print("=" * 60)
    qa_results = evaluate_all_probes(
        data_dir, probe_base_dir, target_layers, token_positions,
        seeds, hidden_dim, test_pair_ids,
    )

    # Print results
    print_results_table(qa_results, target_layers, token_positions)

    # Per-group evaluation (for best position/layer)
    primary_position = config.get("probe_training", {}).get("primary_position", "answer_mean_pool")
    if primary_position in qa_results and pair_metadata:
        print("\n" + "=" * 60)
        print("Step 3: Per-group evaluation")
        print("=" * 60)

        # Find best layer for primary position
        best_layer = None
        best_auc = 0
        for layer_idx, metrics in qa_results.get(primary_position, {}).items():
            auc = metrics.get("ensemble_auc_roc", 0)
            if auc and auc > best_auc:
                best_auc = auc
                best_layer = layer_idx

        if best_layer is not None:
            print(f"  Position: {primary_position}, Layer: {best_layer} (best test AUC: {best_auc:.3f})")

            group_results = evaluate_per_group(
                data_dir, probe_base_dir, primary_position, best_layer,
                seeds, hidden_dim, test_pair_ids, pair_metadata,
            )

            if group_results:
                print(f"\n  {'Group':<15} {'Acc':>8} {'TPR':>8} {'FPR':>8} {'AUC':>8} {'N':>6}")
                print("  " + "-" * 55)
                for group, gm in sorted(group_results.items()):
                    auc_str = f"{gm['auc_roc']:.3f}" if gm.get('auc_roc') is not None else "N/A"
                    print(f"  {group:<15} {gm['accuracy']:>8.3f} {gm['tpr']:>8.3f} {gm['fpr']:>8.3f} {auc_str:>8} {gm['n_examples']:>6}")

    # Prefix robustness table
    if primary_position in qa_results and pair_metadata:
        best_layer_for_prefix = best_layer  # reuse from above

        if best_layer_for_prefix is not None:
            print("\n" + "=" * 60)
            print("Step 4: Prefix robustness (handwritten test pairs)")
            print("=" * 60)

            style_results = evaluate_prefix_robustness(
                data_dir, probe_base_dir, primary_position, best_layer_for_prefix,
                seeds, hidden_dim, test_pair_ids, pair_metadata,
            )

            if style_results:
                print(f"  Position: {primary_position}, Layer: {best_layer_for_prefix}")
                print(f"\n  {'Style':<20} {'Acc':>8} {'TPR':>8} {'FPR':>8} {'N':>6}")
                print("  " + "-" * 50)
                for style, sm in sorted(style_results.items()):
                    print(f"  {style:<20} {sm['accuracy']:>8.3f} {sm['tpr']:>8.3f} {sm['fpr']:>8.3f} {sm['n_examples']:>6}")
            else:
                print("  No handwritten pairs in test set (prefix robustness N/A)")

    # Combiner evaluation
    if combiner_config.get("enabled", False):
        combo_path = combiner_config.get("output_path", "configs/probes/weighted_combination_v2.yaml")
        combo_path = resolve_path(combo_path)

        if os.path.exists(combo_path):
            print("\n" + "=" * 60)
            print("Step 5: Combiner evaluation")
            print("=" * 60)

            combo_results = evaluate_combiner(
                data_dir, probe_base_dir, combo_path,
                test_pair_ids, seeds, hidden_dim,
            )

            if combo_results:
                print(f"  Combiner test AUC: {combo_results['auc_roc']}")
                print(f"  N test examples: {combo_results['n_examples']}")
                print(f"\n  {'Op Point':<15} {'Threshold':>10} {'TPR':>8} {'FPR':>8} {'Accuracy':>10}")
                print("  " + "-" * 55)
                for op_name, op in combo_results["operating_points"].items():
                    print(f"  {op_name:<15} {op['threshold']:>10.4f} {op['tpr']:>8.3f} {op['fpr']:>8.3f} {op['accuracy']:>10.3f}")

    # Adversarial evaluation (optional)
    adversarial_results = {}
    adversarial_path = args.adversarial_set
    if not adversarial_path:
        supp_config = config.get("supplementary", {})
        adv_default = supp_config.get("adversarial_test_set")
        if adv_default:
            adversarial_path = resolve_path(adv_default)

    if adversarial_path and os.path.exists(adversarial_path):
        print("\n" + "=" * 60)
        print("Step 6: Adversarial test set evaluation")
        print("=" * 60)
        adversarial_results = evaluate_adversarial_set(
            data_dir, adversarial_path, probe_base_dir,
            target_layers, token_positions, seeds, hidden_dim,
        )

    # Save full report
    report = {
        "qa_probe_results": qa_results,
        "test_pair_ids": test_pair_ids,
        "n_test_pairs": len(test_pair_ids),
        "target_layers": target_layers,
        "token_positions": token_positions,
        "seeds": seeds,
    }
    if adversarial_results:
        report["adversarial_results"] = adversarial_results

    report_path = os.path.join(data_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")

    # Find best position x layer combination
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS")
    print("=" * 60)

    best_acc = 0
    best_config_found = None
    for position in token_positions:
        for layer_idx in target_layers:
            acc = qa_results.get(position, {}).get(layer_idx, {}).get("ensemble_accuracy")
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_config_found = (position, layer_idx)

    if best_config_found:
        pos, layer = best_config_found
        metrics = qa_results[pos][layer]
        print(f"  Best accuracy: {pos} @ layer {layer}")
        print(f"    Accuracy: {metrics['ensemble_accuracy']:.3f}")
        print(f"    AUC-ROC:  {metrics['ensemble_auc_roc']:.3f}")
        print(f"    F1:       {metrics['ensemble_f1']:.3f}")
        print(f"    TPR:      {metrics.get('tpr', 'N/A')}")
        print(f"    FPR:      {metrics.get('fpr', 'N/A')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
