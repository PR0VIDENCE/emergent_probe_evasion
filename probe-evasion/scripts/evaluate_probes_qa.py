"""
Evaluate QA-trained probes and produce comprehensive report.

Evaluates all probes (4 positions x 5 layers x 4 seeds = 80 probes) on the test set.
Optionally compares against statement-trained probes on QA data to quantify
the distribution shift.

Usage:
    python scripts/evaluate_probes_qa.py \
        --config configs/experiments/qa_probe_training.yaml

    # Also compare against statement-trained probes:
    python scripts/evaluate_probes_qa.py \
        --config configs/experiments/qa_probe_training.yaml \
        --statement-probe-dir data/outputs/preliminary_qwq/probes
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.probes.architectures import LinearProbe
from src.probes.evaluate import evaluate_probe, evaluate_ensemble


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def evaluate_all_probes(
    data_dir: str,
    target_layers: List[int],
    token_positions: List[str],
    seeds: List[int],
    hidden_dim: int,
    test_pair_ids: List[int],
) -> Dict:
    """
    Evaluate all QA-trained probes on the test set.

    Returns:
        Nested dict: {position: {layer: {metrics}}}
    """
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

            probe_dir = os.path.join(data_dir, "probes", position)
            try:
                probes = load_probes(probe_dir, layer_idx, seeds, hidden_dim)
            except FileNotFoundError as e:
                print(f"  Layer {layer_idx}: {e}")
                continue

            # Evaluate ensemble
            ensemble_result = evaluate_ensemble(probes, test_acts[layer_idx], test_labels)

            # Evaluate individual probes
            individual_metrics = []
            for i, probe in enumerate(probes):
                ind_result = evaluate_probe(probe, test_acts[layer_idx], test_labels)
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
                "individual_probes": individual_metrics,
                "n_test_examples": len(test_labels),
            }
            position_results[layer_idx] = layer_result

            print(f"  Layer {layer_idx}: "
                  f"Acc={layer_result['ensemble_accuracy']:.3f}, "
                  f"AUC={layer_result['ensemble_auc_roc']:.3f}, "
                  f"F1={layer_result['ensemble_f1']:.3f}, "
                  f"Agree={layer_result['agreement_ratio']:.3f}")

        results[position] = position_results

    return results


def evaluate_statement_probes_on_qa(
    statement_probe_dir: str,
    data_dir: str,
    target_layers: List[int],
    token_positions: List[str],
    seeds: List[int],
    hidden_dim: int,
    test_pair_ids: List[int],
) -> Dict:
    """
    Evaluate statement-trained probes on QA test data.

    This quantifies the distribution shift: how much worse do probes trained
    on short statements perform when evaluated on long QwQ-32B generations?

    Returns:
        Nested dict: {position: {layer: {metrics}}}
    """
    results = {}

    for position in token_positions:
        print(f"\n--- Position: {position} (statement probes) ---")

        test_acts, test_labels = load_activations_for_position(
            data_dir, position, test_pair_ids, target_layers,
        )

        if len(test_labels) == 0:
            print(f"  No test data for position {position}")
            continue

        position_results = {}

        for layer_idx in target_layers:
            if layer_idx not in test_acts:
                continue

            try:
                probes = load_probes(statement_probe_dir, layer_idx, seeds, hidden_dim)
            except FileNotFoundError:
                print(f"  Layer {layer_idx}: statement probes not found")
                continue

            ensemble_result = evaluate_ensemble(probes, test_acts[layer_idx], test_labels)

            layer_result = {
                "ensemble_accuracy": ensemble_result.get("ensemble_accuracy"),
                "ensemble_f1": ensemble_result.get("ensemble_f1"),
                "ensemble_auc_roc": ensemble_result.get("ensemble_auc_roc"),
                "agreement_ratio": ensemble_result.get("agreement_ratio"),
            }
            position_results[layer_idx] = layer_result

            print(f"  Layer {layer_idx}: "
                  f"Acc={layer_result['ensemble_accuracy']:.3f}, "
                  f"AUC={layer_result['ensemble_auc_roc']:.3f}, "
                  f"F1={layer_result['ensemble_f1']:.3f}")

        results[position] = position_results

    return results


def print_comparison_table(
    qa_results: Dict,
    statement_results: Optional[Dict],
    target_layers: List[int],
    token_positions: List[str],
):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("QA-TRAINED PROBES — Test Set Performance")
    print("=" * 70)

    layer_headers = "  ".join(f"L{l:>3}" for l in target_layers)
    print(f"\n{'Position':<30} {layer_headers}")
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

    if statement_results:
        print("\n" + "=" * 70)
        print("STATEMENT-TRAINED PROBES on QA Data (Distribution Shift)")
        print("=" * 70)

        print(f"\n{'Position':<30} {layer_headers}")
        print("-" * (30 + len(target_layers) * 6))

        for position in token_positions:
            if position not in statement_results:
                continue
            accs = []
            for layer_idx in target_layers:
                if layer_idx in statement_results[position]:
                    accs.append(f"{statement_results[position][layer_idx]['ensemble_accuracy']:.3f}")
                else:
                    accs.append("  N/A")
            print(f"{position:<30} {'  '.join(accs)}")

        # Delta table
        print("\n" + "=" * 70)
        print("DELTA: QA-trained minus Statement-trained (positive = QA better)")
        print("=" * 70)

        print(f"\n{'Position':<30} {layer_headers}")
        print("-" * (30 + len(target_layers) * 6))

        for position in token_positions:
            if position not in qa_results or position not in statement_results:
                continue
            deltas = []
            for layer_idx in target_layers:
                qa_acc = qa_results.get(position, {}).get(layer_idx, {}).get("ensemble_accuracy")
                st_acc = statement_results.get(position, {}).get(layer_idx, {}).get("ensemble_accuracy")
                if qa_acc is not None and st_acc is not None:
                    delta = qa_acc - st_acc
                    sign = "+" if delta >= 0 else ""
                    deltas.append(f"{sign}{delta:.3f}")
                else:
                    deltas.append("  N/A")
            print(f"{position:<30} {'  '.join(deltas)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA-trained probes and produce report"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to qa_probe_training.yaml config")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--statement-probe-dir", type=str, default=None,
                        help="Path to statement-trained probes for comparison")
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
    target_layers = config["target_layers"]
    token_positions = config["token_positions"]
    seeds = config["probe_training"]["random_seeds"]
    hidden_dim = model_config["hidden_dim"]

    # Load split info
    print("=" * 60)
    print("Step 1: Loading split info")
    print("=" * 60)
    split_info_path = os.path.join(data_dir, "split_info.json")
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(
            f"Split info not found: {split_info_path}. "
            f"Run train_probes_qa.py first."
        )

    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    test_pair_ids = split_info["splits"]["test"]
    print(f"  Test pairs: {len(test_pair_ids)} ({len(test_pair_ids) * 2} examples)")

    # Evaluate QA-trained probes
    print("\n" + "=" * 60)
    print("Step 2: Evaluating QA-trained probes")
    print("=" * 60)
    qa_results = evaluate_all_probes(
        data_dir, target_layers, token_positions, seeds, hidden_dim, test_pair_ids,
    )

    # Optionally evaluate statement-trained probes
    statement_results = None
    if args.statement_probe_dir:
        stmt_dir = resolve_path(args.statement_probe_dir)
        print("\n" + "=" * 60)
        print("Step 3: Evaluating statement-trained probes on QA data")
        print("=" * 60)
        statement_results = evaluate_statement_probes_on_qa(
            stmt_dir, data_dir, target_layers, token_positions,
            seeds, hidden_dim, test_pair_ids,
        )

    # Print comparison
    print_comparison_table(qa_results, statement_results, target_layers, token_positions)

    # Save full report
    report = {
        "qa_probe_results": qa_results,
        "statement_probe_results": statement_results,
        "test_pair_ids": test_pair_ids,
        "n_test_pairs": len(test_pair_ids),
        "target_layers": target_layers,
        "token_positions": token_positions,
        "seeds": seeds,
    }

    report_path = os.path.join(data_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")

    # Find best position x layer combination
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS")
    print("=" * 60)

    best_acc = 0
    best_config = None
    for position in token_positions:
        for layer_idx in target_layers:
            acc = qa_results.get(position, {}).get(layer_idx, {}).get("ensemble_accuracy")
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_config = (position, layer_idx)

    if best_config:
        pos, layer = best_config
        metrics = qa_results[pos][layer]
        print(f"  Best accuracy: {pos} @ layer {layer}")
        print(f"    Accuracy: {metrics['ensemble_accuracy']:.3f}")
        print(f"    AUC-ROC:  {metrics['ensemble_auc_roc']:.3f}")
        print(f"    F1:       {metrics['ensemble_f1']:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
