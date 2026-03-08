"""Evaluation functions for trained probes."""

import math
import torch
import yaml
import numpy as np
from typing import Dict, List, Optional
from .architectures import LinearProbe


def evaluate_probe(probe: LinearProbe, activations, labels=None,
                   scaler_mean=None, scaler_scale=None) -> Dict:
    """
    Evaluate a single probe on activation data.

    Args:
        probe: Trained LinearProbe instance.
        activations: Tensor of shape (n_samples, hidden_dim) or
                    (hidden_dim,) for single sample.
        labels: Optional binary tensor for computing metrics.
        scaler_mean: Optional per-feature mean for normalization.
        scaler_scale: Optional per-feature std for normalization.

    Returns:
        Dict containing:
        - activation_strength: Raw probe output (before sigmoid)
        - predicted_label: Binary prediction (0 or 1)
        - confidence: Probability from sigmoid of activation strength
        - accuracy, auc_roc, f1: (if labels provided)
    """
    if scaler_mean is not None and scaler_scale is not None:
        activations = (activations - scaler_mean) / scaler_scale

    probe.eval()
    with torch.no_grad():
        logits = probe(activations)
        confidence = torch.sigmoid(logits)
        predicted = (confidence >= 0.5).long()

    result = {
        "activation_strength": logits,
        "predicted_label": predicted,
        "confidence": confidence,
    }

    if labels is not None:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        labels_np = labels.cpu().numpy()
        preds_np = predicted.cpu().numpy()
        conf_np = confidence.cpu().numpy()

        result["accuracy"] = accuracy_score(labels_np, preds_np)
        result["f1"] = f1_score(labels_np, preds_np)
        try:
            result["auc_roc"] = roc_auc_score(labels_np, conf_np)
        except ValueError:
            result["auc_roc"] = None

    return result


def evaluate_ensemble(probes: List[LinearProbe], activations, labels=None,
                      scaler_mean=None, scaler_scale=None) -> Dict:
    """
    Evaluate a probe ensemble and aggregate results.

    Normalization is applied once before passing to individual probes.

    Args:
        probes: List of trained LinearProbe instances.
        activations: Tensor of shape (n_samples, hidden_dim) or
                    (hidden_dim,) for single sample.
        labels: Optional binary tensor for computing metrics.
        scaler_mean: Optional per-feature mean for normalization.
        scaler_scale: Optional per-feature std for normalization.

    Returns:
        Dict containing:
        - mean_activation: Mean activation strength across probes
        - std_activation: Std dev of activation strengths
        - mean_confidence: Mean confidence across probes
        - majority_vote: Predicted label by majority vote
        - agreement_ratio: Fraction of probes agreeing with majority
        - individual_results: List of per-probe evaluation dicts
    """
    # Normalize once, then pass without scaler to individual evaluations
    if scaler_mean is not None and scaler_scale is not None:
        activations = (activations - scaler_mean) / scaler_scale

    individual_results = [evaluate_probe(p, activations, labels) for p in probes]

    all_logits = torch.stack([r["activation_strength"] for r in individual_results])
    all_conf = torch.stack([r["confidence"] for r in individual_results])
    all_preds = torch.stack([r["predicted_label"] for r in individual_results])

    mean_activation = all_logits.float().mean(dim=0)
    std_activation = all_logits.float().std(dim=0)
    mean_confidence = all_conf.mean(dim=0)

    # Majority vote
    vote_sum = all_preds.float().mean(dim=0)
    majority_vote = (vote_sum >= 0.5).long()
    agreement_ratio = torch.where(
        majority_vote == 1, vote_sum, 1.0 - vote_sum
    ).mean().item()

    result = {
        "mean_activation": mean_activation,
        "std_activation": std_activation,
        "mean_confidence": mean_confidence,
        "majority_vote": majority_vote,
        "agreement_ratio": agreement_ratio,
        "individual_results": individual_results,
    }

    if labels is not None:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        labels_np = labels.cpu().numpy()
        majority_np = majority_vote.cpu().numpy()
        mean_conf_np = mean_confidence.cpu().numpy()

        result["ensemble_accuracy"] = accuracy_score(labels_np, majority_np)
        result["ensemble_f1"] = f1_score(labels_np, majority_np)
        try:
            result["ensemble_auc_roc"] = roc_auc_score(labels_np, mean_conf_np)
        except ValueError:
            result["ensemble_auc_roc"] = None

        if individual_results[0].get("accuracy") is not None:
            result["individual_accuracies"] = [r["accuracy"] for r in individual_results]
            result["individual_auc_rocs"] = [r["auc_roc"] for r in individual_results]
            result["mean_individual_accuracy"] = sum(result["individual_accuracies"]) / len(result["individual_accuracies"])

    return result


def load_weighted_combination(config_path: str) -> Dict:
    """
    Load a learned weighted probe combination config from YAML.

    Args:
        config_path: Path to weighted_combination.yaml.

    Returns:
        Dict with numpy arrays ready for scoring:
        - weights: (20,) array of logistic regression coefficients
        - intercept: float
        - scaler_mean: (20,) array
        - scaler_scale: (20,) array
        - positions: list of position names
        - layers: list of layer indices (ints)
        - operating_points: dict of {name: {threshold, expected_tpr, expected_fpr}}
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return {
        "weights": np.array(config["weights"], dtype=np.float64),
        "intercept": float(config["intercept"]),
        "scaler_mean": np.array(config["scaler"]["mean"], dtype=np.float64),
        "scaler_scale": np.array(config["scaler"]["scale"], dtype=np.float64),
        "positions": config["feature_order"]["positions"],
        "layers": config["feature_order"]["layers"],
        "operating_points": config["operating_points"],
    }


def score_weighted_combination(probe_results: Dict, combo: Dict) -> Dict:
    """
    Score a single trial's probe results using a learned weighted combination.

    Takes the per-position, per-layer mean_confidence values from existing probe
    results and combines them into a single calibrated probability using the
    learned logistic regression weights.

    Args:
        probe_results: Dict of {position: {layer: {mean_confidence, ...}}}.
            Layer keys may be strings or ints.
        combo: Output of load_weighted_combination().

    Returns:
        Dict with:
        - combined_score: float probability (0-1)
        - detections: {op_name: bool} for each operating point
        - feature_vector: list of 20 raw confidence values (for debugging)
    """
    positions = combo["positions"]
    layers = combo["layers"]

    # Extract 20-dim feature vector in the correct order
    feature_vec = []
    for pos in positions:
        pos_results = probe_results.get(pos, {})
        for layer in layers:
            layer_data = pos_results.get(str(layer)) or pos_results.get(layer, {})
            mc = layer_data.get("mean_confidence", 0.0) if layer_data else 0.0
            feature_vec.append(mc)

    x = np.array(feature_vec, dtype=np.float64)

    # StandardScaler transform
    x_scaled = (x - combo["scaler_mean"]) / combo["scaler_scale"]

    # Logistic regression: sigmoid(w @ x + b)
    logit = float(np.dot(combo["weights"], x_scaled) + combo["intercept"])
    score = 1.0 / (1.0 + math.exp(-logit))

    # Check each operating point
    detections = {}
    for op_name, op in combo["operating_points"].items():
        detections[op_name] = score >= op["threshold"]

    return {
        "combined_score": round(score, 6),
        "detections": detections,
        "feature_vector": [round(v, 4) for v in feature_vec],
    }
