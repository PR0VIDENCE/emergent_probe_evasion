"""Evaluation functions for trained probes."""

import torch
from typing import Dict, List
from .architectures import LinearProbe


def evaluate_probe(probe: LinearProbe, activations, labels=None) -> Dict:
    """
    Evaluate a single probe on activation data.

    Args:
        probe: Trained LinearProbe instance.
        activations: Tensor of shape (n_samples, hidden_dim) or
                    (hidden_dim,) for single sample.
        labels: Optional binary tensor for computing metrics.

    Returns:
        Dict containing:
        - activation_strength: Raw probe output (before sigmoid)
        - predicted_label: Binary prediction (0 or 1)
        - confidence: Probability from sigmoid of activation strength
        - accuracy, auc_roc, f1: (if labels provided)
    """
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


def evaluate_ensemble(probes: List[LinearProbe], activations, labels=None) -> Dict:
    """
    Evaluate a probe ensemble and aggregate results.

    Args:
        probes: List of trained LinearProbe instances.
        activations: Tensor of shape (n_samples, hidden_dim) or
                    (hidden_dim,) for single sample.
        labels: Optional binary tensor for computing metrics.

    Returns:
        Dict containing:
        - mean_activation: Mean activation strength across probes
        - std_activation: Std dev of activation strengths
        - mean_confidence: Mean confidence across probes
        - majority_vote: Predicted label by majority vote
        - agreement_ratio: Fraction of probes agreeing with majority
        - individual_results: List of per-probe evaluation dicts
    """
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
