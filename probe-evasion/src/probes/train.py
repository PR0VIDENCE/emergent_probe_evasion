"""Training functions for linear probes."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from .architectures import LinearProbe


def train_probe(activations, labels, config) -> LinearProbe:
    """
    Train a single linear probe on activation data.

    Args:
        activations: Tensor of shape (n_samples, hidden_dim) containing
                    model activations at a specific layer.
        labels: Binary tensor of shape (n_samples,) indicating presence/absence
               of the target concept.
        config: Dict containing training hyperparameters:
               - learning_rate (default 1e-3)
               - num_epochs (default 100)
               - weight_decay (default 0.01)
               - random_seed (default 42)
               - patience (default 10, for early stopping)
               - val_activations (optional tensor for early stopping)
               - val_labels (optional tensor for early stopping)

    Returns:
        Trained LinearProbe instance.
    """
    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)

    hidden_dim = activations.shape[1]
    probe = LinearProbe(hidden_dim)

    device = activations.device
    probe = probe.to(device)

    lr = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 0.01)
    num_epochs = config.get("num_epochs", 100)
    patience = config.get("patience", 10)

    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    val_acts = config.get("val_activations", None)
    val_labels_t = config.get("val_labels", None)
    use_early_stopping = val_acts is not None and val_labels_t is not None

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    labels_float = labels.float()

    for epoch in range(num_epochs):
        probe.train()
        optimizer.zero_grad()

        logits = probe(activations)
        loss = criterion(logits, labels_float)
        loss.backward()
        optimizer.step()

        if use_early_stopping:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_acts)
                val_loss = criterion(val_logits, val_labels_t.float())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    if use_early_stopping and best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    return probe


def train_probe_ensemble(activations, labels, config) -> List[LinearProbe]:
    """
    Train an ensemble of probes with different random seeds.

    Used to get robust estimates of probe accuracy and to enable
    ensemble-based detection methods.

    Args:
        activations: Tensor of shape (n_samples, hidden_dim).
        labels: Binary tensor of shape (n_samples,).
        config: Dict containing:
               - random_seeds: List of seeds for reproducibility
               - (other training hyperparameters)

    Returns:
        List of trained LinearProbe instances.
    """
    seeds = config.get("random_seeds", [42, 123, 456, 789])
    probes = []
    for seed in seeds:
        probe_config = {**config, "random_seed": seed}
        probe = train_probe(activations, labels, probe_config)
        probes.append(probe)
    return probes
