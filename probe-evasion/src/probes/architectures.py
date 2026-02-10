"""Probe architecture definitions."""

import pickle
from pathlib import Path

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Simple linear probe for binary concept detection.

    Maps hidden activations to a single scalar via learned weights,
    then applies sigmoid for probability output.

    Attributes:
        linear: nn.Linear layer mapping hidden_dim -> 1
        hidden_dim: Dimension of input activations
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize linear probe.

        Args:
            hidden_dim: Dimension of model hidden states.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, activations):
        """
        Forward pass through probe.

        Args:
            activations: Tensor of shape (..., hidden_dim).

        Returns:
            Tensor of shape (...,) with probe outputs (pre-sigmoid).
        """
        return self.linear(activations).squeeze(-1)

    @classmethod
    def from_config(cls, config: dict) -> "LinearProbe":
        """
        Instantiate probe from configuration dict.

        Args:
            config: Dict containing at least 'hidden_dim'.

        Returns:
            Initialized LinearProbe instance.
        """
        return cls(hidden_dim=config["hidden_dim"])


class SklearnLogisticProbe:
    """sklearn LogisticRegression probe (Apollo-style).

    Strong L2 regularization, no intercept, StandardScaler built-in.
    Self-contained: scaler is part of the probe, not managed externally.
    """

    def __init__(self, reg_coeff=10.0):
        self.reg_coeff = reg_coeff
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None

    def fit(self, activations, labels):
        """Fit the probe on training data.

        Args:
            activations: Tensor of shape (n_samples, hidden_dim).
            labels: Binary tensor of shape (n_samples,).
        """
        from sklearn.linear_model import LogisticRegression

        self.scaler_mean = activations.mean(dim=0)
        self.scaler_scale = activations.std(dim=0).clamp(min=1e-8)
        X = ((activations - self.scaler_mean) / self.scaler_scale).numpy()
        y = labels.numpy()
        self.model = LogisticRegression(
            C=1.0 / self.reg_coeff, fit_intercept=False,
            max_iter=1000, random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, activations):
        """Predict probabilities for the positive class.

        Args:
            activations: Tensor of shape (n_samples, hidden_dim) or (hidden_dim,).

        Returns:
            Tensor of shape (n_samples,) with positive-class probabilities.
        """
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        X = ((activations - self.scaler_mean) / self.scaler_scale).numpy()
        return torch.tensor(self.model.predict_proba(X)[:, 1], dtype=torch.float32)

    def save(self, path):
        """Save probe to disk via pickle."""
        data = {
            "reg_coeff": self.reg_coeff,
            "model": self.model,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load probe from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        probe = cls(reg_coeff=data["reg_coeff"])
        probe.model = data["model"]
        probe.scaler_mean = data["scaler_mean"]
        probe.scaler_scale = data["scaler_scale"]
        return probe


class MLPProbe(nn.Module):
    """
    Multi-layer perceptron probe for concept detection.

    More expressive than linear probe, useful for checking if
    concept is linearly represented or requires nonlinear readout.

    Attributes:
        layers: nn.Sequential containing MLP layers
        hidden_dim: Dimension of input activations
        mlp_hidden_dim: Dimension of MLP hidden layer
    """

    def __init__(self, hidden_dim: int, mlp_hidden_dim: int = 256):
        """
        Initialize MLP probe.

        Args:
            hidden_dim: Dimension of model hidden states.
            mlp_hidden_dim: Dimension of MLP hidden layer.
        """
        raise NotImplementedError("TODO")

    def forward(self, activations):
        """
        Forward pass through probe.

        Args:
            activations: Tensor of shape (..., hidden_dim).

        Returns:
            Tensor of shape (...,) with probe outputs (pre-sigmoid).
        """
        raise NotImplementedError("TODO")

    @classmethod
    def from_config(cls, config: dict) -> "MLPProbe":
        """
        Instantiate probe from configuration dict.

        Args:
            config: Dict containing 'hidden_dim' and optionally 'mlp_hidden_dim'.

        Returns:
            Initialized MLPProbe instance.
        """
        raise NotImplementedError("TODO")
