"""Probe architecture definitions."""

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
