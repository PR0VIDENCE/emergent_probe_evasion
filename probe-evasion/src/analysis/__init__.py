"""Analysis and visualization modules."""

from .compare_conditions import compare_probe_activations
from .visualize import plot_activation_distributions, plot_condition_comparison

__all__ = [
    "compare_probe_activations",
    "plot_activation_distributions",
    "plot_condition_comparison",
]
