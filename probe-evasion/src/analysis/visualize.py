"""Visualization functions for probe activation analysis."""

from typing import Dict, List


def plot_activation_distributions(results: Dict, output_path: str) -> None:
    """
    Plot distribution of probe activations.

    Creates histogram/KDE plots showing the distribution of probe
    activation values, optionally split by ground truth label.

    Args:
        results: Dict containing:
                - activations: List of activation values
                - labels: Optional ground truth labels
                - condition: Name of experimental condition
        output_path: Path to save the figure (e.g., 'figures/dist.png').
    """
    raise NotImplementedError("TODO")


def plot_condition_comparison(comparison: Dict, output_path: str) -> None:
    """
    Plot comparison between experimental conditions.

    Creates side-by-side violin plots or similar visualization showing
    activation distributions for each condition.

    Args:
        comparison: Dict from compare_probe_activations containing
                   results from multiple conditions.
        output_path: Path to save the figure.
    """
    raise NotImplementedError("TODO")


def plot_activation_over_generation(
    activations_by_step: Dict,
    output_path: str
) -> None:
    """
    Plot how probe activation evolves during token generation.

    Shows activation trajectory over the course of chain-of-thought
    generation, useful for seeing when/if model suppresses concept.

    Args:
        activations_by_step: Dict mapping generation step to activation value.
        output_path: Path to save the figure.
    """
    raise NotImplementedError("TODO")


def plot_layer_comparison(
    activations_by_layer: Dict,
    output_path: str
) -> None:
    """
    Plot probe activations across different model layers.

    Shows how detectable the concept is at different depths in the model.

    Args:
        activations_by_layer: Dict mapping layer index to activation stats.
        output_path: Path to save the figure.
    """
    raise NotImplementedError("TODO")


def create_summary_figure(
    all_results: Dict,
    output_path: str
) -> None:
    """
    Create a multi-panel summary figure for a complete experiment.

    Combines multiple visualizations into a single publication-ready figure.

    Args:
        all_results: Dict containing all experimental results.
        output_path: Path to save the figure.
    """
    raise NotImplementedError("TODO")
