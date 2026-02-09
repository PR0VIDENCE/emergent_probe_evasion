"""Analysis and visualization modules."""

from .aggregate_results import (
    load_results,
    results_to_dataframe,
    aggregate_by_question,
    aggregate_by_regime,
    compute_layer_summary,
    compute_question_difficulty,
    compare_regimes,
    find_high_variance_samples,
    analyze_experiment,
    print_summary,
)

from .visualize import (
    plot_layer_confidence_by_regime,
    plot_confidence_distributions,
    plot_detection_rate_heatmap,
    plot_regime_comparison_bars,
    plot_rollout_variance,
    plot_ensemble_agreement,
    plot_question_ranking,
    create_summary_dashboard,
    COLORS,
)

from .compare_conditions import compare_probe_activations

__all__ = [
    # Aggregation
    "load_results",
    "results_to_dataframe",
    "aggregate_by_question",
    "aggregate_by_regime",
    "compute_layer_summary",
    "compute_question_difficulty",
    "compare_regimes",
    "find_high_variance_samples",
    "analyze_experiment",
    "print_summary",
    # Visualization
    "plot_layer_confidence_by_regime",
    "plot_confidence_distributions",
    "plot_detection_rate_heatmap",
    "plot_regime_comparison_bars",
    "plot_rollout_variance",
    "plot_ensemble_agreement",
    "plot_question_ranking",
    "create_summary_dashboard",
    "COLORS",
    # Comparison
    "compare_probe_activations",
]
