"""Statistical comparison of probe activations across conditions."""

from typing import Dict


def compare_probe_activations(
    condition_a_results: Dict,
    condition_b_results: Dict
) -> Dict:
    """
    Compare probe activation patterns between two experimental conditions.

    Performs statistical tests to determine if there are significant
    differences in probe activations between conditions (e.g., baseline
    vs informed).

    Args:
        condition_a_results: Dict containing:
                            - activations: List of probe activation values
                            - labels: Ground truth concept presence
                            - confidences: Probe confidence scores
                            - metadata: Condition info, questions used, etc.
        condition_b_results: Same structure as condition_a_results.

    Returns:
        Dict containing:
        - mean_diff: Difference in mean activation between conditions
        - effect_size: Cohen's d effect size
        - t_statistic: t-test statistic
        - p_value: Two-tailed p-value from t-test
        - ci_95: 95% confidence interval for difference
        - mann_whitney_u: Non-parametric test statistic
        - mann_whitney_p: P-value for Mann-Whitney U test
        - summary: Human-readable summary string
    """
    raise NotImplementedError("TODO")


def compare_across_question_types(results_by_condition: Dict) -> Dict:
    """
    Compare probe activations across question categories.

    Analyzes whether evasion behavior differs for direct vs circumspect
    vs oblique questions.

    Args:
        results_by_condition: Dict mapping condition name to results dict,
                             where results contain per-question-category data.

    Returns:
        Dict with per-category comparisons and interaction effects.
    """
    raise NotImplementedError("TODO")


def compute_evasion_score(baseline_results: Dict, informed_results: Dict) -> float:
    """
    Compute a scalar evasion score summarizing the degree of probe evasion.

    Args:
        baseline_results: Results from baseline condition.
        informed_results: Results from informed condition.

    Returns:
        Float between 0 and 1 where:
        - 0 = no difference from baseline (no evasion)
        - 1 = complete suppression of probe activation (full evasion)
    """
    raise NotImplementedError("TODO")
