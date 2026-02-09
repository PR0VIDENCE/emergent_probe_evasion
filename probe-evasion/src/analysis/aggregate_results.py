"""Aggregate and analyze experimental results across rollouts and questions."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_results(path: str) -> List[Dict]:
    """Load experimental results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def extract_probe_stats(probe_results: Dict) -> pd.DataFrame:
    """
    Extract probe statistics into a DataFrame.

    Args:
        probe_results: Dict mapping layer -> probe metrics

    Returns:
        DataFrame with columns: layer, mean_confidence, majority_vote,
        agreement_ratio, individual_confidences
    """
    rows = []
    for layer, metrics in probe_results.items():
        rows.append({
            "layer": int(layer),
            "mean_confidence": metrics["mean_confidence"],
            "majority_vote": metrics["majority_vote"],
            "agreement_ratio": metrics["agreement_ratio"],
            "individual_confidences": metrics["individual_confidences"],
            "std_confidence": np.std(metrics["individual_confidences"]),
            "min_confidence": min(metrics["individual_confidences"]),
            "max_confidence": max(metrics["individual_confidences"]),
        })
    return pd.DataFrame(rows).sort_values("layer")


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Convert list of result dicts to a flat DataFrame.

    Each row is one rollout, with probe results expanded into columns.
    """
    rows = []
    for r in results:
        row = {
            "regime": r["regime"],
            "question_id": r["question_id"],
            "question": r["question"],
            "rollout": r["rollout"],
            "elapsed_seconds": r["elapsed_seconds"],
            "thinking_tokens": r.get("thinking_tokens", 0),
            "answer_tokens": r.get("answer_tokens", 0),
        }

        # Flatten probe results per layer
        for layer, metrics in r["probe_results"].items():
            layer_int = int(layer)
            row[f"layer_{layer_int}_mean_conf"] = metrics["mean_confidence"]
            row[f"layer_{layer_int}_majority"] = metrics["majority_vote"]
            row[f"layer_{layer_int}_agreement"] = metrics["agreement_ratio"]
            row[f"layer_{layer_int}_std_conf"] = np.std(metrics["individual_confidences"])

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by question (across rollouts).

    Returns mean and std of probe metrics for each question.
    """
    # Find all layer columns
    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    majority_cols = [c for c in df.columns if c.endswith("_majority")]

    agg_dict = {
        "rollout": "count",
        "elapsed_seconds": ["mean", "std"],
        "answer_tokens": ["mean", "std"],
    }

    for col in conf_cols:
        agg_dict[col] = ["mean", "std"]
    for col in majority_cols:
        agg_dict[col] = ["mean", "sum"]  # mean = detection rate, sum = count

    grouped = df.groupby(["regime", "question_id"]).agg(agg_dict)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]

    return grouped.reset_index()


def aggregate_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by regime (across all questions and rollouts).

    Returns overall statistics per experimental condition.
    """
    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    majority_cols = [c for c in df.columns if c.endswith("_majority")]

    agg_dict = {
        "question_id": "nunique",
        "rollout": "count",
        "elapsed_seconds": ["mean", "std"],
        "answer_tokens": ["mean", "std"],
    }

    for col in conf_cols:
        agg_dict[col] = ["mean", "std"]
    for col in majority_cols:
        agg_dict[col] = "mean"  # detection rate

    grouped = df.groupby("regime").agg(agg_dict)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]

    # Rename for clarity
    grouped = grouped.rename(columns={
        "question_id_nunique": "n_questions",
        "rollout_count": "n_rollouts",
    })

    return grouped.reset_index()


def compute_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per layer across all samples.

    Returns DataFrame with layer-level aggregates.
    """
    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    layers = sorted([int(c.split("_")[1]) for c in conf_cols])

    rows = []
    for regime in df["regime"].unique():
        regime_df = df[df["regime"] == regime]

        for layer in layers:
            conf_col = f"layer_{layer}_mean_conf"
            maj_col = f"layer_{layer}_majority"
            std_col = f"layer_{layer}_std_conf"

            rows.append({
                "regime": regime,
                "layer": layer,
                "mean_confidence": regime_df[conf_col].mean(),
                "std_confidence": regime_df[conf_col].std(),
                "detection_rate": regime_df[maj_col].mean(),
                "mean_ensemble_std": regime_df[std_col].mean() if std_col in regime_df.columns else None,
                "n_samples": len(regime_df),
            })

    return pd.DataFrame(rows)


def compute_question_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate question 'difficulty' based on probe activation patterns.

    Higher activation = easier to detect tree concept.
    """
    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]

    # Average confidence across all layers
    df = df.copy()
    df["avg_confidence_all_layers"] = df[conf_cols].mean(axis=1)

    question_stats = df.groupby(["regime", "question_id", "question"]).agg({
        "avg_confidence_all_layers": ["mean", "std"],
        "rollout": "count",
    }).reset_index()

    question_stats.columns = ["regime", "question_id", "question",
                              "mean_conf", "std_conf", "n_rollouts"]

    return question_stats.sort_values("mean_conf", ascending=False)


def print_summary(df: pd.DataFrame, title: str = "Results Summary"):
    """Print a formatted summary of results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

    # Overall stats
    print(f"Total samples: {len(df)}")
    print(f"Regimes: {df['regime'].unique().tolist()}")
    print(f"Questions: {df['question_id'].nunique()}")
    print(f"Rollouts per question: {df.groupby('question_id')['rollout'].count().mean():.1f}")

    # Layer summary
    layer_summary = compute_layer_summary(df)
    print(f"\n--- Layer Summary by Regime ---\n")

    for regime in layer_summary["regime"].unique():
        regime_data = layer_summary[layer_summary["regime"] == regime]
        print(f"\nRegime: {regime}")
        print(f"{'Layer':<8} {'Mean Conf':<12} {'Std':<10} {'Detection Rate':<15}")
        print("-" * 45)
        for _, row in regime_data.iterrows():
            print(f"{row['layer']:<8} {row['mean_confidence']:<12.3f} "
                  f"{row['std_confidence']:<10.3f} {row['detection_rate']:<15.2%}")


def compare_regimes(
    df: pd.DataFrame,
    regime_a: str = "baseline",
    regime_b: str = "informed"
) -> pd.DataFrame:
    """
    Compare probe activations between two regimes.

    Returns per-layer comparison with effect sizes.
    """
    df_a = df[df["regime"] == regime_a]
    df_b = df[df["regime"] == regime_b]

    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    layers = sorted([int(c.split("_")[1]) for c in conf_cols])

    comparisons = []
    for layer in layers:
        col = f"layer_{layer}_mean_conf"

        a_vals = df_a[col].values
        b_vals = df_b[col].values

        # Basic stats
        mean_a, mean_b = a_vals.mean(), b_vals.mean()
        std_a, std_b = a_vals.std(), b_vals.std()

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

        # Simple t-test (scipy import only if available)
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(a_vals, b_vals)
        except ImportError:
            t_stat, p_value = None, None

        comparisons.append({
            "layer": layer,
            f"mean_{regime_a}": mean_a,
            f"mean_{regime_b}": mean_b,
            "difference": mean_a - mean_b,
            "cohens_d": cohens_d,
            "t_statistic": t_stat,
            "p_value": p_value,
        })

    return pd.DataFrame(comparisons)


def find_high_variance_samples(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Find samples where probes within ensemble disagree significantly.

    These might be interesting edge cases.
    """
    std_cols = [c for c in df.columns if c.endswith("_std_conf")]

    if not std_cols:
        print("No std columns found - need individual probe confidences")
        return pd.DataFrame()

    # Find rows where any layer has high within-ensemble variance
    mask = df[std_cols].max(axis=1) > threshold
    high_var = df[mask].copy()

    return high_var[["regime", "question_id", "rollout"] + std_cols]


# --- Main analysis functions ---

def analyze_experiment(results_path: str, output_dir: Optional[str] = None):
    """
    Run full analysis pipeline on experimental results.

    Args:
        results_path: Path to JSON results file
        output_dir: Optional directory to save analysis outputs
    """
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    df = results_to_dataframe(results)
    print(f"Loaded {len(df)} samples")

    # Print summary
    print_summary(df)

    # Aggregate by question
    by_question = aggregate_by_question(df)
    print("\n--- Per-Question Aggregates ---")
    print(by_question.to_string())

    # Aggregate by regime
    by_regime = aggregate_by_regime(df)
    print("\n--- Per-Regime Aggregates ---")
    print(by_regime.to_string())

    # Question difficulty
    difficulty = compute_question_difficulty(df)
    print("\n--- Question Difficulty (by mean probe confidence) ---")
    print(difficulty.to_string())

    # Save outputs if requested
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_path / "full_results.csv", index=False)
        by_question.to_csv(out_path / "by_question.csv", index=False)
        by_regime.to_csv(out_path / "by_regime.csv", index=False)
        compute_layer_summary(df).to_csv(out_path / "layer_summary.csv", index=False)
        difficulty.to_csv(out_path / "question_difficulty.csv", index=False)

        print(f"\nSaved analysis outputs to: {out_path}")

    return df, by_question, by_regime


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze probe evasion experiment results")
    parser.add_argument("results_path", help="Path to JSON results file")
    parser.add_argument("--output-dir", "-o", help="Directory to save analysis outputs")
    parser.add_argument("--compare", nargs=2, metavar=("REGIME_A", "REGIME_B"),
                        help="Compare two regimes (e.g., --compare baseline informed)")

    args = parser.parse_args()

    df, by_question, by_regime = analyze_experiment(args.results_path, args.output_dir)

    if args.compare:
        regime_a, regime_b = args.compare
        if regime_a in df["regime"].values and regime_b in df["regime"].values:
            comparison = compare_regimes(df, regime_a, regime_b)
            print(f"\n--- Comparison: {regime_a} vs {regime_b} ---")
            print(comparison.to_string())
        else:
            print(f"Warning: One or both regimes not found in data")
