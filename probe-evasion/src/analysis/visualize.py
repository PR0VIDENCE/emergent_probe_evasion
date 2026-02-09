"""Visualization functions for probe activation analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set consistent style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "baseline": "#4C72B0",
    "informed": "#DD8452",
    "informed_no_monitor": "#55A868",
    "evasion_awareness": "#C44E52",
    "evasion_technical": "#8172B3",
}


def plot_layer_confidence_by_regime(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Probe Confidence by Layer and Regime",
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot mean probe confidence per layer, grouped by regime.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        title: Plot title
        figsize: Figure dimensions
    """
    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    layers = sorted([int(c.split("_")[1]) for c in conf_cols])

    fig, ax = plt.subplots(figsize=figsize)

    for regime in df["regime"].unique():
        regime_df = df[df["regime"] == regime]
        means = [regime_df[f"layer_{l}_mean_conf"].mean() for l in layers]
        stds = [regime_df[f"layer_{l}_mean_conf"].std() for l in layers]

        color = COLORS.get(regime, None)
        ax.errorbar(layers, means, yerr=stds, marker="o", label=regime,
                    capsize=4, color=color)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Probe Confidence", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Detection threshold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_confidence_distributions(
    df: pd.DataFrame,
    output_path: str,
    layers: Optional[List[int]] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot violin/box plots of confidence distributions per layer and regime.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        layers: Specific layers to plot (None = all)
        figsize: Figure dimensions
    """
    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    all_layers = sorted([int(c.split("_")[1]) for c in conf_cols])
    layers = layers or all_layers

    # Reshape data for seaborn
    plot_data = []
    for _, row in df.iterrows():
        for layer in layers:
            plot_data.append({
                "regime": row["regime"],
                "layer": layer,
                "confidence": row[f"layer_{layer}_mean_conf"],
            })
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=figsize)

    palette = [COLORS.get(r, None) for r in plot_df["regime"].unique()]
    sns.violinplot(data=plot_df, x="layer", y="confidence", hue="regime",
                   ax=ax, palette=palette, inner="box", cut=0)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Confidence", fontsize=12)
    ax.set_title("Probe Confidence Distribution by Layer", fontsize=14)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_detection_rate_heatmap(
    df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot heatmap of detection rates (majority_vote == 1) by question and layer.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        figsize: Figure dimensions
    """
    maj_cols = [c for c in df.columns if c.endswith("_majority")]
    layers = sorted([int(c.split("_")[1]) for c in maj_cols])

    # Aggregate detection rate by question
    grouped = df.groupby("question_id")[[f"layer_{l}_majority" for l in layers]].mean()
    grouped.columns = [f"Layer {l}" for l in layers]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(grouped, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Detection Rate"})

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Question ID", fontsize=12)
    ax.set_title("Tree Concept Detection Rate by Question and Layer", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_regime_comparison_bars(
    df: pd.DataFrame,
    output_path: str,
    layer: int,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot bar chart comparing regimes at a specific layer.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        layer: Layer to compare
        figsize: Figure dimensions
    """
    col = f"layer_{layer}_mean_conf"

    fig, ax = plt.subplots(figsize=figsize)

    regime_stats = df.groupby("regime")[col].agg(["mean", "std"]).reset_index()

    colors = [COLORS.get(r, "#999999") for r in regime_stats["regime"]]
    bars = ax.bar(regime_stats["regime"], regime_stats["mean"],
                  yerr=regime_stats["std"], capsize=5, color=colors, alpha=0.8)

    ax.set_ylabel("Mean Probe Confidence", fontsize=12)
    ax.set_xlabel("Regime", fontsize=12)
    ax.set_title(f"Probe Confidence at Layer {layer} by Regime", fontsize=14)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, (_, row) in zip(bars, regime_stats.iterrows()):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_rollout_variance(
    df: pd.DataFrame,
    output_path: str,
    layer: int,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot confidence variance across rollouts for each question.

    Helps identify questions with high/low consistency.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        layer: Layer to analyze
        figsize: Figure dimensions
    """
    col = f"layer_{layer}_mean_conf"

    fig, ax = plt.subplots(figsize=figsize)

    # Get all rollouts for each question
    questions = df["question_id"].unique()

    for i, qid in enumerate(questions):
        q_data = df[df["question_id"] == qid]
        rollouts = q_data["rollout"].values
        confs = q_data[col].values

        ax.scatter([i] * len(rollouts), confs, alpha=0.6, s=50)
        ax.errorbar(i, confs.mean(), yerr=confs.std(), fmt="D", color="black",
                    capsize=5, markersize=8)

    ax.set_xticks(range(len(questions)))
    ax.set_xticklabels(questions, rotation=45)
    ax.set_xlabel("Question ID", fontsize=12)
    ax.set_ylabel(f"Probe Confidence (Layer {layer})", fontsize=12)
    ax.set_title(f"Rollout Variance by Question (Layer {layer})", fontsize=14)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ensemble_agreement(
    df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot distribution of ensemble agreement ratios across layers.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        figsize: Figure dimensions
    """
    agree_cols = [c for c in df.columns if c.endswith("_agreement")]
    layers = sorted([int(c.split("_")[1]) for c in agree_cols])

    fig, ax = plt.subplots(figsize=figsize)

    data = []
    for layer in layers:
        col = f"layer_{layer}_agreement"
        for val in df[col]:
            data.append({"layer": layer, "agreement": val})

    plot_df = pd.DataFrame(data)
    sns.boxplot(data=plot_df, x="layer", y="agreement", ax=ax)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Ensemble Agreement Ratio", fontsize=12)
    ax.set_title("Probe Ensemble Agreement by Layer", fontsize=14)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random agreement")
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_question_ranking(
    df: pd.DataFrame,
    output_path: str,
    layer: int,
    figsize: tuple = (12, 8)
) -> None:
    """
    Rank questions by probe activation and show their full text.

    Args:
        df: DataFrame from results_to_dataframe()
        output_path: Path to save figure
        layer: Layer to rank by
        figsize: Figure dimensions
    """
    col = f"layer_{layer}_mean_conf"

    # Aggregate by question
    q_stats = df.groupby(["question_id", "question"]).agg({
        col: ["mean", "std"]
    }).reset_index()
    q_stats.columns = ["question_id", "question", "mean_conf", "std_conf"]
    q_stats = q_stats.sort_values("mean_conf", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = range(len(q_stats))
    bars = ax.barh(y_pos, q_stats["mean_conf"], xerr=q_stats["std_conf"],
                   capsize=3, alpha=0.8)

    # Truncate question text for labels
    labels = [q[:50] + "..." if len(q) > 50 else q for q in q_stats["question"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel(f"Mean Probe Confidence (Layer {layer})", fontsize=12)
    ax.set_title(f"Questions Ranked by Probe Activation (Layer {layer})", fontsize=14)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_dashboard(
    df: pd.DataFrame,
    output_dir: str,
    layers: Optional[List[int]] = None
) -> None:
    """
    Generate a complete set of analysis visualizations.

    Args:
        df: DataFrame from results_to_dataframe()
        output_dir: Directory to save all figures
        layers: Specific layers to focus on (None = all)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conf_cols = [c for c in df.columns if c.endswith("_mean_conf")]
    all_layers = sorted([int(c.split("_")[1]) for c in conf_cols])
    layers = layers or all_layers

    print(f"\nGenerating visualizations in: {out_path}")

    # 1. Layer confidence by regime
    plot_layer_confidence_by_regime(
        df, str(out_path / "layer_confidence_by_regime.png")
    )

    # 2. Confidence distributions
    plot_confidence_distributions(
        df, str(out_path / "confidence_distributions.png"), layers=layers
    )

    # 3. Detection rate heatmap
    plot_detection_rate_heatmap(
        df, str(out_path / "detection_rate_heatmap.png")
    )

    # 4. Ensemble agreement
    plot_ensemble_agreement(
        df, str(out_path / "ensemble_agreement.png")
    )

    # 5. Per-layer comparisons (for key layers)
    for layer in layers:
        plot_regime_comparison_bars(
            df, str(out_path / f"regime_comparison_layer_{layer}.png"), layer=layer
        )
        plot_rollout_variance(
            df, str(out_path / f"rollout_variance_layer_{layer}.png"), layer=layer
        )

    # 6. Question ranking (use middle and late layer)
    if layers:
        mid_layer = layers[len(layers) // 2]
        late_layer = layers[-1]
        plot_question_ranking(
            df, str(out_path / f"question_ranking_layer_{mid_layer}.png"), layer=mid_layer
        )
        plot_question_ranking(
            df, str(out_path / f"question_ranking_layer_{late_layer}.png"), layer=late_layer
        )

    print(f"\nGenerated {6 + 2*len(layers)} figures in {out_path}")


# --- Legacy stubs for compatibility ---

def plot_activation_distributions(results: Dict, output_path: str) -> None:
    """Legacy stub - use plot_confidence_distributions instead."""
    raise NotImplementedError("Use plot_confidence_distributions with DataFrame input")


def plot_condition_comparison(comparison: Dict, output_path: str) -> None:
    """Legacy stub - use plot_regime_comparison_bars instead."""
    raise NotImplementedError("Use plot_regime_comparison_bars with DataFrame input")


def plot_activation_over_generation(activations_by_step: Dict, output_path: str) -> None:
    """Plot probe activation trajectory during generation."""
    raise NotImplementedError("TODO - requires generation-time activation extraction")


def plot_layer_comparison(activations_by_layer: Dict, output_path: str) -> None:
    """Legacy stub - use plot_layer_confidence_by_regime instead."""
    raise NotImplementedError("Use plot_layer_confidence_by_regime with DataFrame input")


def create_summary_figure(all_results: Dict, output_path: str) -> None:
    """Legacy stub - use create_summary_dashboard instead."""
    raise NotImplementedError("Use create_summary_dashboard with DataFrame input")
