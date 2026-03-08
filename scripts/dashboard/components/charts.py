"""Plotly chart builders: heatmaps, bars, lines, box plots, ROC curves."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


# Shared color scale: green (low) -> yellow (0.5) -> red (high)
CONFIDENCE_COLORSCALE = [
    [0.0, "#1a9641"],
    [0.25, "#a6d96a"],
    [0.5, "#ffffbf"],
    [0.75, "#fdae61"],
    [1.0, "#d7191c"],
]

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=60, r=20, t=40, b=40),
)


def probe_heatmap(matrix, x_labels, y_labels, title="Probe Confidence",
                  zmin=0.0, zmax=1.0):
    """
    Plotly heatmap with confidence color scale.

    Args:
        matrix: 2D array-like, shape (len(y_labels), len(x_labels))
        x_labels: Column labels
        y_labels: Row labels
        title: Chart title
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(x) for x in x_labels],
        y=[str(y) for y in y_labels],
        colorscale=CONFIDENCE_COLORSCALE,
        zmin=zmin,
        zmax=zmax,
        text=np.round(np.array(matrix, dtype=float), 3),
        texttemplate="%{text:.3f}",
        textfont=dict(size=10),
        hovertemplate="x: %{x}<br>y: %{y}<br>confidence: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, **DARK_LAYOUT)
    return fig


def regime_comparison_bars(summaries, positions, regimes, layers, title="Mean Confidence by Regime"):
    """
    Grouped bar chart: one group per regime, one bar per position.

    Args:
        summaries: dict {regime: {position: {layer_X: {mean_confidence: ...}}}}
        positions: list of position names
        regimes: list of regime names
        layers: list of layer indices (for averaging)
    """
    fig = go.Figure()

    for pos in positions:
        means = []
        for regime in regimes:
            regime_data = summaries.get(regime, {}).get(pos, {})
            vals = [regime_data.get(f"layer_{l}", {}).get("mean_confidence", 0)
                    for l in layers]
            means.append(np.mean(vals) if vals else 0)
        fig.add_trace(go.Bar(name=pos, x=regimes, y=means))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Regime",
        yaxis_title="Mean Confidence",
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    return fig


def layer_line_chart(summaries, position, layers, regimes, title="Confidence by Layer"):
    """
    Line chart: confidence vs layer for each regime.

    Args:
        summaries: dict {regime: {position: {layer_X: {mean_confidence: ...}}}}
        position: position name to plot
        layers: list of layer indices (x-axis)
        regimes: list of regime names
    """
    fig = go.Figure()
    for regime in regimes:
        pos_data = summaries.get(regime, {}).get(position, {})
        y_vals = [pos_data.get(f"layer_{l}", {}).get("mean_confidence", None)
                  for l in layers]
        fig.add_trace(go.Scatter(
            x=[str(l) for l in layers],
            y=y_vals,
            mode="lines+markers",
            name=regime,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Mean Confidence",
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    return fig


def confidence_box_plots(trials_by_regime, position, layers, title="Confidence Distribution"):
    """
    Box plots of confidence distribution by regime.

    Args:
        trials_by_regime: dict {regime: [trial_dicts]}
        position: position name
        layers: list of layer indices (averaged)
    """
    fig = go.Figure()

    for regime, trials in trials_by_regime.items():
        values = []
        for trial in trials:
            pr = trial.get("probe_results", {})
            # For feedback regimes, use last turn
            if "turns" in trial and trial["turns"]:
                pr = trial["turns"][-1].get("probe_results", {})
            pos_data = pr.get(position, {})
            layer_confs = [pos_data.get(l, pos_data.get(str(l), {})).get("mean_confidence", None)
                          for l in layers]
            layer_confs = [c for c in layer_confs if c is not None]
            if layer_confs:
                values.append(np.mean(layer_confs))

        if values:
            fig.add_trace(go.Box(y=values, name=regime))

    fig.update_layout(
        title=title,
        yaxis_title="Mean Confidence",
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    return fig


def detection_rate_bars(summaries, operating_points, regimes, trials_by_regime,
                        title="Weighted Combo Detection Rate"):
    """Bar chart of weighted combination detection rates by operating point."""
    from src.probes.evaluate import score_weighted_combination

    fig = go.Figure()

    for op_name in operating_points:
        rates = []
        for regime in regimes:
            trials = trials_by_regime.get(regime, [])
            if not trials:
                rates.append(0)
                continue
            detected = 0
            for trial in trials:
                ws = trial.get("weighted_score", {})
                if ws and ws.get("detections", {}).get(op_name):
                    detected += 1
            rates.append(detected / len(trials) if trials else 0)
        fig.add_trace(go.Bar(name=op_name, x=regimes, y=rates))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Regime",
        yaxis_title="Detection Rate",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        **DARK_LAYOUT,
    )
    return fig


def token_stacked_bar(trials_by_regime, title="Token Counts by Regime"):
    """Stacked bar: thinking vs answer tokens by regime."""
    regimes = []
    think_means = []
    answer_means = []

    for regime, trials in trials_by_regime.items():
        think_vals = []
        answer_vals = []
        for trial in trials:
            if "turns" in trial and trial["turns"]:
                t = trial["turns"][-1]
            else:
                t = trial
            think_vals.append(t.get("thinking_tokens", 0))
            answer_vals.append(t.get("answer_tokens", 0))
        regimes.append(regime)
        think_means.append(np.mean(think_vals) if think_vals else 0)
        answer_means.append(np.mean(answer_vals) if answer_vals else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Thinking", x=regimes, y=think_means))
    fig.add_trace(go.Bar(name="Answer", x=regimes, y=answer_means))
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Regime",
        yaxis_title="Mean Tokens",
        **DARK_LAYOUT,
    )
    return fig


# ─── Discrimination analysis charts ──────────────────────────────────


def tp_tn_distribution(pos_confs, neg_confs, title="TP/TN Score Distribution"):
    """Overlapping histograms of positive vs negative confidence distributions.

    Args:
        pos_confs: list of float, positive (tree) trial confidences
        neg_confs: list of float, negative (non-tree) trial confidences
        title: chart title
    """
    fig = go.Figure()
    if pos_confs:
        fig.add_trace(go.Histogram(
            x=pos_confs, name="Positive (tree)",
            opacity=0.6, nbinsx=30,
            marker_color="#d7191c",
        ))
    if neg_confs:
        fig.add_trace(go.Histogram(
            x=neg_confs, name="Negative (non-tree)",
            opacity=0.6, nbinsx=30,
            marker_color="#1a9641",
        ))
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title="Probe Confidence",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        **DARK_LAYOUT,
    )
    return fig


def roc_curve_chart(curves, title="ROC Curve"):
    """ROC curve with AUC annotation.

    Args:
        curves: list of dicts with keys "fpr", "tpr", "auc", "label"
        title: chart title
    """
    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        showlegend=False,
    ))

    for curve in curves:
        label = f"{curve['label']} (AUC={curve['auc']:.3f})"
        fig.add_trace(go.Scatter(
            x=curve["fpr"], y=curve["tpr"],
            mode="lines", name=label,
            line=dict(width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
        **DARK_LAYOUT,
    )
    return fig


def separation_heatmap(matrix, x_labels, y_labels, title="Separation",
                       zmin=0.0, zmax=1.0, colorbar_title="AUC"):
    """Diverging colorscale heatmap for d-prime/AUC values.

    Blue = poor separation, white = 0.5, red = good separation.

    Args:
        matrix: 2D array-like
        x_labels: column labels
        y_labels: row labels
        title: chart title
        zmin, zmax: color scale range
        colorbar_title: label for the colorbar
    """
    # Diverging: blue (poor) -> white (0.5) -> red (good)
    diverging_scale = [
        [0.0, "#2166ac"],
        [0.25, "#67a9cf"],
        [0.5, "#f7f7f7"],
        [0.75, "#ef8a62"],
        [1.0, "#b2182b"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(x) for x in x_labels],
        y=[str(y) for y in y_labels],
        colorscale=diverging_scale,
        zmin=zmin,
        zmax=zmax,
        text=np.round(np.array(matrix, dtype=float), 3),
        texttemplate="%{text:.3f}",
        textfont=dict(size=10),
        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
        colorbar=dict(title=colorbar_title),
    ))
    fig.update_layout(title=title, **DARK_LAYOUT)
    return fig


def per_question_strip_chart(data_points, title="Per-Question Probe Scores"):
    """Strip/jitter plot showing individual trial scores per question, colored by category.

    Args:
        data_points: list of dicts with keys "question_id", "confidence", "category", "regime"
        title: chart title
    """
    if not data_points:
        fig = go.Figure()
        fig.update_layout(title=title, **DARK_LAYOUT)
        return fig

    # Sort by category then question_id so positives and negatives are grouped
    categories = sorted(set(d["category"] for d in data_points))
    cat_colors = {"positive": "#d7191c", "negative": "#1a9641"}

    fig = go.Figure()
    for cat in categories:
        pts = [d for d in data_points if d["category"] == cat]
        fig.add_trace(go.Box(
            x=[d["question_id"] for d in pts],
            y=[d["confidence"] for d in pts],
            name=cat.capitalize(),
            marker_color=cat_colors.get(cat, "#888"),
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            line=dict(width=1),
            marker=dict(size=4, opacity=0.7),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Question ID",
        yaxis_title="Probe Confidence",
        yaxis=dict(range=[0, 1]),
        boxmode="group",
        **DARK_LAYOUT,
    )
    return fig


def tp_tn_box_by_regime(regime_data, title="TP vs TN Distribution by Regime"):
    """Side-by-side box plots: positive vs negative per regime.

    Args:
        regime_data: list of dicts with keys "regime", "category", "confidence"
        title: chart title
    """
    if not regime_data:
        fig = go.Figure()
        fig.update_layout(title=title, **DARK_LAYOUT)
        return fig

    cat_colors = {"positive": "#d7191c", "negative": "#1a9641"}

    fig = go.Figure()
    for cat in ["positive", "negative"]:
        pts = [d for d in regime_data if d["category"] == cat]
        if pts:
            fig.add_trace(go.Box(
                x=[d["regime"] for d in pts],
                y=[d["confidence"] for d in pts],
                name=f"{'Tree' if cat == 'positive' else 'Non-tree'}",
                marker_color=cat_colors[cat],
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                marker=dict(size=4, opacity=0.6),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Regime",
        yaxis_title="Probe Confidence",
        yaxis=dict(range=[0, 1]),
        boxmode="group",
        **DARK_LAYOUT,
    )
    return fig
