"""Plotly chart builders: heatmaps, bars, lines, box plots."""

import plotly.graph_objects as go
import plotly.express as px
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
