"""Tab 5: Results Browser — load and visualize existing experiment results."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from components.data_loader import (
    discover_regimes, load_all_trials, get_trial_probe_results,
    get_unique_questions, get_unique_rollouts, load_summary,
)
from components.charts import (
    probe_heatmap, regime_comparison_bars, layer_line_chart,
    confidence_box_plots, token_stacked_bar, detection_rate_bars,
)
from components.probe_display import render_probe_scores, confidence_color

from state import get_target_layers


def render():
    st.header("Results Browser")

    # ─── Data loading ────────────────────────────────────────────────
    data_dir = st.text_input(
        "Results directory",
        value="data/outputs/evasion",
        key="results_browser_dir",
    )

    # Resolve path
    from state import resolve_path
    abs_dir = resolve_path(data_dir)

    available = discover_regimes(abs_dir)
    if not available:
        st.warning(f"No regime JSON files found in {abs_dir}/trials/")
        return

    selected_regimes = st.multiselect(
        "Regimes to load", available, default=available[:5],
        key="results_browser_regimes",
    )
    if not selected_regimes:
        st.info("Select at least one regime.")
        return

    trials_by_regime = load_all_trials(abs_dir, selected_regimes)
    if not trials_by_regime:
        st.warning("No trial data loaded.")
        return

    st.success(f"Loaded {sum(len(v) for v in trials_by_regime.values())} trials "
               f"across {len(trials_by_regime)} regimes")

    # Determine layers from data
    target_layers = get_target_layers()
    if not target_layers:
        # Infer from first trial
        first_trials = next(iter(trials_by_regime.values()))
        pr = get_trial_probe_results(first_trials[0])
        first_pos = next(iter(pr.values()), {})
        target_layers = sorted(int(k) for k in first_pos.keys()
                               if str(k).isdigit())

    # Determine positions from data
    first_trials = next(iter(trials_by_regime.values()))
    pr = get_trial_probe_results(first_trials[0])
    all_positions = sorted(pr.keys())

    # Aggregate summary via existing function
    all_trials_flat = []
    for trials in trials_by_regime.values():
        all_trials_flat.extend(trials)

    from run_evasion_experiment import aggregate_results
    summary = aggregate_results(all_trials_flat, target_layers)
    regimes = list(trials_by_regime.keys())

    # ─── Position selector ───────────────────────────────────────────
    position = st.selectbox(
        "Position", all_positions,
        index=all_positions.index("answer_mean_pool") if "answer_mean_pool" in all_positions else 0,
        key="results_browser_position",
    )

    # ─── Heatmaps ────────────────────────────────────────────────────
    st.subheader("Regime x Layer Heatmap")

    # Build matrix: rows=regimes, cols=layers
    matrix = []
    for regime in regimes:
        row = []
        for layer_idx in target_layers:
            val = summary.get(regime, {}).get(position, {}).get(
                f"layer_{layer_idx}", {}
            ).get("mean_confidence", 0)
            row.append(val)
        matrix.append(row)

    fig = probe_heatmap(
        matrix,
        x_labels=[f"L{l}" for l in target_layers],
        y_labels=regimes,
        title=f"Mean Confidence — {position}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Regime x question heatmap ───────────────────────────────────
    st.subheader("Regime x Question Heatmap")
    # Average across displayed layers for the selected position
    layer_subset = st.multiselect(
        "Layers to average",
        target_layers,
        default=target_layers[:5] if len(target_layers) > 5 else target_layers,
        key="results_browser_layer_subset",
    )

    # Collect per-question means
    all_questions = []
    for trials in trials_by_regime.values():
        for qid in get_unique_questions(trials):
            if qid not in all_questions:
                all_questions.append(qid)

    q_matrix = []
    for regime in regimes:
        row = []
        trials = trials_by_regime.get(regime, [])
        by_q = {}
        for trial in trials:
            qid = trial["question_id"]
            pr = get_trial_probe_results(trial)
            pos_data = pr.get(position, {})
            confs = [pos_data.get(l, pos_data.get(str(l), {})).get("mean_confidence", None)
                     for l in layer_subset]
            confs = [c for c in confs if c is not None]
            if confs:
                by_q.setdefault(qid, []).append(np.mean(confs))
        for qid in all_questions:
            vals = by_q.get(qid, [])
            row.append(np.mean(vals) if vals else 0)
        q_matrix.append(row)

    fig_q = probe_heatmap(
        q_matrix,
        x_labels=all_questions,
        y_labels=regimes,
        title=f"Mean Confidence by Question — {position}",
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # ─── Bar + line charts ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mean Confidence by Regime")
        fig_bar = regime_comparison_bars(summary, all_positions, regimes, target_layers)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Confidence by Layer")
        fig_line = layer_line_chart(summary, position, target_layers, regimes)
        st.plotly_chart(fig_line, use_container_width=True)

    # ─── Box plots ───────────────────────────────────────────────────
    st.subheader("Confidence Distribution")
    fig_box = confidence_box_plots(trials_by_regime, position, target_layers)
    st.plotly_chart(fig_box, use_container_width=True)

    # ─── Weighted combo detection rates ──────────────────────────────
    combo = st.session_state.get("weighted_combo")
    if combo is not None:
        st.subheader("Weighted Combination Detection Rates")
        # Score trials that don't have scores yet
        from src.probes.evaluate import score_weighted_combination
        for trials in trials_by_regime.values():
            for trial in trials:
                if "weighted_score" not in trial:
                    pr = get_trial_probe_results(trial)
                    if pr:
                        trial["weighted_score"] = score_weighted_combination(pr, combo)

        fig_det = detection_rate_bars(
            summary, combo["operating_points"], regimes, trials_by_regime,
        )
        st.plotly_chart(fig_det, use_container_width=True)

    # ─── Token analysis ──────────────────────────────────────────────
    st.subheader("Token Analysis")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_tokens = token_stacked_bar(trials_by_regime)
        st.plotly_chart(fig_tokens, use_container_width=True)

    with col_t2:
        # Scatter: thinking tokens vs probe confidence
        import plotly.graph_objects as go
        fig_scatter = go.Figure()
        for regime, trials in trials_by_regime.items():
            x_vals = []
            y_vals = []
            for trial in trials:
                if "turns" in trial and trial["turns"]:
                    t = trial["turns"][-1]
                else:
                    t = trial
                think_tok = t.get("thinking_tokens", 0)
                pr = t.get("probe_results", {})
                pos_data = pr.get(position, {})
                confs = [pos_data.get(l, pos_data.get(str(l), {})).get("mean_confidence", None)
                         for l in target_layers]
                confs = [c for c in confs if c is not None]
                if confs:
                    x_vals.append(think_tok)
                    y_vals.append(np.mean(confs))
            if x_vals:
                fig_scatter.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    name=regime, opacity=0.7,
                ))
        fig_scatter.update_layout(
            title="Thinking Tokens vs Confidence",
            xaxis_title="Thinking Tokens",
            yaxis_title="Mean Confidence",
            yaxis=dict(range=[0, 1]),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ─── Response viewer ─────────────────────────────────────────────
    st.subheader("Response Viewer")

    viewer_regime = st.selectbox(
        "Regime", regimes, key="viewer_regime",
    )
    viewer_trials = trials_by_regime.get(viewer_regime, [])
    questions_available = get_unique_questions(viewer_trials)
    viewer_question = st.selectbox(
        "Question", questions_available, key="viewer_question",
    )

    # Filter to question
    q_trials = [t for t in viewer_trials if t["question_id"] == viewer_question]
    rollouts = get_unique_rollouts(q_trials)
    viewer_rollout = st.selectbox(
        "Rollout", rollouts, key="viewer_rollout",
    )

    # Find the trial
    trial = next((t for t in q_trials if t["rollout"] == viewer_rollout), None)
    if trial is None:
        st.warning("Trial not found.")
        return

    # Multi-turn support
    if "turns" in trial and trial["turns"]:
        turn_idx = st.slider(
            "Turn", 0, len(trial["turns"]) - 1,
            value=len(trial["turns"]) - 1,
            key="viewer_turn",
        )
        turn_data = trial["turns"][turn_idx]
    else:
        turn_data = trial

    # Two-column layout
    left, right = st.columns([3, 2])

    with left:
        # Question text
        st.markdown(f"**Question:** {trial.get('question', '')}")

        # Thinking trace
        thinking = turn_data.get("thinking_trace", "")
        if thinking:
            with st.expander("Thinking Trace", expanded=False):
                st.text(thinking)

        # Answer
        answer = turn_data.get("answer", "")
        st.markdown("**Answer:**")
        st.markdown(answer)

        # Metadata
        think_tok = turn_data.get("thinking_tokens", 0)
        ans_tok = turn_data.get("answer_tokens", 0)
        truncated = turn_data.get("think_truncated", False)
        st.caption(f"Thinking: {think_tok} tokens | Answer: {ans_tok} tokens"
                   + (" | TRUNCATED" if truncated else ""))

    with right:
        st.markdown("**Probe Scores**")
        pr = turn_data.get("probe_results", {})
        show_layers = target_layers[:10] if len(target_layers) > 10 else target_layers
        render_probe_scores(pr, all_positions, target_layers, show_layers=show_layers)

        # Weighted score
        ws = turn_data.get("weighted_score", {})
        if ws:
            score = ws.get("combined_score", 0)
            color = confidence_color(score)
            st.markdown(f"**Weighted Score:** "
                        f"<span style='color:{color};font-size:18px;font-weight:bold;'>"
                        f"{score:.4f}</span>", unsafe_allow_html=True)
            dets = ws.get("detections", {})
            det_strs = [f"{k}: {'YES' if v else 'no'}" for k, v in dets.items()]
            st.caption(" | ".join(det_strs))
