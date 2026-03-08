"""Tab 2: Compare Probes — side-by-side comparison of two probe sets."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from state import (
    model_is_loaded, probes_are_loaded, get_target_layers,
    load_probes, resolve_path,
)
from components.data_loader import (
    discover_regimes, load_trials, get_trial_probe_results,
    get_unique_questions,
)
from components.charts import probe_heatmap, DARK_LAYOUT
from components.probe_display import confidence_color

from prefill_ablation import assemble_prefill_sequence, run_prefill_probe


def render():
    st.header("Compare Probes")

    if not model_is_loaded():
        st.warning("Load model in sidebar first.")
        return

    config = st.session_state.experiment_config
    if config is None:
        st.warning("Load experiment config in sidebar first.")
        return

    target_layers = get_target_layers()

    # ─── Probe set setup ─────────────────────────────────────────────
    st.subheader("Probe Sets")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Probe Set A**")
        probe_dir_a = st.text_input("Probe directory A",
                                     value="/workspace/probe_data/probes",
                                     key="compare_probe_dir_a")
        if st.button("Load A", key="compare_load_a"):
            load_probes(probe_dir_a, key="probe_set_primary")
            st.rerun()
        if probes_are_loaded("probe_set_primary"):
            ps_a = st.session_state.probe_set_primary
            st.success(f"A: {len(ps_a['layers'])} layers")
        else:
            st.info("Not loaded")

    with col_b:
        st.markdown("**Probe Set B**")
        probe_dir_b = st.text_input("Probe directory B",
                                     value="/workspace/probe_data/probes_v2",
                                     key="compare_probe_dir_b")
        if st.button("Load B", key="compare_load_b"):
            load_probes(probe_dir_b, key="probe_set_secondary")
            st.rerun()
        if probes_are_loaded("probe_set_secondary"):
            ps_b = st.session_state.probe_set_secondary
            st.success(f"B: {len(ps_b['layers'])} layers")
        else:
            st.info("Not loaded")

    if not probes_are_loaded("probe_set_primary") or not probes_are_loaded("probe_set_secondary"):
        st.info("Load both probe sets to compare.")
        return

    ps_a = st.session_state.probe_set_primary
    ps_b = st.session_state.probe_set_secondary

    # ─── Trial data selection ────────────────────────────────────────
    st.subheader("Trial Data")
    data_dir = st.text_input("Results directory",
                              value="data/outputs/evasion",
                              key="compare_data_dir")
    abs_dir = resolve_path(data_dir)

    available = discover_regimes(abs_dir)
    if not available:
        st.warning(f"No regime files found in {abs_dir}")
        return

    selected_regime = st.selectbox("Regime", available, key="compare_regime")
    trials = load_trials(abs_dir, selected_regime)
    if not trials:
        st.warning("No trials loaded.")
        return

    st.caption(f"{len(trials)} trials loaded")

    # Question filter
    all_questions = get_unique_questions(trials)
    selected_questions = st.multiselect(
        "Questions", all_questions, default=all_questions[:5],
        key="compare_questions",
    )
    filtered_trials = [t for t in trials if t["question_id"] in selected_questions]

    if not filtered_trials:
        st.info("Select questions to compare.")
        return

    # ─── Scoring ─────────────────────────────────────────────────────
    st.subheader("Score Trials")

    max_trials = st.number_input("Max trials to score", 1, len(filtered_trials),
                                  min(10, len(filtered_trials)),
                                  key="compare_max_trials")

    if st.button("Score with both probe sets", type="primary", key="compare_score_btn"):
        score_trials = filtered_trials[:max_trials]
        results_a = []
        results_b = []

        progress = st.progress(0)
        for i, trial in enumerate(score_trials):
            progress.progress((i + 1) / len(score_trials),
                             text=f"Scoring trial {i+1}/{len(score_trials)}...")

            thinking = trial.get("thinking_trace", "")
            answer = trial.get("answer", "")
            if not answer:
                continue

            # Build prompt (use baseline template for consistency)
            q_text = trial.get("question", "")
            prompt_text = f"Answer the following question.\n\nQuestion: {q_text}\n\nPlease provide your answer:"

            with st.session_state.gpu_lock:
                output_ids, input_len = assemble_prefill_sequence(
                    prompt_text, thinking, answer,
                    st.session_state.tokenizer,
                )

                # Score with probe set A
                res_a = run_prefill_probe(
                    output_ids, input_len,
                    st.session_state.model, st.session_state.tokenizer,
                    target_layers, ps_a["probes"], ps_a["scalers"], ps_a["per_position"],
                )

                # Score with probe set B
                res_b = run_prefill_probe(
                    output_ids, input_len,
                    st.session_state.model, st.session_state.tokenizer,
                    target_layers, ps_b["probes"], ps_b["scalers"], ps_b["per_position"],
                )

            results_a.append({"trial": trial, **res_a})
            results_b.append({"trial": trial, **res_b})

        progress.empty()
        st.session_state["compare_results_a"] = results_a
        st.session_state["compare_results_b"] = results_b

    # ─── Visualizations ──────────────────────────────────────────────
    results_a = st.session_state.get("compare_results_a", [])
    results_b = st.session_state.get("compare_results_b", [])

    if not results_a or not results_b:
        return

    st.success(f"Scored {len(results_a)} trials with both probe sets")

    # Determine positions
    pr_first = results_a[0].get("probe_results", {})
    all_positions = sorted(pr_first.keys())
    position = st.selectbox(
        "Position", all_positions,
        index=all_positions.index("answer_mean_pool") if "answer_mean_pool" in all_positions else 0,
        key="compare_position",
    )

    # Side-by-side heatmaps (position x layer)
    st.subheader("Mean Confidence Heatmaps")
    col_ha, col_hb = st.columns(2)

    def _build_layer_means(results, position, layers):
        """Build mean confidence per layer across all scored trials."""
        layer_confs = {l: [] for l in layers}
        for res in results:
            pr = res.get("probe_results", {})
            pos_data = pr.get(position, {})
            for l in layers:
                ld = pos_data.get(l, pos_data.get(str(l), {}))
                if ld:
                    layer_confs[l].append(ld.get("mean_confidence", 0))
        return [np.mean(layer_confs[l]) if layer_confs[l] else 0 for l in layers]

    means_a = _build_layer_means(results_a, position, target_layers)
    means_b = _build_layer_means(results_b, position, target_layers)

    with col_ha:
        fig_a = probe_heatmap(
            [means_a],
            x_labels=[f"L{l}" for l in target_layers],
            y_labels=["Probe Set A"],
            title=f"Probe Set A — {position}",
        )
        st.plotly_chart(fig_a, use_container_width=True)

    with col_hb:
        fig_b = probe_heatmap(
            [means_b],
            x_labels=[f"L{l}" for l in target_layers],
            y_labels=["Probe Set B"],
            title=f"Probe Set B — {position}",
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # Scatter plot: A vs B per trial
    st.subheader("Probe Set A vs B (per trial)")
    scatter_a = []
    scatter_b = []
    for res_a, res_b in zip(results_a, results_b):
        pr_a = res_a.get("probe_results", {}).get(position, {})
        pr_b = res_b.get("probe_results", {}).get(position, {})
        confs_a = [pr_a.get(l, pr_a.get(str(l), {})).get("mean_confidence", None) for l in target_layers]
        confs_b = [pr_b.get(l, pr_b.get(str(l), {})).get("mean_confidence", None) for l in target_layers]
        confs_a = [c for c in confs_a if c is not None]
        confs_b = [c for c in confs_b if c is not None]
        if confs_a and confs_b:
            scatter_a.append(np.mean(confs_a))
            scatter_b.append(np.mean(confs_b))

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=scatter_a, y=scatter_b, mode="markers",
        marker=dict(size=8, opacity=0.7),
        text=[f"Trial {i}" for i in range(len(scatter_a))],
    ))
    # Diagonal line
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(dash="dash", color="gray"))
    fig_scatter.update_layout(
        title=f"A vs B — {position}",
        xaxis_title="Probe Set A (mean confidence)",
        yaxis_title="Probe Set B (mean confidence)",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Per-layer comparison bar chart
    st.subheader("Per-Layer Comparison")
    fig_layer = go.Figure()
    fig_layer.add_trace(go.Bar(
        name="Probe Set A",
        x=[f"L{l}" for l in target_layers],
        y=means_a,
    ))
    fig_layer.add_trace(go.Bar(
        name="Probe Set B",
        x=[f"L{l}" for l in target_layers],
        y=means_b,
    ))
    fig_layer.update_layout(
        barmode="group",
        title=f"Mean Confidence by Layer — {position}",
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_layer, use_container_width=True)

    # Summary table with deltas
    st.subheader("Summary")
    import pandas as pd
    rows = []
    for i, l in enumerate(target_layers):
        delta = means_b[i] - means_a[i]
        rows.append({
            "Layer": f"L{l}",
            "Probe A": f"{means_a[i]:.4f}",
            "Probe B": f"{means_b[i]:.4f}",
            "Delta (B-A)": f"{delta:+.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
