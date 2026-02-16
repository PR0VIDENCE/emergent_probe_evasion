"""Tab 3: FP Analysis — false positive rate analysis of negative control trials."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from components.data_loader import load_trials, get_trial_probe_results, get_unique_questions
from components.charts import probe_heatmap, DARK_LAYOUT
from components.probe_display import confidence_color
from state import resolve_path, get_target_layers


def render():
    st.header("FP Analysis")

    # ─── Input ───────────────────────────────────────────────────────
    data_dir = st.text_input(
        "Results directory",
        value="data/outputs/evasion",
        key="fp_data_dir",
    )
    abs_dir = resolve_path(data_dir)

    regime_name = st.text_input(
        "Negative control regime name",
        value="negative_control_baseline",
        key="fp_regime_name",
    )

    trials = load_trials(abs_dir, regime_name)
    if trials is None:
        # Also try without "negative_" prefix
        trials = load_trials(abs_dir, "negative_baseline")
    if trials is None:
        st.warning(f"No trial data found for '{regime_name}' in {abs_dir}")
        st.info("Run the evasion experiment with negative control questions to generate this data.")
        return

    st.success(f"Loaded {len(trials)} negative control trials")

    # Determine layers and positions
    target_layers = get_target_layers()
    pr_first = get_trial_probe_results(trials[0])
    all_positions = sorted(pr_first.keys())

    if not target_layers:
        first_pos = next(iter(pr_first.values()), {})
        target_layers = sorted(int(k) for k in first_pos.keys() if str(k).isdigit())

    # Position selector
    position = st.selectbox(
        "Position", all_positions,
        index=all_positions.index("answer_mean_pool") if "answer_mean_pool" in all_positions else 0,
        key="fp_position",
    )

    # FP threshold
    threshold = st.slider("FP threshold", 0.0, 1.0, 0.5, 0.01, key="fp_threshold")

    # ─── Compute FP statistics ───────────────────────────────────────
    questions = get_unique_questions(trials)

    # Per-question, per-layer FP rates
    fp_data = {}  # {question_id: {layer: {confs: [], fp_rate: float}}}
    for qid in questions:
        q_trials = [t for t in trials if t["question_id"] == qid]
        layer_data = {}
        for layer_idx in target_layers:
            confs = []
            for trial in q_trials:
                pr = get_trial_probe_results(trial)
                pos_data = pr.get(position, {})
                ld = pos_data.get(layer_idx, pos_data.get(str(layer_idx), {}))
                if ld:
                    confs.append(ld.get("mean_confidence", 0))
            fp_rate = np.mean([c >= threshold for c in confs]) if confs else 0
            layer_data[layer_idx] = {"confs": confs, "fp_rate": fp_rate, "mean_conf": np.mean(confs) if confs else 0}
        fp_data[qid] = layer_data

    # Overall stats
    all_confs = []
    for trial in trials:
        pr = get_trial_probe_results(trial)
        pos_data = pr.get(position, {})
        for layer_idx in target_layers:
            ld = pos_data.get(layer_idx, pos_data.get(str(layer_idx), {}))
            if ld:
                all_confs.append(ld.get("mean_confidence", 0))

    overall_fp_rate = np.mean([c >= threshold for c in all_confs]) if all_confs else 0

    # Worst-case question
    q_fp_rates = {}
    for qid, layer_data in fp_data.items():
        all_q_confs = []
        for ld in layer_data.values():
            all_q_confs.extend(ld["confs"])
        q_fp_rates[qid] = np.mean([c >= threshold for c in all_q_confs]) if all_q_confs else 0

    worst_q = max(q_fp_rates, key=q_fp_rates.get) if q_fp_rates else "N/A"
    worst_fp = q_fp_rates.get(worst_q, 0)

    # ─── Summary cards ───────────────────────────────────────────────
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        color = confidence_color(overall_fp_rate)
        st.metric("Overall FP Rate", f"{overall_fp_rate:.1%}")
    with col2:
        st.metric("Worst Question", worst_q)
        st.caption(f"FP Rate: {worst_fp:.1%}")
    with col3:
        combo = st.session_state.get("weighted_combo")
        if combo is not None:
            from src.probes.evaluate import score_weighted_combination
            combo_fps = 0
            combo_total = 0
            for trial in trials:
                pr = get_trial_probe_results(trial)
                if pr:
                    ws = score_weighted_combination(pr, combo)
                    if ws["combined_score"] >= threshold:
                        combo_fps += 1
                    combo_total += 1
            combo_fp_rate = combo_fps / combo_total if combo_total else 0
            st.metric("Weighted Combo FP Rate", f"{combo_fp_rate:.1%}")
        else:
            st.metric("Weighted Combo FP Rate", "N/A")
            st.caption("Load weighted combo in sidebar")

    # ─── Heatmap: question x layer FP rate ───────────────────────────
    st.subheader("Question x Layer FP Rate")
    matrix = []
    for qid in questions:
        row = [fp_data[qid][l]["fp_rate"] for l in target_layers]
        matrix.append(row)

    fig = probe_heatmap(
        matrix,
        x_labels=[f"L{l}" for l in target_layers],
        y_labels=questions,
        title=f"FP Rate — {position} (threshold={threshold})",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Heatmap: question x layer mean confidence ───────────────────
    st.subheader("Question x Layer Mean Confidence")
    conf_matrix = []
    for qid in questions:
        row = [fp_data[qid][l]["mean_conf"] for l in target_layers]
        conf_matrix.append(row)

    fig_conf = probe_heatmap(
        conf_matrix,
        x_labels=[f"L{l}" for l in target_layers],
        y_labels=questions,
        title=f"Mean Confidence — {position}",
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # ─── Sortable dataframe ──────────────────────────────────────────
    st.subheader("Per-Question Breakdown")
    import pandas as pd

    rows = []
    for qid in questions:
        q_trial = next((t for t in trials if t["question_id"] == qid), {})
        q_text = q_trial.get("question", qid)
        row = {"question_id": qid, "question": q_text[:80], "fp_rate": q_fp_rates.get(qid, 0)}
        for l in target_layers[:10]:  # Limit columns
            row[f"L{l}_conf"] = fp_data[qid][l]["mean_conf"]
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=[c for c in df.columns if c.endswith("_conf")],
                                      cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
        height=400,
    )

    # ─── Weighted combination analysis ───────────────────────────────
    if combo is not None:
        st.subheader("Weighted Combination Analysis")

        # Score all trials
        from src.probes.evaluate import score_weighted_combination
        combo_scores = []
        for trial in trials:
            pr = get_trial_probe_results(trial)
            if pr:
                ws = score_weighted_combination(pr, combo)
                combo_scores.append(ws["combined_score"])

        if combo_scores:
            # Histogram
            import plotly.graph_objects as go
            fig_hist = go.Figure(data=[
                go.Histogram(x=combo_scores, nbinsx=30, name="Score distribution")
            ])
            # Add operating point lines
            for op_name, op in combo["operating_points"].items():
                fig_hist.add_vline(
                    x=op["threshold"],
                    line_dash="dash",
                    annotation_text=op_name,
                    annotation_position="top",
                )
            fig_hist.update_layout(
                title="Negative Control Score Distribution",
                xaxis_title="Combined Score",
                yaxis_title="Count",
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Operating point analysis
            st.markdown("**Observed vs Expected FP Rates**")
            op_rows = []
            for op_name, op in combo["operating_points"].items():
                observed_fpr = np.mean([s >= op["threshold"] for s in combo_scores])
                op_rows.append({
                    "Operating Point": op_name,
                    "Threshold": f"{op['threshold']:.4f}",
                    "Expected FPR": f"{op['expected_fpr']:.1%}",
                    "Observed FPR": f"{observed_fpr:.1%}",
                    "Expected TPR": f"{op['expected_tpr']:.1%}",
                })
            st.table(pd.DataFrame(op_rows))

            # Custom threshold slider
            custom_thresh = st.slider(
                "Custom threshold", 0.0, 1.0, 0.5, 0.01,
                key="fp_custom_threshold",
            )
            custom_fp = np.mean([s >= custom_thresh for s in combo_scores])
            st.metric(f"FP Rate at threshold {custom_thresh:.2f}", f"{custom_fp:.1%}")

    # ─── Problematic questions ───────────────────────────────────────
    st.subheader("Problematic Questions (sorted by FP rate)")
    sorted_qs = sorted(q_fp_rates.items(), key=lambda x: x[1], reverse=True)

    for qid, fp_rate in sorted_qs:
        if fp_rate == 0:
            continue
        q_trial = next((t for t in trials if t["question_id"] == qid), {})
        q_text = q_trial.get("question", qid)
        color = confidence_color(fp_rate)
        header = f"{qid} — FP Rate: {fp_rate:.1%}"

        with st.expander(header, expanded=False):
            st.markdown(f"**Question:** {q_text}")
            # Show a sample response
            q_trials = [t for t in trials if t["question_id"] == qid]
            if q_trials:
                sample = q_trials[0]
                if "turns" in sample and sample["turns"]:
                    sample = sample["turns"][-1]
                answer = sample.get("answer", "")
                if answer:
                    st.markdown(f"**Sample answer:** {answer[:500]}")
