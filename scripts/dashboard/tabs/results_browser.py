"""Tab 5: Results Browser — TP/TN analysis, per-question breakdown, discrimination metrics, response viewer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from components.data_loader import (
    discover_regimes, load_all_trials, get_trial_probe_results,
    get_unique_questions, get_unique_rollouts, load_summary,
    classify_trials, get_trial_confidence,
)
from components.charts import (
    probe_heatmap, regime_comparison_bars, layer_line_chart,
    confidence_box_plots, token_stacked_bar, detection_rate_bars,
    tp_tn_distribution, roc_curve_chart, separation_heatmap,
    per_question_strip_chart, tp_tn_box_by_regime,
)
from components.probe_display import render_probe_scores, confidence_color

from state import get_target_layers


def _compute_dprime(pos_confs, neg_confs):
    """Compute d-prime: (mean_pos - mean_neg) / sqrt(0.5 * (var_pos + var_neg))."""
    if len(pos_confs) < 2 or len(neg_confs) < 2:
        return None
    var_pos = np.var(pos_confs, ddof=1)
    var_neg = np.var(neg_confs, ddof=1)
    denom = np.sqrt(0.5 * (var_pos + var_neg))
    if denom < 1e-10:
        return None
    return (np.mean(pos_confs) - np.mean(neg_confs)) / denom


def _compute_auc(pos_confs, neg_confs):
    """Compute AUC for binary classification (positive=1, negative=0)."""
    if not pos_confs or not neg_confs:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        labels = [1] * len(pos_confs) + [0] * len(neg_confs)
        scores = list(pos_confs) + list(neg_confs)
        return roc_auc_score(labels, scores)
    except Exception:
        return None


def _compute_roc(pos_confs, neg_confs):
    """Compute ROC curve points. Returns (fpr, tpr, auc) or None."""
    if not pos_confs or not neg_confs:
        return None
    try:
        from sklearn.metrics import roc_curve, auc
        labels = [1] * len(pos_confs) + [0] * len(neg_confs)
        scores = list(pos_confs) + list(neg_confs)
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr.tolist(), tpr.tolist(), auc(fpr, tpr)
    except Exception:
        return None


def _detection_rate(confs, threshold):
    """Fraction of confidences above threshold."""
    if not confs:
        return 0.0
    return sum(1 for c in confs if c >= threshold) / len(confs)


def render():
    st.header("Results Browser")

    # ─── Data loading ────────────────────────────────────────────────
    data_dir = st.text_input(
        "Results directory",
        value="data/outputs/evasion",
        key="results_browser_dir",
    )

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
        first_trials = next(iter(trials_by_regime.values()))
        pr = get_trial_probe_results(first_trials[0])
        first_pos = next(iter(pr.values()), {})
        target_layers = sorted(int(k) for k in first_pos.keys()
                               if str(k).isdigit())

    # Determine positions from data
    first_trials = next(iter(trials_by_regime.values()))
    pr = get_trial_probe_results(first_trials[0])
    all_positions = sorted(pr.keys())
    regimes = list(trials_by_regime.keys())

    # ─── Global controls ─────────────────────────────────────────────
    col_pos, col_layers = st.columns(2)
    with col_pos:
        position = st.selectbox(
            "Position", all_positions,
            index=all_positions.index("answer_mean_pool") if "answer_mean_pool" in all_positions else 0,
            key="results_browser_position",
        )
    with col_layers:
        layer_subset = st.multiselect(
            "Layers to average",
            target_layers,
            default=target_layers,
            key="results_browser_layer_subset",
        )

    if not layer_subset:
        st.info("Select at least one layer.")
        return

    # ─── Pre-compute per-trial confidences and classifications ────────
    # Build flat data structures for all analyses
    all_regime_data = []  # for box plots: {regime, category, confidence}
    all_strip_data = []   # for strip chart: {question_id, confidence, category, regime}

    classified_by_regime = {}
    pos_confs_by_regime = {}
    neg_confs_by_regime = {}

    for regime in regimes:
        trials = trials_by_regime[regime]
        classified = classify_trials(trials)
        classified_by_regime[regime] = classified

        pos_confs = []
        for trial in classified["positive"]:
            c = get_trial_confidence(trial, position, layer_subset)
            if c is not None:
                pos_confs.append(c)
                all_regime_data.append({"regime": regime, "category": "positive", "confidence": c})
                all_strip_data.append({
                    "question_id": trial["question_id"],
                    "confidence": c, "category": "positive", "regime": regime,
                })

        neg_confs = []
        for trial in classified["negative"]:
            c = get_trial_confidence(trial, position, layer_subset)
            if c is not None:
                neg_confs.append(c)
                all_regime_data.append({"regime": regime, "category": "negative", "confidence": c})
                all_strip_data.append({
                    "question_id": trial["question_id"],
                    "confidence": c, "category": "negative", "regime": regime,
                })

        pos_confs_by_regime[regime] = pos_confs
        neg_confs_by_regime[regime] = neg_confs

    # ─────────────────────────────────────────────────────────────────
    # Section 1: TP/TN Separation
    # ─────────────────────────────────────────────────────────────────
    st.subheader("1. TP/TN Separation")

    # Summary metrics
    cols = st.columns(len(regimes))
    for i, regime in enumerate(regimes):
        pos = pos_confs_by_regime[regime]
        neg = neg_confs_by_regime[regime]
        with cols[i]:
            st.markdown(f"**{regime}**")
            st.caption(
                f"Tree: n={len(pos)}, "
                f"mean={np.mean(pos):.3f}" if pos else f"Tree: n=0"
            )
            st.caption(
                f"Non-tree: n={len(neg)}, "
                f"mean={np.mean(neg):.3f}" if neg else f"Non-tree: n=0"
            )

    # Side-by-side box plots
    fig_box = tp_tn_box_by_regime(
        all_regime_data,
        title=f"Tree vs Non-tree Distribution — {position}",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Per-regime histograms
    hist_cols = st.columns(min(len(regimes), 3))
    for i, regime in enumerate(regimes):
        with hist_cols[i % len(hist_cols)]:
            fig_hist = tp_tn_distribution(
                pos_confs_by_regime[regime],
                neg_confs_by_regime[regime],
                title=regime,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # Section 2: Per-Question Breakdown
    # ─────────────────────────────────────────────────────────────────
    st.subheader("2. Per-Question Breakdown")

    # Collect all question IDs, sorted: positives first, then negatives
    all_questions_pos = []
    all_questions_neg = []
    question_texts = {}
    for trials in trials_by_regime.values():
        for trial in trials:
            qid = trial["question_id"]
            if qid not in question_texts:
                question_texts[qid] = trial.get("question", "")
            if qid.startswith("nc"):
                if qid not in all_questions_neg:
                    all_questions_neg.append(qid)
            else:
                if qid not in all_questions_pos:
                    all_questions_pos.append(qid)

    all_questions_ordered = sorted(all_questions_pos) + sorted(all_questions_neg)

    # Build per-question mean confidence matrix: questions (rows) x regimes (cols)
    q_matrix = []
    q_labels = []
    for qid in all_questions_ordered:
        row = []
        for regime in regimes:
            trials = trials_by_regime[regime]
            confs = []
            for trial in trials:
                if trial["question_id"] == qid:
                    c = get_trial_confidence(trial, position, layer_subset)
                    if c is not None:
                        confs.append(c)
            row.append(np.mean(confs) if confs else float("nan"))
        q_matrix.append(row)
        q_labels.append(qid)

    if q_matrix:
        # Heatmap: question_id (rows) x regime (cols)
        fig_qheat = probe_heatmap(
            q_matrix,
            x_labels=regimes,
            y_labels=q_labels,
            title=f"Mean Confidence by Question — {position}",
        )
        # Add horizontal line to separate positive and negative questions
        if all_questions_pos and all_questions_neg:
            divider_y = len(all_questions_pos) - 0.5
            fig_qheat.add_hline(y=divider_y, line_dash="dash", line_color="white",
                                line_width=2, annotation_text="── positive ↑ │ negative ↓ ──",
                                annotation_position="top right",
                                annotation_font_color="white")
        st.plotly_chart(fig_qheat, use_container_width=True)

    # Strip chart
    fig_strip = per_question_strip_chart(
        all_strip_data,
        title=f"Individual Trial Scores by Question — {position}",
    )
    st.plotly_chart(fig_strip, use_container_width=True)

    # Sortable dataframe
    df_rows = []
    for qid in all_questions_ordered:
        cat = "negative" if qid.startswith("nc") else "positive"
        text = question_texts.get(qid, "")
        if len(text) > 80:
            text = text[:77] + "..."
        row_data = {"question_id": qid, "category": cat, "question": text}
        for regime in regimes:
            trials = trials_by_regime[regime]
            confs = []
            for trial in trials:
                if trial["question_id"] == qid:
                    c = get_trial_confidence(trial, position, layer_subset)
                    if c is not None:
                        confs.append(c)
            row_data[regime] = round(np.mean(confs), 4) if confs else None
        df_rows.append(row_data)

    if df_rows:
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Identify extremes
        col_hard, col_fp = st.columns(2)
        with col_hard:
            pos_rows = [r for r in df_rows if r["category"] == "positive"]
            if pos_rows:
                # Hardest to detect: lowest mean confidence across regimes
                for r in pos_rows:
                    regime_vals = [r[reg] for reg in regimes if r.get(reg) is not None]
                    r["_mean"] = np.mean(regime_vals) if regime_vals else 1.0
                sorted_pos = sorted(pos_rows, key=lambda x: x["_mean"])
                st.markdown("**Hardest-to-detect tree questions:**")
                for r in sorted_pos[:3]:
                    st.caption(f"{r['question_id']}: mean conf={r['_mean']:.3f} — {r['question']}")

        with col_fp:
            neg_rows = [r for r in df_rows if r["category"] == "negative"]
            if neg_rows:
                # Highest FP: highest mean confidence across regimes
                for r in neg_rows:
                    regime_vals = [r[reg] for reg in regimes if r.get(reg) is not None]
                    r["_mean"] = np.mean(regime_vals) if regime_vals else 0.0
                sorted_neg = sorted(neg_rows, key=lambda x: x["_mean"], reverse=True)
                st.markdown("**Highest false-positive negative questions:**")
                for r in sorted_neg[:3]:
                    st.caption(f"{r['question_id']}: mean conf={r['_mean']:.3f} — {r['question']}")

    # ─────────────────────────────────────────────────────────────────
    # Section 3: Discrimination Metrics
    # ─────────────────────────────────────────────────────────────────
    st.subheader("3. Discrimination Metrics")

    # Check we have both positive and negative trials
    has_both = any(pos_confs_by_regime[r] for r in regimes) and any(neg_confs_by_regime[r] for r in regimes)
    if not has_both:
        st.info("Need both positive (tree) and negative (non-tree) trials for discrimination metrics. "
                "Make sure your results include both question types.")
    else:
        # Per-regime, per-layer AUC heatmap
        st.markdown("#### AUC Heatmap: Regime × Layer")
        auc_matrix = []
        for regime in regimes:
            row = []
            for layer_idx in layer_subset:
                pos_c = []
                neg_c = []
                for trial in classified_by_regime[regime]["positive"]:
                    c = get_trial_confidence(trial, position, [layer_idx])
                    if c is not None:
                        pos_c.append(c)
                for trial in classified_by_regime[regime]["negative"]:
                    c = get_trial_confidence(trial, position, [layer_idx])
                    if c is not None:
                        neg_c.append(c)
                auc_val = _compute_auc(pos_c, neg_c)
                row.append(auc_val if auc_val is not None else 0.5)
            auc_matrix.append(row)

        fig_auc_heat = separation_heatmap(
            auc_matrix,
            x_labels=[f"L{l}" for l in layer_subset],
            y_labels=regimes,
            title=f"AUC by Regime × Layer — {position}",
            colorbar_title="AUC",
        )
        st.plotly_chart(fig_auc_heat, use_container_width=True)

        # Summary table
        st.markdown("#### Summary Table")
        summary_rows = []
        for regime in regimes:
            pos = pos_confs_by_regime[regime]
            neg = neg_confs_by_regime[regime]
            auc_val = _compute_auc(pos, neg)
            dprime = _compute_dprime(pos, neg)

            # Detection rates at thresholds
            tpr_50 = _detection_rate(pos, 0.5)
            fpr_50 = _detection_rate(neg, 0.5)
            tpr_80 = _detection_rate(pos, 0.8)
            fpr_80 = _detection_rate(neg, 0.8)

            # Best discriminating layer (highest AUC)
            best_layer = None
            best_layer_auc = 0.0
            for layer_idx in layer_subset:
                pos_c = []
                neg_c = []
                for trial in classified_by_regime[regime]["positive"]:
                    c = get_trial_confidence(trial, position, [layer_idx])
                    if c is not None:
                        pos_c.append(c)
                for trial in classified_by_regime[regime]["negative"]:
                    c = get_trial_confidence(trial, position, [layer_idx])
                    if c is not None:
                        neg_c.append(c)
                layer_auc = _compute_auc(pos_c, neg_c)
                if layer_auc is not None and layer_auc > best_layer_auc:
                    best_layer_auc = layer_auc
                    best_layer = layer_idx

            summary_rows.append({
                "Regime": regime,
                "AUC": round(auc_val, 4) if auc_val is not None else "N/A",
                "d'": round(dprime, 3) if dprime is not None else "N/A",
                "TPR@0.5": f"{tpr_50:.1%}",
                "FPR@0.5": f"{fpr_50:.1%}",
                "TPR@0.8": f"{tpr_80:.1%}",
                "FPR@0.8": f"{fpr_80:.1%}",
                "Best Layer": f"L{best_layer}" if best_layer is not None else "N/A",
                "Best Layer AUC": round(best_layer_auc, 4) if best_layer is not None else "N/A",
            })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # ROC curves overlay
        st.markdown("#### ROC Curves")
        roc_layer_select = st.selectbox(
            "Layer for ROC",
            layer_subset,
            index=0,
            key="roc_layer_select",
        )

        roc_curves = []
        for regime in regimes:
            pos_c = []
            neg_c = []
            for trial in classified_by_regime[regime]["positive"]:
                c = get_trial_confidence(trial, position, [roc_layer_select])
                if c is not None:
                    pos_c.append(c)
            for trial in classified_by_regime[regime]["negative"]:
                c = get_trial_confidence(trial, position, [roc_layer_select])
                if c is not None:
                    neg_c.append(c)
            roc_result = _compute_roc(pos_c, neg_c)
            if roc_result is not None:
                fpr, tpr, auc_val = roc_result
                roc_curves.append({"fpr": fpr, "tpr": tpr, "auc": auc_val, "label": regime})

        if roc_curves:
            fig_roc = roc_curve_chart(
                roc_curves,
                title=f"ROC Curves — {position}, Layer {roc_layer_select}",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Not enough data to compute ROC curves.")

    # ─────────────────────────────────────────────────────────────────
    # Section 4: Response Viewer (kept from original)
    # ─────────────────────────────────────────────────────────────────
    st.subheader("4. Response Viewer")

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
        show_layers = target_layers
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
