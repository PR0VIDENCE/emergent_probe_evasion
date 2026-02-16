"""Tab 1: Run Experiments — configure and run evasion experiments with live results."""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from state import (
    model_is_loaded, probes_are_loaded, get_all_regimes, get_questions,
    get_negative_questions, get_target_layers, get_generation_config,
    resolve_path,
)
from components.charts import probe_heatmap, regime_comparison_bars, DARK_LAYOUT
from components.data_loader import get_trial_probe_results
from experiment_runner import ExperimentRunner


def render():
    st.header("Run Experiments")

    # ─── Prerequisites check ─────────────────────────────────────────
    if not model_is_loaded():
        st.warning("Load model in sidebar first.")
        return
    if not probes_are_loaded():
        st.warning("Load probes in sidebar first.")
        return
    config = st.session_state.experiment_config
    if config is None:
        st.warning("Load experiment config in sidebar first.")
        return

    # ─── Config section ──────────────────────────────────────────────
    st.subheader("Configuration")

    all_regimes = get_all_regimes()
    regime_names = [r["name"] for r in all_regimes]
    main_regime_names = [r["name"] for r in config.get("regimes", [])]

    selected_regime_names = st.multiselect(
        "Regimes",
        regime_names,
        default=main_regime_names[:3],
        key="run_regimes",
    )
    selected_regimes = [r for r in all_regimes if r["name"] in selected_regime_names]

    # Questions
    all_questions = get_questions()
    neg_questions = get_negative_questions()
    question_ids = [q["id"] for q in all_questions]

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        selected_question_ids = st.multiselect(
            "Questions",
            question_ids,
            default=question_ids[:5],
            key="run_questions",
        )
    with col_q2:
        include_negative = st.checkbox("Include negative controls", value=False, key="run_include_neg")

    selected_questions = [q for q in all_questions if q["id"] in selected_question_ids]
    if include_negative:
        selected_questions.extend(neg_questions)

    # Generation settings
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        num_rollouts = st.number_input("Rollouts", 1, 10, 1, key="run_rollouts")
    with col_s2:
        batch_size = st.number_input("Batch size", 1, 16, 5, key="run_batch_size")
    with col_s3:
        max_tokens = st.number_input("Max tokens (0=use regime default)", 0, 16384, 0, key="run_max_tokens")

    max_tokens_override = max_tokens if max_tokens > 0 else None

    # Prompt template editor
    with st.expander("Prompt Template Editor", expanded=False):
        templates = st.session_state.prompt_templates
        if templates:
            # Only show templates used by selected regimes
            used_templates = set()
            for r in selected_regimes:
                used_templates.add(r.get("template", ""))
                used_templates.add(r.get("template_initial", ""))
                used_templates.add(r.get("template_followup", ""))
            used_templates.discard("")

            for tname in sorted(used_templates):
                if tname in templates:
                    new_val = st.text_area(
                        tname,
                        value=templates[tname],
                        height=200,
                        key=f"template_edit_{tname}",
                    )
                    templates[tname] = new_val

            if st.button("Reset Templates", key="reset_templates"):
                from src.prompts.templates import PROMPT_TEMPLATES
                st.session_state.prompt_templates = dict(PROMPT_TEMPLATES)
                st.rerun()

    # Output directory
    output_dir = st.text_input(
        "Output directory",
        value="data/outputs/dashboard_run",
        key="run_output_dir",
    )

    # ─── Run controls ────────────────────────────────────────────────
    st.subheader("Run")

    # Initialize runner in session state
    if "experiment_runner" not in st.session_state or st.session_state.experiment_runner is None:
        st.session_state.experiment_runner = ExperimentRunner()

    runner = st.session_state.experiment_runner

    col_run, col_stop = st.columns(2)

    with col_run:
        run_disabled = runner.is_running or not selected_regimes or not selected_questions
        if st.button("Run Experiment", disabled=run_disabled, type="primary", key="run_btn"):
            ps = st.session_state.probe_set_primary
            gen_config = get_generation_config()
            target_layers = get_target_layers()

            runner.start(
                regimes=selected_regimes,
                questions=selected_questions,
                model=st.session_state.model,
                tokenizer=st.session_state.tokenizer,
                target_layers=target_layers,
                probe_ensembles=ps["probes"],
                scalers=ps["scalers"],
                per_position=ps["per_position"],
                generation_config=gen_config,
                concept=config["concept"],
                num_probes=config["num_probes_per_layer"],
                num_rollouts=num_rollouts,
                max_new_tokens_override=max_tokens_override,
                batch_size=batch_size,
                gpu_lock=st.session_state.gpu_lock,
                prompt_templates_override=st.session_state.prompt_templates,
                config=config,
            )
            st.rerun()

    with col_stop:
        if st.button("Stop", disabled=not runner.is_running, key="stop_btn"):
            runner.stop()
            st.info("Stop signal sent. Will halt after current batch.")

    # ─── Live results ────────────────────────────────────────────────
    if runner.is_running or runner.all_trials:
        _render_live_results(runner, output_dir)


def _render_live_results(runner, output_dir):
    """Render live progress and results from the experiment runner."""
    target_layers = get_target_layers()

    # Drain queue
    new_results = runner.get_results()

    # Progress
    if runner.is_running:
        progress = runner.completed_trials / runner.total_trials if runner.total_trials > 0 else 0
        eta = runner.eta_seconds
        eta_str = f" — ETA: {eta/60:.1f}m" if eta else ""
        st.progress(progress, text=f"Regime: {runner.current_regime} | "
                    f"{runner.completed_trials}/{runner.total_trials} trials{eta_str}")

        # Auto-refresh
        time.sleep(0.5)
        st.rerun()

    if runner.error:
        st.error(f"Experiment error: {runner.error}")

    if not runner.all_trials:
        return

    # Show status
    if not runner.is_running:
        st.success(f"Completed: {len(runner.all_trials)} trials")

    # Group trials by regime
    trials_by_regime = {}
    for trial in runner.all_trials:
        regime = trial.get("regime", "unknown")
        trials_by_regime.setdefault(regime, []).append(trial)

    regimes = list(trials_by_regime.keys())

    # Aggregate
    from run_evasion_experiment import aggregate_results
    summary = aggregate_results(runner.all_trials, target_layers)

    # Determine positions
    first_trial = runner.all_trials[0]
    pr = get_trial_probe_results(first_trial)
    all_positions = sorted(pr.keys())

    position = st.selectbox(
        "Position",
        all_positions,
        index=all_positions.index("answer_mean_pool") if "answer_mean_pool" in all_positions else 0,
        key="run_results_position",
    )

    # Bar chart
    st.subheader("Mean Confidence by Regime")
    fig_bar = regime_comparison_bars(summary, all_positions, regimes, target_layers)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    st.subheader("Regime x Layer Heatmap")
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

    # Weighted combo scoring
    combo = st.session_state.get("weighted_combo")
    if combo is not None:
        from src.probes.evaluate import score_weighted_combination
        for trial in runner.all_trials:
            if "weighted_score" not in trial:
                p = get_trial_probe_results(trial)
                if p:
                    trial["weighted_score"] = score_weighted_combination(p, combo)

        st.subheader("Weighted Combination Scores")
        import plotly.graph_objects as go
        fig_combo = go.Figure()
        for regime in regimes:
            scores = [t.get("weighted_score", {}).get("combined_score", 0)
                      for t in trials_by_regime[regime]
                      if t.get("weighted_score")]
            if scores:
                fig_combo.add_trace(go.Box(y=scores, name=regime))
        fig_combo.update_layout(
            yaxis=dict(range=[0, 1]), yaxis_title="Combined Score",
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig_combo, use_container_width=True)

    # Save button
    if not runner.is_running:
        abs_output = resolve_path(output_dir)
        if st.button("Save Results", key="save_results_btn"):
            saved = runner.save_results(abs_output, target_layers)
            st.success(f"Saved to {saved}")
