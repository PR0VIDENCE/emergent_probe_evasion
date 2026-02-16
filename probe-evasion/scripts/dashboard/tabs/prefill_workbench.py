"""Tab 4: Prefill Workbench — assemble synthetic sequences and score probes."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from state import (
    model_is_loaded, probes_are_loaded, get_target_layers,
    get_all_regimes, get_questions, resolve_path,
)
from components.data_loader import (
    load_trials, build_trial_index, get_unique_questions, get_unique_rollouts,
    get_trial_probe_results,
)
from components.probe_display import render_probe_scores, confidence_color
from components.charts import probe_heatmap, DARK_LAYOUT

from prefill_ablation import (
    SWAP_CONDITIONS, ABLATION_CONDITIONS, MINIMAL_PROMPT,
    assemble_prefill_sequence, run_prefill_probe, build_prompt_for_regime,
    find_regime_config,
)


def render():
    st.header("Prefill Workbench")

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

    target_layers = get_target_layers()
    ps = st.session_state.probe_set_primary
    all_regimes = get_all_regimes()
    regime_names = [r["name"] for r in all_regimes]

    # ─── Data loading for regime-based sources ───────────────────────
    st.subheader("Source Data")
    data_dir = st.text_input(
        "Trial data directory", value="data/outputs/evasion",
        key="prefill_data_dir",
    )
    abs_data_dir = resolve_path(data_dir)

    # Regime selectors for source data
    col_ra, col_rb = st.columns(2)
    with col_ra:
        regime_a_name = st.selectbox("Regime A (control)", regime_names,
                                      index=regime_names.index("baseline") if "baseline" in regime_names else 0,
                                      key="prefill_regime_a")
    with col_rb:
        regime_b_name = st.selectbox("Regime B (treatment)", regime_names,
                                      index=regime_names.index("technical_disclosure") if "technical_disclosure" in regime_names else 1,
                                      key="prefill_regime_b")

    # Load trial data
    trials_a = load_trials(abs_data_dir, regime_a_name)
    trials_b = load_trials(abs_data_dir, regime_b_name)

    index_a = build_trial_index(trials_a) if trials_a else {}
    index_b = build_trial_index(trials_b) if trials_b else {}

    if trials_a:
        st.caption(f"Regime A ({regime_a_name}): {len(trials_a)} trials")
    else:
        st.caption(f"Regime A ({regime_a_name}): no data loaded")
    if trials_b:
        st.caption(f"Regime B ({regime_b_name}): {len(trials_b)} trials")
    else:
        st.caption(f"Regime B ({regime_b_name}): no data loaded")

    # Question/rollout selection for "from regime" sources
    available_questions = get_unique_questions(trials_a or trials_b or [])
    if available_questions:
        src_question = st.selectbox("Question", available_questions, key="prefill_question")
        # Get rollouts available for this question in both regimes
        available_rollouts = []
        for rollout in range(10):
            if (not trials_a or (src_question, rollout) in index_a) and \
               (not trials_b or (src_question, rollout) in index_b):
                available_rollouts.append(rollout)
        if not available_rollouts:
            available_rollouts = [0]
        src_rollout = st.selectbox("Rollout", available_rollouts, key="prefill_rollout")
    else:
        src_question = None
        src_rollout = 0

    st.divider()

    # ─── Three-column source picker ──────────────────────────────────
    st.subheader("Sequence Assembly")

    col_p, col_r, col_a = st.columns(3)

    with col_p:
        st.markdown("**PROMPT**")
        prompt_source = st.radio(
            "Source", ["From Regime A", "From Regime B", "Minimal", "Custom"],
            key="prefill_prompt_src",
        )
        if prompt_source == "Custom":
            prompt_text = st.text_area("Custom prompt", value="", height=100, key="prefill_prompt_custom")
        elif prompt_source == "Minimal":
            prompt_text = MINIMAL_PROMPT
        else:
            regime_name = regime_a_name if prompt_source == "From Regime A" else regime_b_name
            regime_cfg = find_regime_config(config, regime_name)
            if regime_cfg and src_question:
                q_text = next((q["text"] for q in get_questions() if q["id"] == src_question), "")
                prompt_text = build_prompt_for_regime(
                    regime_cfg, q_text, config["concept"],
                    target_layers, config["num_probes_per_layer"],
                )
            else:
                prompt_text = ""
        st.caption(f"Length: {len(prompt_text)} chars")

    with col_r:
        st.markdown("**REASONING**")
        reasoning_source = st.radio(
            "Source", ["From Regime A", "From Regime B", "Empty", "Custom"],
            key="prefill_reasoning_src",
        )
        if reasoning_source == "Custom":
            thinking_trace = st.text_area("Custom reasoning", value="", height=100, key="prefill_reasoning_custom")
        elif reasoning_source == "Empty":
            thinking_trace = None
        else:
            idx = index_a if reasoning_source == "From Regime A" else index_b
            trial = idx.get((src_question, src_rollout))
            thinking_trace = trial.get("thinking_trace", "") if trial else ""
        if thinking_trace:
            st.caption(f"Length: {len(thinking_trace)} chars")
        else:
            st.caption("Empty (no reasoning)")

    with col_a:
        st.markdown("**ANSWER**")
        answer_source = st.radio(
            "Source", ["From Regime A", "From Regime B", "Custom"],
            key="prefill_answer_src",
        )
        if answer_source == "Custom":
            answer_text = st.text_area("Custom answer", value="", height=100, key="prefill_answer_custom")
        else:
            idx = index_a if answer_source == "From Regime A" else index_b
            trial = idx.get((src_question, src_rollout))
            answer_text = trial.get("answer", "") if trial else ""
        st.caption(f"Length: {len(answer_text)} chars")

    # ─── Preset buttons ──────────────────────────────────────────────
    st.subheader("Presets")

    def _apply_preset(cond):
        """Set radio buttons based on a condition dict."""
        prompt_map = {"A": "From Regime A", "B": "From Regime B", "minimal": "Minimal"}
        reason_map = {"A": "From Regime A", "B": "From Regime B", "none": "Empty"}
        answer_map = {"A": "From Regime A", "B": "From Regime B"}
        st.session_state.prefill_prompt_src = prompt_map.get(cond["prompt"], "Custom")
        st.session_state.prefill_reasoning_src = reason_map.get(cond["reasoning"], "Custom")
        st.session_state.prefill_answer_src = answer_map.get(cond["answer"], "Custom")

    # Swap conditions
    st.markdown("**Swap Conditions**")
    swap_cols = st.columns(4)
    for i, (name, cond) in enumerate(SWAP_CONDITIONS.items()):
        with swap_cols[i % 4]:
            if st.button(name, key=f"preset_swap_{name}"):
                _apply_preset(cond)
                st.rerun()

    # Ablation conditions
    st.markdown("**Ablation Conditions**")
    abl_cols = st.columns(3)
    for i, (name, cond) in enumerate(ABLATION_CONDITIONS.items()):
        with abl_cols[i % 3]:
            if st.button(name, key=f"preset_abl_{name}"):
                _apply_preset(cond)
                st.rerun()

    st.divider()

    # ─── Single run ──────────────────────────────────────────────────
    st.subheader("Run Forward Pass")

    if st.button("Run Forward Pass", type="primary", key="prefill_run_btn"):
        if not prompt_text or not answer_text:
            st.error("Need both prompt and answer text.")
            return

        with st.spinner("Running forward pass..."):
            with st.session_state.gpu_lock:
                output_ids, input_len = assemble_prefill_sequence(
                    prompt_text, thinking_trace, answer_text,
                    st.session_state.tokenizer,
                )
                result = run_prefill_probe(
                    output_ids, input_len,
                    st.session_state.model, st.session_state.tokenizer,
                    target_layers, ps["probes"], ps["scalers"], ps["per_position"],
                )

        st.session_state["prefill_last_result"] = result

    # Display last result
    result = st.session_state.get("prefill_last_result")
    if result:
        pr = result.get("probe_results", {})
        positions = sorted(pr.keys())

        render_probe_scores(pr, positions, target_layers,
                           show_layers=target_layers[:10] if len(target_layers) > 10 else target_layers)

        # Weighted score
        combo = st.session_state.get("weighted_combo")
        if combo is not None:
            from src.probes.evaluate import score_weighted_combination
            ws = score_weighted_combination(pr, combo)
            score = ws["combined_score"]
            color = confidence_color(score)
            st.markdown(f"**Weighted Score:** "
                        f"<span style='color:{color};font-size:18px;font-weight:bold;'>"
                        f"{score:.4f}</span>", unsafe_allow_html=True)

        # Compare to AAA baseline if cached
        baseline = st.session_state.get("prefill_baseline_result")
        if baseline:
            st.markdown("**Delta from AAA baseline:**")
            base_pr = baseline.get("probe_results", {})
            for pos in positions:
                pos_data = pr.get(pos, {})
                base_pos = base_pr.get(pos, {})
                deltas = []
                for l in target_layers[:5]:
                    curr = pos_data.get(l, pos_data.get(str(l), {})).get("mean_confidence", 0)
                    base = base_pos.get(l, base_pos.get(str(l), {})).get("mean_confidence", 0)
                    delta = curr - base
                    sign = "+" if delta > 0 else ""
                    deltas.append(f"L{l}: {sign}{delta:.3f}")
                st.caption(f"{pos}: {' | '.join(deltas)}")

        # Cache as baseline button
        if st.button("Cache as AAA baseline", key="cache_baseline_btn"):
            st.session_state["prefill_baseline_result"] = result
            st.success("Cached!")

    # ─── Batch mode ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Batch Mode")

    batch_conditions = st.multiselect(
        "Conditions to run",
        list(SWAP_CONDITIONS.keys()) + list(ABLATION_CONDITIONS.keys()),
        default=["AAA", "BBB"],
        key="prefill_batch_conditions",
    )

    if st.button("Run Batch", key="prefill_batch_btn"):
        if not src_question:
            st.error("Select a question first.")
            return

        all_conditions = {**SWAP_CONDITIONS, **ABLATION_CONDITIONS}
        regime_a_cfg = find_regime_config(config, regime_a_name)
        regime_b_cfg = find_regime_config(config, regime_b_name)
        q_text = next((q["text"] for q in get_questions() if q["id"] == src_question), "")

        batch_results = {}
        progress = st.progress(0)

        for i, cond_name in enumerate(batch_conditions):
            cond = all_conditions[cond_name]
            progress.progress((i + 1) / len(batch_conditions),
                             text=f"Running {cond_name}...")

            # Resolve sources
            if cond["prompt"] == "A":
                p = build_prompt_for_regime(regime_a_cfg, q_text, config["concept"], target_layers, config["num_probes_per_layer"])
            elif cond["prompt"] == "B":
                p = build_prompt_for_regime(regime_b_cfg, q_text, config["concept"], target_layers, config["num_probes_per_layer"])
            else:
                p = MINIMAL_PROMPT

            trial_a = index_a.get((src_question, src_rollout))
            trial_b = index_b.get((src_question, src_rollout))

            if cond["reasoning"] == "A":
                r = trial_a.get("thinking_trace", "") if trial_a else ""
            elif cond["reasoning"] == "B":
                r = trial_b.get("thinking_trace", "") if trial_b else ""
            else:
                r = None

            if cond["answer"] == "A":
                a = trial_a.get("answer", "") if trial_a else ""
            elif cond["answer"] == "B":
                a = trial_b.get("answer", "") if trial_b else ""
            else:
                a = ""

            with st.session_state.gpu_lock:
                output_ids, input_len = assemble_prefill_sequence(
                    p, r, a, st.session_state.tokenizer,
                )
                res = run_prefill_probe(
                    output_ids, input_len,
                    st.session_state.model, st.session_state.tokenizer,
                    target_layers, ps["probes"], ps["scalers"], ps["per_position"],
                )
            batch_results[cond_name] = res

        progress.empty()
        st.session_state["prefill_batch_results"] = batch_results

    # Display batch results
    batch_results = st.session_state.get("prefill_batch_results")
    if batch_results:
        st.markdown("**Batch Results**")

        # Summary table
        import pandas as pd
        rows = []
        pr_first = next(iter(batch_results.values()))["probe_results"]
        positions = sorted(pr_first.keys())

        for cond_name, res in batch_results.items():
            pr = res["probe_results"]
            row = {"condition": cond_name}
            for pos in positions:
                pos_data = pr.get(pos, {})
                confs = [pos_data.get(l, pos_data.get(str(l), {})).get("mean_confidence", 0)
                         for l in target_layers]
                confs = [c for c in confs if c is not None]
                row[pos] = np.mean(confs) if confs else 0
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.background_gradient(
                subset=[c for c in df.columns if c != "condition"],
                cmap="RdYlGn_r", vmin=0, vmax=1,
            ),
            use_container_width=True,
        )

        # Heatmap comparison
        position_sel = st.selectbox(
            "Position for heatmap", positions,
            index=positions.index("answer_mean_pool") if "answer_mean_pool" in positions else 0,
            key="prefill_batch_pos",
        )

        matrix = []
        cond_names = list(batch_results.keys())
        for cond_name in cond_names:
            pr = batch_results[cond_name]["probe_results"]
            pos_data = pr.get(position_sel, {})
            row = [pos_data.get(l, pos_data.get(str(l), {})).get("mean_confidence", 0)
                   for l in target_layers]
            matrix.append(row)

        fig = probe_heatmap(
            matrix,
            x_labels=[f"L{l}" for l in target_layers],
            y_labels=cond_names,
            title=f"Batch Comparison — {position_sel}",
        )
        st.plotly_chart(fig, use_container_width=True)
