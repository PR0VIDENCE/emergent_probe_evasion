"""Probe Evasion Dashboard — entry point, tab routing, session state init."""

import streamlit as st

st.set_page_config(
    page_title="Probe Evasion Dashboard",
    page_icon=":microscope:",
    layout="wide",
)

from state import init_session_state, load_model, load_experiment_config, load_probes, load_weighted_combo, model_is_loaded, probes_are_loaded, get_gpu_info, resolve_path

# Initialize session state
init_session_state()

# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Probe Evasion")

    # Experiment config
    st.subheader("Config")
    config_path = st.text_input(
        "Experiment config",
        value="configs/experiments/evasion_affordances.yaml",
        key="config_path_input",
    )
    if st.button("Load Config"):
        load_experiment_config(config_path)
        st.rerun()

    if st.session_state.experiment_config is not None:
        st.success("Config loaded")
    else:
        st.info("Load config to begin")

    st.divider()

    # Model status
    st.subheader("Model")
    if model_is_loaded():
        mc = st.session_state.model_config
        st.success(f"Loaded: {mc['model_id']}")
        gpu = get_gpu_info()
        if gpu:
            st.caption(f"VRAM: {gpu['used_gb']:.1f} / {gpu['total_gb']:.1f} GB")
    else:
        st.warning("Model not loaded")
        if st.session_state.experiment_config is not None:
            if st.button("Load Model"):
                load_model()
                st.rerun()

    st.divider()

    # Probe set
    st.subheader("Probes")
    probe_dir = st.text_input(
        "Probe directory",
        value="/workspace/probe_data/probes",
        key="probe_dir_input",
    )
    if st.button("Load Probes"):
        load_probes(probe_dir, key="probe_set_primary")
        st.rerun()

    if probes_are_loaded("probe_set_primary"):
        ps = st.session_state.probe_set_primary
        n_layers = len(ps["layers"])
        pos_label = "per-position" if ps["per_position"] else "shared"
        st.success(f"{n_layers} layers, {pos_label}")
        st.caption(f"Path: {ps['path']}")
    else:
        st.info("No probes loaded")

    st.divider()

    # Weighted combination
    st.subheader("Weighted Combo")
    combo_path = st.text_input(
        "Combo config",
        value="configs/probes/weighted_combination.yaml",
        key="combo_path_input",
    )
    if st.button("Load Combo"):
        load_weighted_combo(combo_path)
        st.rerun()

    if st.session_state.weighted_combo is not None:
        combo = st.session_state.weighted_combo
        st.success(f"{len(combo['weights'])}-dim, {len(combo['operating_points'])} ops")
    else:
        st.info("No combo loaded")


# ─── Tab Routing ─────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Run Experiments",
    "Compare Probes",
    "FP Analysis",
    "Prefill Workbench",
    "Results Browser",
])

with tab1:
    from tabs.run_experiments import render as render_run_experiments
    render_run_experiments()

with tab2:
    from tabs.compare_probes import render as render_compare_probes
    render_compare_probes()

with tab3:
    from tabs.fp_analysis import render as render_fp_analysis
    render_fp_analysis()

with tab4:
    from tabs.prefill_workbench import render as render_prefill_workbench
    render_prefill_workbench()

with tab5:
    from tabs.results_browser import render as render_results_browser
    render_results_browser()
