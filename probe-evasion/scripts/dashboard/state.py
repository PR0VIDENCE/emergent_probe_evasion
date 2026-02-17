"""Session state management: model/probe/config loading and helpers.

Heavy objects (model, probes, config) are stored in module-level globals
so they survive Streamlit session resets (browser disconnect/reconnect).
st.session_state is re-populated from these globals on each new session.
"""

import os
import sys
import threading
from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ─── Module-level persistent cache ──────────────────────────────────
# These survive browser disconnects as long as the Streamlit process lives.
_cache = {
    "model": None,
    "tokenizer": None,
    "model_config": None,
    "experiment_config": None,
    "prompt_templates": None,
    "probe_set_primary": None,
    "probe_set_secondary": None,
    "weighted_combo": None,
    "gpu_lock": threading.Lock(),
}


def resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_session_state():
    """Initialize session state, restoring from module cache on reconnect."""
    defaults = {
        "model": None,
        "tokenizer": None,
        "model_config": None,
        "probe_set_primary": None,
        "probe_set_secondary": None,
        "weighted_combo": None,
        "experiment_config": None,
        "prompt_templates": None,
        "experiment_runner": None,
        "trial_cache": {},
        "gpu_lock": _cache["gpu_lock"],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            # Restore from module cache if available, else use default
            cached = _cache.get(key)
            st.session_state[key] = cached if cached is not None else val


def load_model():
    """Load the model and tokenizer into session state + module cache."""
    from src.inference.extract_activations import load_model_and_tokenizer

    config = st.session_state.experiment_config
    if config is None:
        st.error("Load experiment config first.")
        return

    model_config_path = resolve_path(config["model_config"])
    model_config = load_config(model_config_path)
    st.session_state.model_config = model_config
    _cache["model_config"] = model_config

    with st.spinner(f"Loading {model_config['model_id']}..."):
        model, tokenizer = load_model_and_tokenizer(model_config)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        _cache["model"] = model
        _cache["tokenizer"] = tokenizer


def load_experiment_config(config_path: str):
    """Load experiment config YAML."""
    path = resolve_path(config_path)
    config = load_config(path)
    st.session_state.experiment_config = config
    _cache["experiment_config"] = config

    # Also load model config
    model_config_path = resolve_path(config["model_config"])
    model_config = load_config(model_config_path)
    st.session_state.model_config = model_config
    _cache["model_config"] = model_config

    # Load prompt templates (mutable copy)
    from src.prompts.templates import PROMPT_TEMPLATES
    templates = dict(PROMPT_TEMPLATES)
    st.session_state.prompt_templates = templates
    _cache["prompt_templates"] = templates


def _discover_probe_layers(probe_dir: str, positions: list = None) -> list:
    """Scan a probe directory to find which layers have weight files."""
    import re
    search_dirs = []
    if positions:
        for pos in positions:
            pos_dir = os.path.join(probe_dir, pos)
            if os.path.isdir(pos_dir):
                search_dirs.append(pos_dir)
                break  # Only need one position dir to find layers
    if not search_dirs:
        search_dirs.append(probe_dir)

    layers = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = re.match(r"layer(\d+)_seed\d+\.pt", fname)
            if m:
                layers.add(int(m.group(1)))
    return sorted(layers)


def load_probes(probe_dir: str, key: str = "probe_set_primary"):
    """Load probe ensembles + scalers into session state + module cache."""
    from run_evasion_experiment import load_probe_ensembles

    config = st.session_state.experiment_config
    model_config = st.session_state.model_config
    if config is None or model_config is None:
        st.error("Load experiment config first.")
        return

    probe_positions = config.get("probe_positions", None)
    num_probes = config["num_probes_per_layer"]
    hidden_dim = model_config["hidden_dim"]

    # Auto-detect which layers exist on disk rather than trusting config
    discovered_layers = _discover_probe_layers(probe_dir, probe_positions)
    if not discovered_layers:
        st.error(f"No probe weight files found in {probe_dir}")
        return

    with st.spinner(f"Loading probes from {probe_dir} ({len(discovered_layers)} layers)..."):
        loaded = load_probe_ensembles(
            probe_dir, discovered_layers, num_probes, hidden_dim,
            positions=probe_positions,
        )
        probe_set = {
            "probes": loaded["probes"],
            "scalers": loaded["scalers"],
            "per_position": loaded["per_position"],
            "path": probe_dir,
            "layers": discovered_layers,
        }
        st.session_state[key] = probe_set
        _cache[key] = probe_set


def load_weighted_combo(config_path: str):
    """Load weighted combination config."""
    from src.probes.evaluate import load_weighted_combination
    path = resolve_path(config_path)
    combo = load_weighted_combination(path)
    st.session_state.weighted_combo = combo
    _cache["weighted_combo"] = combo


def get_target_layers() -> list:
    """Get target layers from the loaded probe set, falling back to config."""
    ps = st.session_state.get("probe_set_primary")
    if ps is not None and ps.get("layers"):
        return ps["layers"]
    config = st.session_state.experiment_config
    model_config = st.session_state.model_config
    if config is None or model_config is None:
        return []
    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))
    return target_layers


def get_all_regimes() -> list:
    """Get all regimes (main + control) from config."""
    config = st.session_state.experiment_config
    if config is None:
        return []
    regimes = list(config.get("regimes", []))
    regimes.extend(config.get("control_regimes", []))
    return regimes


def get_questions() -> list:
    """Get questions from config."""
    config = st.session_state.experiment_config
    if config is None:
        return []
    return config.get("questions", [])


def get_negative_questions() -> list:
    """Get negative control questions from config."""
    config = st.session_state.experiment_config
    if config is None:
        return []
    return config.get("negative_control_questions", [])


def get_generation_config() -> dict:
    """Get generation config from experiment config."""
    config = st.session_state.experiment_config
    if config is None:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    return config.get("generation", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})


def model_is_loaded() -> bool:
    return st.session_state.model is not None


def probes_are_loaded(key: str = "probe_set_primary") -> bool:
    return st.session_state.get(key) is not None


def get_gpu_info() -> dict:
    """Get GPU memory info if available."""
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info()
            return {
                "free_gb": mem[0] / 1e9,
                "total_gb": mem[1] / 1e9,
                "used_gb": (mem[1] - mem[0]) / 1e9,
            }
    except Exception:
        pass
    return None
