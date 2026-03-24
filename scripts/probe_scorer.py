"""
Interactive Probe Scorer — Streamlit app for manual prompt/reasoning/answer
editing with immediate probe feedback.

Run: streamlit run scripts/probe_scorer.py --server.port 8501 --server.address 0.0.0.0

Lets you type prompt, reasoning, and answer text, then runs a forward pass
through the model and scores probes at multiple positions. Supports:
- Loading scenario YAML files and selecting pairs (deceptive/honest)
- System message field for accurate chat template formatting
- Caching a baseline result and showing deltas on subsequent runs
"""

import os
import re
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.prompts.templates import PROMPT_TEMPLATES

# ─── Module-level cache (survives Streamlit reruns) ───────────────────
_cache = {
    "model": None,
    "tokenizer": None,
    "model_config": None,
    "experiment_config": None,
    "probe_ensembles": None,
    "scalers": None,
    "per_position": None,
    "target_layers": None,
}


def resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_layers(probe_dir: str, positions: list = None) -> list:
    """Auto-discover available layers from probe directory."""
    search_dir = os.path.join(probe_dir, positions[0]) if positions else probe_dir
    layers = set()
    if os.path.isdir(search_dir):
        for fname in os.listdir(search_dir):
            m = re.match(r"layer(\d+)_seed\d+\.pt", fname)
            if m:
                layers.add(int(m.group(1)))
    return sorted(layers)


def confidence_color(v: float) -> str:
    """Map confidence 0-1 to green-yellow-red hex color."""
    if v <= 0.5:
        t = v / 0.5
        r = int(26 + t * (255 - 26))
        g = int(150 + t * (255 - 150))
        b = int(65 + t * (191 - 65))
    else:
        t = (v - 0.5) / 0.5
        r = int(255 - t * (255 - 215))
        g = int(255 - t * (255 - 25))
        b = int(191 - t * (191 - 28))
    return f"#{r:02x}{g:02x}{b:02x}"


def find_scenario_files() -> list:
    """Find all scenario YAML files under data/concepts/deception*."""
    base = PROJECT_ROOT / "data" / "concepts"
    files = []
    if base.is_dir():
        for f in sorted(base.rglob("scenarios*.yaml")):
            rel = f.relative_to(PROJECT_ROOT)
            files.append(str(rel))
    return files


def load_scenarios(path: str) -> list:
    """Load scenarios list from a YAML file."""
    data = load_yaml(resolve_path(path))
    return data.get("scenarios", [])


def assemble_prefill_with_system(
    system_text, prompt_text, thinking_trace, answer_text, tokenizer
):
    """
    Like assemble_prefill_sequence but supports a system message for
    accurate chat template formatting matching the training pipeline.
    """
    import torch

    messages = []
    if system_text and system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": prompt_text})

    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_ids = tokenizer(
        chat_prompt, return_tensors="pt", truncation=True, max_length=4096
    )["input_ids"]
    input_len = prompt_ids.shape[1]

    encode = lambda text: torch.tensor(
        [tokenizer.encode(text, add_special_tokens=False)]
    )

    if thinking_trace is not None and thinking_trace.strip():
        think_ids = encode(thinking_trace)
        close_ids = encode("\n</think>\n")
    else:
        think_ids = torch.zeros(1, 0, dtype=torch.long)
        close_ids = encode("</think>\n")

    answer_ids = encode(answer_text) if answer_text else torch.zeros(1, 0, dtype=torch.long)

    output_ids = torch.cat([prompt_ids, think_ids, close_ids, answer_ids], dim=1)

    think_end_pos = input_len + think_ids.shape[1]
    close_end_pos = think_end_pos + close_ids.shape[1]
    seq_len = output_ids.shape[1]

    end_of_reasoning = think_end_pos + close_ids.shape[1] - 1
    answer_start = close_end_pos
    answer_end = seq_len - 1

    first_answer_sentence_end = answer_end
    if answer_text:
        sent_match = re.search(r'[.!?]', answer_text)
        if sent_match:
            text_to_sent = answer_text[:sent_match.end()]
            sent_tokens = tokenizer.encode(text_to_sent, add_special_tokens=False)
            offset = len(sent_tokens) - 1
            candidate = answer_start + offset
            if candidate <= answer_end:
                first_answer_sentence_end = candidate

    positions = {
        "last_token": seq_len - 1,
        "end_of_reasoning": min(end_of_reasoning, seq_len - 1),
        "first_answer_sentence_end": first_answer_sentence_end,
        "answer_start": min(answer_start, seq_len - 1),
        "answer_end": answer_end,
    }

    return output_ids, input_len, positions


# ─── Streamlit App ────────────────────────────────────────────────────

st.set_page_config(page_title="Probe Scorer", layout="wide")
st.title("Interactive Probe Scorer")

# Initialize session state
for key in ["baseline", "last_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Sidebar: Config & Loading ────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    config_path = st.text_input(
        "Experiment config",
        value="configs/experiments/deception_prewritten.yaml",
    )

    # Auto-derive default probe dir from config
    _default_probe_dir = "/workspace/deception_prewritten_data/probes"
    try:
        _cfg = load_yaml(resolve_path(config_path))
        _base = _cfg.get("storage", {}).get("base_dir", "")
        if _base:
            _default_probe_dir = os.path.join(_base, "probes")
    except Exception:
        pass

    probe_dir_input = st.text_input(
        "Probe directory",
        value=_default_probe_dir,
    )

    if st.button("Load Model + Probes", type="primary"):
        try:
            full_config_path = resolve_path(config_path)
            config = load_yaml(full_config_path)
            model_config = load_yaml(resolve_path(config["model_config"]))

            _cache["experiment_config"] = config
            _cache["model_config"] = model_config

            probe_dir = resolve_path(probe_dir_input)
            probe_positions = config.get("probe_positions",
                                         config.get("token_positions", None))
            num_probes = config.get(
                "num_probes_per_layer",
                len(config.get("probe_training", {}).get(
                    "random_seeds", [42, 123, 456, 789])))
            hidden_dim = model_config["hidden_dim"]

            target_layers = discover_layers(probe_dir, probe_positions)
            if not target_layers:
                st.error(f"No probe files found in {probe_dir}")
                st.stop()
            _cache["target_layers"] = target_layers

            with st.spinner(f"Loading probes ({len(target_layers)} layers)..."):
                from run_evasion_experiment import load_probe_ensembles
                loaded = load_probe_ensembles(
                    probe_dir, target_layers, num_probes, hidden_dim,
                    positions=probe_positions,
                )
                _cache["probe_ensembles"] = loaded["probes"]
                _cache["scalers"] = loaded["scalers"]
                _cache["per_position"] = loaded["per_position"]

            with st.spinner(f"Loading {model_config['model_id']}..."):
                from src.inference.extract_activations import load_model_and_tokenizer
                model, tokenizer = load_model_and_tokenizer(model_config)
                _cache["model"] = model
                _cache["tokenizer"] = tokenizer

            st.success("Model + probes loaded!")
        except Exception as e:
            st.error(f"Load failed: {e}")

    # Status
    st.divider()
    st.subheader("Status")

    if _cache["model"] is not None:
        mc = _cache["model_config"]
        st.markdown(f"**Model**: {mc['model_id']}")
        st.markdown(f"**Layers**: {len(_cache['target_layers'])} "
                    f"({_cache['target_layers']})")
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                used = (mem[1] - mem[0]) / 1e9
                total = mem[1] / 1e9
                st.markdown(f"**VRAM**: {used:.1f} / {total:.1f} GB")
        except Exception:
            pass
    else:
        st.info("Model not loaded")

# ─── Main Area ────────────────────────────────────────────────────────

# ── Scenario loader ──────────────────────────────────────────────────
st.subheader("Scenario Loader")

scenario_files = find_scenario_files()
file_labels = ["(none — manual input)"] + scenario_files
selected_file = st.selectbox("Scenario file", file_labels)

scenarios = []
if selected_file != "(none — manual input)":
    try:
        scenarios = load_scenarios(selected_file)
    except Exception as e:
        st.error(f"Failed to load scenarios: {e}")

# Pair selector + deceptive/honest toggle
selected_scenario = None
if scenarios:
    pair_labels = []
    for s in scenarios:
        pid = s.get("pair_id", "?")
        cat = s.get("category", "")
        q_preview = s.get("question", "")[:60]
        pair_labels.append(f"#{pid} [{cat}] {q_preview}")

    sel_col, mode_col = st.columns([3, 1])
    with sel_col:
        selected_pair_label = st.selectbox("Select pair", pair_labels)
    with mode_col:
        completion_mode = st.radio("Completion", ["deceptive", "honest"],
                                   horizontal=True)

    pair_idx = pair_labels.index(selected_pair_label)
    selected_scenario = scenarios[pair_idx]

# ── Text input areas ─────────────────────────────────────────────────
st.divider()

# Determine defaults from selected scenario
default_system = ""
default_question = ""
default_reasoning = ""
default_answer = ""

if selected_scenario is not None:
    default_system = selected_scenario.get("scenario", "")
    default_question = selected_scenario.get("question", "")
    prefix = selected_scenario.get("answer_prefix", "")
    if completion_mode == "deceptive":
        completion = selected_scenario.get("deceptive_completion", "")
    else:
        completion = selected_scenario.get("honest_completion", "")
    default_answer = f"{prefix} {completion}".strip() if prefix else completion

# System message
system_text = st.text_area(
    "System message (scenario context)",
    value=default_system, height=100,
    help="Sent as the system role in the chat template, matching the training pipeline format.",
)

# Three columns: question, reasoning, answer
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**User message** (question)")
    prompt_text = st.text_area(
        "prompt_input", value=default_question, height=250,
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**Reasoning** (optional `<think>` block)")
    reasoning_text = st.text_area(
        "reasoning_input", value=default_reasoning, height=250,
        label_visibility="collapsed",
    )

with col3:
    st.markdown("**Answer** (assistant completion)")
    answer_text = st.text_area(
        "answer_input", value=default_answer, height=250,
        label_visibility="collapsed",
    )

# ── Action buttons ───────────────────────────────────────────────────
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
with btn_col1:
    score_clicked = st.button("Score", type="primary",
                              disabled=_cache["model"] is None)
with btn_col2:
    cache_clicked = st.button("Cache as baseline",
                              disabled=st.session_state.last_result is None)

if cache_clicked and st.session_state.last_result is not None:
    st.session_state.baseline = st.session_state.last_result
    st.toast("Baseline cached!")

if score_clicked:
    if _cache["model"] is None:
        st.error("Load model + probes first.")
        st.stop()

    if not prompt_text.strip() and not answer_text.strip():
        st.warning("Enter at least a question or answer.")
        st.stop()

    from prefill_ablation import run_prefill_probe

    tokenizer = _cache["tokenizer"]
    model = _cache["model"]
    target_layers = _cache["target_layers"]
    probe_ensembles = _cache["probe_ensembles"]
    scalers = _cache["scalers"]
    per_position = _cache["per_position"]

    reasoning_arg = reasoning_text.strip() if reasoning_text.strip() else None

    with st.spinner("Running forward pass..."):
        t0 = time.time()
        output_ids, input_len, positions = assemble_prefill_with_system(
            system_text, prompt_text, reasoning_arg, answer_text, tokenizer,
        )

        result = run_prefill_probe(
            output_ids, positions, model,
            target_layers, probe_ensembles, scalers, per_position,
        )
        elapsed = time.time() - t0

    seq_len = output_ids.shape[1]
    think_tokens = 0
    if reasoning_arg:
        think_tokens = len(tokenizer.encode(reasoning_arg, add_special_tokens=False))
    answer_tokens = 0
    if answer_text.strip():
        answer_tokens = len(tokenizer.encode(answer_text, add_special_tokens=False))

    result["_meta"] = {
        "seq_len": seq_len,
        "prompt_tokens": input_len,
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "elapsed": elapsed,
    }

    st.session_state.last_result = result

# ─── Display Results ──────────────────────────────────────────────────

result = st.session_state.last_result
if result is not None:
    meta = result["_meta"]
    probe_results = result["probe_results"]
    target_layers = _cache["target_layers"]

    st.divider()

    st.markdown(
        f"**Sequence**: {meta['seq_len']} tokens "
        f"(prompt={meta['prompt_tokens']}, "
        f"think={meta['think_tokens']}, "
        f"answer={meta['answer_tokens']}) "
        f"— forward pass {meta['elapsed']:.2f}s"
    )

    display_positions = ["answer_mean_pool", "last_token",
                         "end_of_reasoning", "first_answer_sentence_end"]
    available_positions = [p for p in display_positions if p in probe_results]

    baseline = st.session_state.baseline

    for pos in available_positions:
        pos_data = probe_results.get(pos, {})
        if not pos_data:
            continue

        confs = [pos_data[l]["mean_confidence"] for l in target_layers if l in pos_data]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        color = confidence_color(mean_conf)

        delta_str = ""
        if baseline is not None:
            bl_pos = baseline["probe_results"].get(pos, {})
            bl_confs = [bl_pos[l]["mean_confidence"] for l in target_layers if l in bl_pos]
            bl_mean = sum(bl_confs) / len(bl_confs) if bl_confs else 0.0
            delta = mean_conf - bl_mean
            sign = "+" if delta >= 0 else ""
            delta_color = "#4caf50" if delta < 0 else "#f44336"
            delta_str = (f' <span style="color:{delta_color};font-size:13px;">'
                         f'({sign}{delta:.3f})</span>')

        st.markdown(
            f'**{pos}** — '
            f'<span style="color:{color};font-weight:bold;">{mean_conf:.3f}</span>'
            f'{delta_str}',
            unsafe_allow_html=True,
        )

        for layer_idx in target_layers:
            layer_data = pos_data.get(layer_idx, {})
            if not layer_data:
                continue
            conf = layer_data["mean_confidence"]
            bar_color = confidence_color(conf)
            pct = int(conf * 100)

            layer_delta = ""
            if baseline is not None:
                bl_layer = baseline["probe_results"].get(pos, {}).get(layer_idx, {})
                if bl_layer:
                    bl_conf = bl_layer["mean_confidence"]
                    d = conf - bl_conf
                    sign = "+" if d >= 0 else ""
                    d_color = "#4caf50" if d < 0 else "#f44336"
                    layer_delta = (f'<span style="color:{d_color};font-size:10px;'
                                   f'margin-left:4px;">({sign}{d:.3f})</span>')

            html = (
                f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="width:50px;font-size:11px;color:#aaa;">L{layer_idx}</span>'
                f'<div style="flex:1;background:#333;border-radius:4px;height:16px;'
                f'position:relative;">'
                f'<div style="width:{pct}%;background:{bar_color};height:100%;'
                f'border-radius:4px;"></div>'
                f'</div>'
                f'<span style="width:55px;text-align:right;font-size:11px;'
                f'color:{bar_color};margin-left:4px;">{conf:.3f}</span>'
                f'{layer_delta}'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

        st.markdown("")  # spacing
