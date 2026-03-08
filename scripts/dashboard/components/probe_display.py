"""Probe score display components: colored bars, confidence coloring."""

import streamlit as st


def confidence_color(v: float) -> str:
    """Map confidence 0-1 to green-yellow-red hex color."""
    if v <= 0.5:
        # Green to yellow
        t = v / 0.5
        r = int(26 + t * (255 - 26))
        g = int(150 + t * (255 - 150))
        b = int(65 + t * (191 - 65))
    else:
        # Yellow to red
        t = (v - 0.5) / 0.5
        r = int(255 - t * (255 - 215))
        g = int(255 - t * (255 - 25))
        b = int(191 - t * (191 - 28))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_probe_scores(probe_results: dict, positions: list, layers: list,
                        show_layers: list = None):
    """
    Render colored horizontal bars for probe scores per position/layer.

    Args:
        probe_results: {position: {layer: {mean_confidence: float, ...}}}
        positions: position names to display
        layers: all layer indices
        show_layers: subset of layers to show (default: all)
    """
    display_layers = show_layers or layers

    for pos in positions:
        pos_data = probe_results.get(pos, {})
        if not pos_data:
            continue

        st.markdown(f"**{pos}**")
        for layer_idx in display_layers:
            layer_data = pos_data.get(layer_idx, pos_data.get(str(layer_idx), {}))
            if not layer_data:
                continue
            conf = layer_data.get("mean_confidence", 0)
            color = confidence_color(conf)
            pct = int(conf * 100)
            html = (
                f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="width:50px;font-size:11px;color:#aaa;">L{layer_idx}</span>'
                f'<div style="flex:1;background:#333;border-radius:4px;height:16px;position:relative;">'
                f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>'
                f'</div>'
                f'<span style="width:55px;text-align:right;font-size:11px;color:{color};margin-left:4px;">'
                f'{conf:.3f}</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)


def render_probe_scores_compact(probe_results: dict, position: str,
                                layers: list) -> str:
    """Return HTML string for a compact single-position probe display."""
    pos_data = probe_results.get(position, {})
    if not pos_data:
        return "<em>No data</em>"

    parts = []
    for layer_idx in layers:
        layer_data = pos_data.get(layer_idx, pos_data.get(str(layer_idx), {}))
        if not layer_data:
            continue
        conf = layer_data.get("mean_confidence", 0)
        color = confidence_color(conf)
        parts.append(f'<span style="color:{color};font-weight:bold;">'
                     f'L{layer_idx}={conf:.2f}</span>')

    return " | ".join(parts) if parts else "<em>No data</em>"
