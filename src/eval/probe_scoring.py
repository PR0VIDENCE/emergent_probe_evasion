"""
Canonical probe score (§2.3).

A probe bundle lives under <probes_dir>/<position>/ as:
    layer{N}_seed{S}.pt   # LinearProbe state_dict ({linear.weight, linear.bias})
    layer{N}_scaler.pt    # {scaler_mean, scaler_scale}
plus combiner.json describing {position, layers, seeds}.

The canonical score for a rollout is a single scalar:
  for each (layer, seed) at the canonical position, standardize x with the
  saved scaler and compute the pre-sigmoid logit w·x + b; average across all
  layers and seeds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def load_probe_bundle(probes_dir, position: str) -> dict:
    """Load every (layer, seed) probe + per-layer scaler at one position.

    Returns {"position", "layers", "seeds", "W"{(layer,seed)->(w,b)},
             "scalers"{layer->(mean,scale)}}.
    """
    pos_dir = Path(probes_dir) / position
    W = {}
    layers = set()
    seeds = set()
    for pt in sorted(pos_dir.glob("layer*_seed*.pt")):
        m = re.match(r"layer(\d+)_seed(\d+)\.pt", pt.name)
        if not m:
            continue
        layer, seed = int(m.group(1)), int(m.group(2))
        sd = torch.load(pt, weights_only=True, map_location="cpu")
        w = sd["linear.weight"].float().numpy().reshape(-1)
        b = float(sd["linear.bias"].float().numpy().reshape(-1)[0])
        W[(layer, seed)] = (w, b)
        layers.add(layer)
        seeds.add(seed)

    scalers = {}
    for pt in sorted(pos_dir.glob("layer*_scaler.pt")):
        m = re.match(r"layer(\d+)_scaler\.pt", pt.name)
        if not m:
            continue
        layer = int(m.group(1))
        sd = torch.load(pt, weights_only=True, map_location="cpu")
        mean = sd["scaler_mean"]
        scale = sd["scaler_scale"]
        mean = mean.float().numpy() if mean is not None else None
        scale = scale.float().numpy() if scale is not None else None
        scalers[layer] = (mean, scale)

    return {
        "position": position,
        "layers": sorted(layers),
        "seeds": sorted(seeds),
        "W": W,
        "scalers": scalers,
    }


def score_activation(bundle: dict, act: dict) -> Optional[float]:
    """Compute the canonical combined logit for one rollout's activation dict.

    `act` is {position: {layer: tensor}}.  Returns None if the canonical
    position / required layers are missing.
    """
    position = bundle["position"]
    if position not in act:
        return None
    pos_act = act[position]
    logits = []
    for (layer, seed), (w, b) in bundle["W"].items():
        if layer not in pos_act:
            continue
        x = pos_act[layer]
        x = x.float().numpy() if hasattr(x, "float") else np.asarray(x, dtype=np.float32)
        mean, scale = bundle["scalers"].get(layer, (None, None))
        if mean is not None and scale is not None:
            x = (x - mean) / scale
        logits.append(float(np.dot(w, x) + b))
    if not logits:
        return None
    return float(np.mean(logits))


def score_matrix(bundle: dict, acts: List[dict]) -> np.ndarray:
    """Vectorized convenience: returns np.array of scores (NaN where missing)."""
    out = []
    for a in acts:
        s = score_activation(bundle, a)
        out.append(np.nan if s is None else s)
    return np.array(out, dtype=np.float64)


def write_combiner(probes_dir, position: str, layers: List[int], seeds: List[int]):
    payload = {
        "method": "mean_logit",
        "position": position,
        "layers": layers,
        "seeds": seeds,
        "note": "canonical score = mean over (layer, seed) of standardized w·x + b",
    }
    Path(probes_dir).mkdir(parents=True, exist_ok=True)
    (Path(probes_dir) / "combiner.json").write_text(json.dumps(payload, indent=2))
