"""
Stage 4: train Apollo-style probes + validate against the 0.95 val-AUROC gate.

Trains a per-(layer, seed) LinearProbe ensemble at the concept's canonical
position over all target layers, saves them in the canonical on-disk format
(§2.3), then computes the canonical combined score (mean logit across all
layers + seeds) on the validation split and writes validation.json with
pass/fail.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import common, probe_scoring
from .judging import resolve_probe_label
from src.probes.train import train_probe_ensemble
from src.probes.evaluate import evaluate_ensemble


def _unit_id(rollout: dict) -> str:
    md = rollout.get("metadata", {})
    return md.get("unit_id") or rollout["contrastive_id"]


def _split_units(units: List[str], seed: int,
                 ratios=(0.7, 0.15, 0.15)) -> Dict[str, set]:
    rng = random.Random(seed)
    u = sorted(set(units))
    rng.shuffle(u)
    n = len(u)
    n_tr = int(n * ratios[0])
    n_va = max(1, int(n * ratios[1])) if n > 2 else 0
    return {
        "train": set(u[:n_tr]),
        "val": set(u[n_tr:n_tr + n_va]),
        "test": set(u[n_tr + n_va:]),
    }


def _load_pos_layer_matrix(rollouts: List[dict], acts_dir: Path, position: str,
                           layers: List[int]):
    """Return {layer: tensor(n, hidden)}, labels tensor, kept rollout list."""
    collected = {l: [] for l in layers}
    labels = []
    kept = []
    for r in rollouts:
        pt = acts_dir / f"{r['rollout_id']}.pt"
        if not pt.exists():
            continue
        act = torch.load(pt, weights_only=False, map_location="cpu")
        if position not in act:
            continue
        pos_act = act[position]
        if not all(l in pos_act for l in layers):
            continue
        for l in layers:
            collected[l].append(pos_act[l].float())
        labels.append(r["_probe_label"])
        kept.append(r)
    out = {l: (torch.stack(v) if v else None) for l, v in collected.items()}
    return out, torch.tensor(labels, dtype=torch.float32), kept


def train_and_validate(config: dict, concept: str) -> dict:
    cc = config["_concept_configs"][concept]
    paths = common.stage_paths(config, concept)
    layers = common.target_layers(config)
    position = cc["probe"].get("canonical_position", "answer_mean_pool")
    threshold = float(cc["probe"].get("validation_threshold", 0.95))
    seed = cc["probe"].get("seed", 42)

    rollouts = common.read_jsonl(paths["pt_labeled"])
    # Attach probe labels and keep only included (positive/negative) rollouts.
    included = []
    for r in rollouts:
        pl = resolve_probe_label(r, cc)
        if pl in (0, 1):
            r["_probe_label"] = pl
            included.append(r)

    result = {
        "concept": concept, "position": position, "threshold": threshold,
        "n_included": len(included),
    }
    if len(included) < 8:
        result.update({"passed": False, "val_auroc": None,
                       "reason": f"too few labeled rollouts ({len(included)})"})
        _write_validation(paths, result)
        return result

    # Split by unit to prevent contrastive leakage.
    splits = _split_units([_unit_id(r) for r in included], seed)
    by_split = {s: [r for r in included if _unit_id(r) in ids] for s, ids in splits.items()}
    # Fall back: if val empty, borrow from train tail.
    if not by_split["val"]:
        by_split["val"] = by_split["train"][-max(2, len(by_split["train"]) // 10):]

    acts_dir = paths["pt_acts"]
    train_acts, train_y, _ = _load_pos_layer_matrix(by_split["train"], acts_dir, position, layers)
    val_acts, val_y, _ = _load_pos_layer_matrix(by_split["val"], acts_dir, position, layers)

    if len(train_y) < 4 or len(set(train_y.tolist())) < 2:
        result.update({"passed": False, "val_auroc": None,
                       "reason": "train split lacks both classes"})
        _write_validation(paths, result)
        return result

    probe_cfg = config.get("probe_training", {})
    seeds = probe_cfg.get("random_seeds", [42, 123, 456, 789])
    probes_dir = paths["probes_dir"]
    pos_dir = probes_dir / position
    pos_dir.mkdir(parents=True, exist_ok=True)

    per_layer_val_auroc = {}
    trained_layers = []
    for layer in layers:
        if train_acts.get(layer) is None:
            continue
        train_config = {
            "random_seeds": seeds,
            "learning_rate": probe_cfg.get("learning_rate", 1e-3),
            "num_epochs": probe_cfg.get("num_epochs", 100),
            "weight_decay": probe_cfg.get("weight_decay", 0.1),
            "patience": probe_cfg.get("patience", 10),
            "normalize": probe_cfg.get("normalize", True),
            "val_activations": val_acts.get(layer),
            "val_labels": val_y if val_acts.get(layer) is not None else None,
        }
        ens = train_probe_ensemble(train_acts[layer], train_y, train_config)
        for probe, s in zip(ens["probes"], seeds):
            torch.save(probe.state_dict(), pos_dir / f"layer{layer}_seed{s}.pt")
        if ens["scaler_mean"] is not None:
            torch.save({"scaler_mean": ens["scaler_mean"], "scaler_scale": ens["scaler_scale"]},
                       pos_dir / f"layer{layer}_scaler.pt")
        trained_layers.append(layer)
        if val_acts.get(layer) is not None and len(set(val_y.tolist())) > 1:
            ev = evaluate_ensemble(ens["probes"], val_acts[layer], val_y,
                                   scaler_mean=ens["scaler_mean"], scaler_scale=ens["scaler_scale"])
            per_layer_val_auroc[layer] = ev.get("ensemble_auc_roc")

    probe_scoring.write_combiner(probes_dir, position, trained_layers, list(seeds))

    # Canonical combined score on the validation split.
    bundle = probe_scoring.load_probe_bundle(probes_dir, position)
    val_scores, val_labels = [], []
    for r in by_split["val"]:
        pt = acts_dir / f"{r['rollout_id']}.pt"
        if not pt.exists():
            continue
        act = torch.load(pt, weights_only=False, map_location="cpu")
        s = probe_scoring.score_activation(bundle, act)
        if s is not None:
            val_scores.append(s)
            val_labels.append(r["_probe_label"])

    from sklearn.metrics import roc_auc_score
    val_auroc = float(roc_auc_score(val_labels, val_scores)) if len(set(val_labels)) > 1 else None

    # Held-out (unit-disjoint) test split = our clean generalization / OOD number.
    # The test split is never seen during training or early stopping.
    ood_threshold = float(cc["probe"].get("ood_validation_threshold", 0.8))
    test_scores, test_labels = [], []
    for r in by_split["test"]:
        pt = acts_dir / f"{r['rollout_id']}.pt"
        if not pt.exists():
            continue
        act = torch.load(pt, weights_only=False, map_location="cpu")
        s = probe_scoring.score_activation(bundle, act)
        if s is not None:
            test_scores.append(s)
            test_labels.append(r["_probe_label"])
    test_auroc = (float(roc_auc_score(test_labels, test_scores))
                  if len(set(test_labels)) > 1 else None)

    val_ok = val_auroc is not None and val_auroc >= threshold
    # OOD gate only blocks when measurable (≥2 test units with both classes).
    ood_ok = test_auroc is None or test_auroc >= ood_threshold
    passed = val_ok and ood_ok

    if not val_ok:
        reason = ("val AUROC below threshold" if val_auroc is not None else "val AUROC undefined")
    elif not ood_ok:
        reason = f"OOD (test) AUROC {round(test_auroc, 4)} below {ood_threshold}"
    else:
        reason = "passed"

    result.update({
        "passed": passed,
        "val_auroc": round(val_auroc, 4) if val_auroc is not None else None,
        "test_auroc": round(test_auroc, 4) if test_auroc is not None else None,
        "ood_threshold": ood_threshold,
        "n_train": len(by_split["train"]), "n_val": len(val_labels),
        "n_test": len(test_labels),
        "trained_layers": trained_layers,
        "per_layer_val_auroc": {str(k): (round(v, 4) if v is not None else None)
                                for k, v in per_layer_val_auroc.items()},
        "reason": reason,
    })
    _write_validation(paths, result)

    # Also write results.json (per-layer detail for the dashboard).
    paths["probe_results"].write_text(json.dumps({
        "position": position,
        "per_layer_val_auroc": result["per_layer_val_auroc"],
        "canonical_val_auroc": result["val_auroc"],
        "canonical_test_auroc": result["test_auroc"],
    }, indent=2))
    return result


def _write_validation(paths, result: dict):
    paths["validation"].parent.mkdir(parents=True, exist_ok=True)
    paths["validation"].write_text(json.dumps(result, indent=2))
