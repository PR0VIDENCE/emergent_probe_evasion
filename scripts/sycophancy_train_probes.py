"""
Train SyA probes on Stage 1 activations (DiffMean + LR ensemble).

Contrast: SyA-vs-(GA+pushback) (design decision 2 in the experiment YAML)
  Positives:  T3 + endorsed_incorrect (sycophantic capitulation)
  Negatives:  T4 + endorsed_correct (GA)  |  T3 + endorsed_correct (honest pushback)
  Excluded:   neither, T4 + endorsed_incorrect

Splits — "diverse-scenarios" eval strategy:
  TRAIN POOL  = rollouts where original system_prompt = sycophancy_extreme.
                This is where most SyA positives live (56 of 63 at last count).
                Split into train / val at the QID level (no qid in both — prevents
                the same question's T3 and T4 from leaking across splits).
  OOD SLICES  = each non-train system_prompt becomes an independent test set:
                  ood_neutral          — natural-ish SyA, no prompt pressure
                  ood_sycophancy_strong — milder elicitation, held-out prompt
                Tests cross-prompt transfer (the canonical leakage diagnostic).
  Per-source breakdown (trivia_qa / truthful_qa) reported alongside each slice.

Probes — per (layer, position):
  DiffMean        — direction = unit(mean_pos − mean_neg). Sample-efficient,
                    no learned params, works at our N (~50 positives).
  LR ensemble     — 4-seed sklearn LogisticRegression with class_weight='balanced'
                    and StandardScaler. Scores averaged across seeds.

Outputs (in --output-dir, default data_dir/stage1_probes/):
  results.json     — full AUROC table: {(layer, pos): {slice: {n_pos, n_neg, diffmean_auroc, lr_auroc}}}
  diffmean_directions.pt  — {(layer, pos): unit vector} for downstream scoring
  lr_probes.pkl    — {(layer, pos): [(scaler, clf)] per seed}
  summary.md       — best-layer-position table + position-AUROC diagnostic

Usage:
  uv run python scripts/sycophancy_train_probes.py --config configs/experiments/qa_probe_training_sycophancy.yaml --activations-dir data/concepts/sycophancy_qa_v2/stage1_activations
"""

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# Label assignment (the SyA-vs-(GA+pushback) contrast)
# ----------------------------------------------------------------------------

def assign_label(framing, judge_label):
    """Return 1 (SyA positive), 0 (negative), or None (excluded)."""
    if framing == "T3_user_wrong" and judge_label == "endorsed_incorrect":
        return 1  # SyA — capitulated to wrong user assertion
    if framing == "T4_user_right" and judge_label == "endorsed_correct":
        return 0  # GA — agreed with correct user
    if framing == "T3_user_wrong" and judge_label == "endorsed_correct":
        return 0  # honest pushback — disagreed with wrong user
    return None  # excluded


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------

def load_dataset(activations_dir):
    """Load rollouts: pair extraction_log entries with their .pt activation files.

    Returns a list of dicts: each with keys
      rollout_id, question_id, framing, system_prompt_id, source, label_judge,
      label (0/1), activations (dict position → {layer: torch.Tensor})
    """
    log_path = activations_dir / "extraction_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"No extraction_log at {log_path}. "
                                f"Run sycophancy_extract_activations.py first.")
    rollouts = []
    n_unlabeled = n_excluded = n_missing_act = 0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            judge = meta.get("label_judge")
            if judge is None:
                n_unlabeled += 1
                continue
            label = assign_label(meta["framing"], judge)
            if label is None:
                n_excluded += 1
                continue
            rid = meta["rollout_id"]
            act_path = activations_dir / "activations" / f"{rid}.pt"
            if not act_path.exists():
                n_missing_act += 1
                continue
            acts = torch.load(act_path, weights_only=False, map_location="cpu")
            rollouts.append({
                "rollout_id": rid,
                "question_id": meta["question_id"],
                "framing": meta["framing"],
                "system_prompt_id": meta["system_prompt_id"],
                "source": meta["source"],
                "label_judge": judge,
                "label": label,
                "activations": acts,
            })
    print(f"Loaded {len(rollouts)} labeled+contrast rollouts")
    if n_unlabeled:    print(f"  skipped {n_unlabeled} unlabeled")
    if n_excluded:     print(f"  skipped {n_excluded} excluded (neither/T4+inc)")
    if n_missing_act:  print(f"  skipped {n_missing_act} missing activation .pt")
    return rollouts


def build_features(rollouts, layer, position):
    """Stack activations at one (layer, position) into a feature matrix.

    Returns:
      X: (n, hidden_dim) float32 numpy array
      y: (n,) int numpy array
      meta_per_row: list of dicts with system_prompt_id, source, question_id
    """
    X, y, meta_rows = [], [], []
    for r in rollouts:
        acts = r["activations"]
        if position not in acts or layer not in acts[position]:
            continue
        X.append(acts[position][layer].float().numpy())
        y.append(r["label"])
        meta_rows.append({
            "system_prompt_id": r["system_prompt_id"],
            "source": r["source"],
            "question_id": r["question_id"],
            "framing": r["framing"],
        })
    return np.stack(X) if X else np.zeros((0, 0)), np.array(y), meta_rows


# ----------------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------------

def fit_diffmean(X, y):
    """Direction = unit-normalized (mean_pos − mean_neg). Returns a 1-D vector."""
    pos = X[y == 1].mean(axis=0)
    neg = X[y == 0].mean(axis=0)
    direction = pos - neg
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return direction
    return direction / norm


def score_diffmean(X, direction):
    return X @ direction


def fit_lr(X_train, y_train, seed, C=1.0):
    """Fit StandardScaler + class-balanced LogisticRegression."""
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    clf = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        C=C,
        max_iter=2000,
        random_state=seed,
    ).fit(Xs, y_train)
    return scaler, clf


def score_lr(X, scaler, clf):
    return clf.decision_function(scaler.transform(X))


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def qid_train_split(qids, train_frac, seed):
    """Group-aware shuffle. Returns set of qids assigned to train."""
    rng = random.Random(seed)
    unique = sorted(set(qids))
    rng.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    return set(unique[:n_train])


def safe_auroc(y, scores):
    """AUROC that returns None when undefined (single-class slice)."""
    y = np.asarray(y)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return None
    return float(roc_auc_score(y, scores))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--activations-dir", required=True,
                        help="Directory containing extraction_log.jsonl + activations/")
    parser.add_argument("--output-dir", default=None,
                        help="Default: <data_dir>/stage1_probes/")
    parser.add_argument("--train-pool-prompt", default="sycophancy_extreme",
                        help="Which original system_prompt_id is in-distribution training data.")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="QID-grouped val fraction of the training pool.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--lr-C", type=float, default=1.0)
    parser.add_argument("--min-train-pos", type=int, default=5,
                        help="Skip (layer, pos) cells with fewer SyA positives in train.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Show top-K (layer, position) by val AUROC in summary.")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)
    target_layers = config["target_layers"]
    positions = config["token_positions"]
    data_dir = PROJECT_ROOT / config["data_dir"]

    activations_dir = Path(args.activations_dir)
    if not activations_dir.is_absolute():
        activations_dir = PROJECT_ROOT / activations_dir

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "stage1_probes"
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = load_dataset(activations_dir)
    if not rollouts:
        raise SystemExit("No rollouts loaded — nothing to train on.")

    # Class breakdown summary
    by_class_sp = defaultdict(Counter)
    for r in rollouts:
        by_class_sp[r["system_prompt_id"]][r["label"]] += 1
    print("\nClass × system_prompt:")
    for sp in sorted(by_class_sp):
        c = by_class_sp[sp]
        print(f"  {sp:<22s}  pos={c[1]:>3d}  neg={c[0]:>4d}")

    # Build train/val/OOD masks once (same across all layer×position)
    train_pool_rollouts = [r for r in rollouts if r["system_prompt_id"] == args.train_pool_prompt]
    train_pool_qids = [r["question_id"] for r in train_pool_rollouts]
    if not train_pool_qids:
        raise SystemExit(f"No rollouts in train pool '{args.train_pool_prompt}'")
    train_qids = qid_train_split(train_pool_qids, 1 - args.val_frac, args.seed)
    print(f"\nTrain pool: {args.train_pool_prompt}  "
          f"({len(train_pool_rollouts)} rollouts, {len(set(train_pool_qids))} unique qids)")
    print(f"  qid split: {len(train_qids)} train qids, "
          f"{len(set(train_pool_qids)) - len(train_qids)} val qids")

    ood_prompts = sorted({r["system_prompt_id"] for r in rollouts
                         if r["system_prompt_id"] != args.train_pool_prompt})
    print(f"\nOOD slices: {ood_prompts}")

    results = {}  # (layer, position) -> slice -> metrics
    diffmean_directions = {}
    lr_probes = {}

    n_skipped = 0
    n_trained = 0

    for layer in target_layers:
        for position in positions:
            X, y, meta_rows = build_features(rollouts, layer, position)
            if X.shape[0] == 0:
                continue

            sps = np.array([m["system_prompt_id"] for m in meta_rows])
            srcs = np.array([m["source"] for m in meta_rows])
            qids = np.array([m["question_id"] for m in meta_rows])
            train_pool_mask = (sps == args.train_pool_prompt)
            train_mask = train_pool_mask & np.array([qid in train_qids for qid in qids])
            val_mask = train_pool_mask & ~train_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            n_train_pos = int((y_train == 1).sum())
            if n_train_pos < args.min_train_pos:
                n_skipped += 1
                continue
            n_trained += 1

            slice_results = {
                "train": {
                    "n_pos": n_train_pos,
                    "n_neg": int((y_train == 0).sum()),
                },
                "val": {
                    "n_pos": int((y_val == 1).sum()),
                    "n_neg": int((y_val == 0).sum()),
                },
            }

            # --- DiffMean ---
            direction = fit_diffmean(X_train, y_train)
            slice_results["train"]["diffmean_auroc"] = safe_auroc(y_train, score_diffmean(X_train, direction))
            slice_results["val"]["diffmean_auroc"] = safe_auroc(y_val, score_diffmean(X_val, direction))

            # --- LR ensemble ---
            ensemble_scalers_clfs = []
            ensemble_train_scores = []
            ensemble_val_scores = []
            ensemble_ood_scores = defaultdict(list)
            for seed in args.lr_seeds:
                scaler, clf = fit_lr(X_train, y_train, seed, C=args.lr_C)
                ensemble_scalers_clfs.append((scaler, clf))
                ensemble_train_scores.append(score_lr(X_train, scaler, clf))
                ensemble_val_scores.append(score_lr(X_val, scaler, clf))

            train_scores_lr = np.mean(ensemble_train_scores, axis=0)
            val_scores_lr = np.mean(ensemble_val_scores, axis=0)
            slice_results["train"]["lr_auroc"] = safe_auroc(y_train, train_scores_lr)
            slice_results["val"]["lr_auroc"] = safe_auroc(y_val, val_scores_lr)

            # --- Per-source val breakdown ---
            val_srcs = srcs[val_mask]
            for src_val in sorted(set(val_srcs)):
                src_mask = val_srcs == src_val
                slice_name = f"val_{src_val}"
                slice_results[slice_name] = {
                    "n_pos": int((y_val[src_mask] == 1).sum()),
                    "n_neg": int((y_val[src_mask] == 0).sum()),
                    "diffmean_auroc": safe_auroc(y_val[src_mask], score_diffmean(X_val[src_mask], direction)),
                    "lr_auroc": safe_auroc(y_val[src_mask], val_scores_lr[src_mask]),
                }

            # --- OOD slices ---
            for ood_sp in ood_prompts:
                ood_mask = sps == ood_sp
                if ood_mask.sum() == 0:
                    continue
                X_ood, y_ood = X[ood_mask], y[ood_mask]
                ood_diffmean_scores = score_diffmean(X_ood, direction)
                ood_lr_scores = np.mean([score_lr(X_ood, sc, cl) for sc, cl in ensemble_scalers_clfs], axis=0)
                slice_name = f"ood_{ood_sp}"
                slice_results[slice_name] = {
                    "n_pos": int((y_ood == 1).sum()),
                    "n_neg": int((y_ood == 0).sum()),
                    "diffmean_auroc": safe_auroc(y_ood, ood_diffmean_scores),
                    "lr_auroc": safe_auroc(y_ood, ood_lr_scores),
                }
                # Per-source within OOD
                ood_srcs = srcs[ood_mask]
                for src_val in sorted(set(ood_srcs)):
                    src_mask = ood_srcs == src_val
                    sub_name = f"ood_{ood_sp}_{src_val}"
                    slice_results[sub_name] = {
                        "n_pos": int((y_ood[src_mask] == 1).sum()),
                        "n_neg": int((y_ood[src_mask] == 0).sum()),
                        "diffmean_auroc": safe_auroc(y_ood[src_mask], ood_diffmean_scores[src_mask]),
                        "lr_auroc": safe_auroc(y_ood[src_mask], ood_lr_scores[src_mask]),
                    }

            # Pooled OOD (all non-extreme prompts combined)
            if len(ood_prompts) > 1:
                pooled_mask = ~train_pool_mask
                X_p, y_p = X[pooled_mask], y[pooled_mask]
                p_diffmean = score_diffmean(X_p, direction)
                p_lr = np.mean([score_lr(X_p, sc, cl) for sc, cl in ensemble_scalers_clfs], axis=0)
                slice_results["ood_pooled"] = {
                    "n_pos": int((y_p == 1).sum()),
                    "n_neg": int((y_p == 0).sum()),
                    "diffmean_auroc": safe_auroc(y_p, p_diffmean),
                    "lr_auroc": safe_auroc(y_p, p_lr),
                }

            key = f"layer{layer}_{position}"
            results[key] = {"layer": layer, "position": position, **slice_results}
            diffmean_directions[key] = direction
            lr_probes[key] = ensemble_scalers_clfs

    print(f"\nTrained {n_trained} (layer, position) cells; skipped {n_skipped} for <{args.min_train_pos} train positives")

    # Save artifacts
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(diffmean_directions, output_dir / "diffmean_directions.pt")
    with open(output_dir / "lr_probes.pkl", "wb") as f:
        pickle.dump(lr_probes, f)

    # Write a human-readable summary
    write_summary(output_dir / "summary.md", results, args, ood_prompts)

    print(f"\nResults: {output_dir / 'results.json'}")
    print(f"Summary: {output_dir / 'summary.md'}")


def write_summary(path, results, args, ood_prompts):
    """Markdown table: top-K (layer, position) by val AUROC, plus position diagnostic."""
    lines = []
    lines.append(f"# Sycophancy probe results\n")
    lines.append(f"Contrast: SyA-vs-(GA+pushback)\n")
    lines.append(f"Train pool: `{args.train_pool_prompt}` (in-distribution)\n")
    lines.append(f"OOD slices: {', '.join(f'`{p}`' for p in ood_prompts)}\n")
    lines.append("")

    # Top-K by val DiffMean AUROC
    rows = []
    for key, r in results.items():
        val = r.get("val", {})
        rows.append({
            "key": key,
            "layer": r["layer"],
            "position": r["position"],
            "val_diffmean": val.get("diffmean_auroc"),
            "val_lr": val.get("lr_auroc"),
            "train_diffmean": r.get("train", {}).get("diffmean_auroc"),
            "train_lr": r.get("train", {}).get("lr_auroc"),
        })

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    # Sort by val_diffmean (None last)
    sortable = [r for r in rows if r["val_diffmean"] is not None]
    sortable.sort(key=lambda r: r["val_diffmean"], reverse=True)
    lines.append(f"## Top {args.top_k} by val DiffMean AUROC\n")
    lines.append("| layer | position | val DiffMean | val LR | train DiffMean | train LR |")
    lines.append("|---|---|---|---|---|---|")
    for r in sortable[:args.top_k]:
        lines.append(f"| {r['layer']} | {r['position']} | {fmt(r['val_diffmean'])} | {fmt(r['val_lr'])} | {fmt(r['train_diffmean'])} | {fmt(r['train_lr'])} |")
    lines.append("")

    # Top by val LR
    sortable_lr = [r for r in rows if r["val_lr"] is not None]
    sortable_lr.sort(key=lambda r: r["val_lr"], reverse=True)
    lines.append(f"## Top {args.top_k} by val LR AUROC\n")
    lines.append("| layer | position | val LR | val DiffMean | train LR | train DiffMean |")
    lines.append("|---|---|---|---|---|---|")
    for r in sortable_lr[:args.top_k]:
        lines.append(f"| {r['layer']} | {r['position']} | {fmt(r['val_lr'])} | {fmt(r['val_diffmean'])} | {fmt(r['train_lr'])} | {fmt(r['train_diffmean'])} |")
    lines.append("")

    # Position diagnostic: best layer per position (the textual-anchor check)
    lines.append("## Position-AUROC diagnostic (textual-anchor check)\n")
    lines.append("Best val DiffMean AUROC per position. If `reasoning_mean` matches "
                 "`answer_mean_pool`, signal lives in the deliberation, not just the answer token.\n")
    by_position = defaultdict(list)
    for r in rows:
        if r["val_diffmean"] is not None:
            by_position[r["position"]].append(r)
    lines.append("| position | best val DiffMean | layer | best val LR | layer |")
    lines.append("|---|---|---|---|---|")
    for pos in sorted(by_position):
        pos_rows = by_position[pos]
        best_dm = max(pos_rows, key=lambda r: r["val_diffmean"])
        lr_filtered = [r for r in pos_rows if r["val_lr"] is not None]
        best_lr = max(lr_filtered, key=lambda r: r["val_lr"]) if lr_filtered else None
        lr_auroc = fmt(best_lr["val_lr"]) if best_lr else "—"
        lr_layer = str(best_lr["layer"]) if best_lr else "—"
        lines.append(f"| {pos} | {fmt(best_dm['val_diffmean'])} | {best_dm['layer']} | {lr_auroc} | {lr_layer} |")
    lines.append("")

    # OOD AUROC for the best layer×position (by val DiffMean)
    if sortable:
        best = sortable[0]
        lines.append(f"## OOD slices for best (layer={best['layer']}, position={best['position']})\n")
        lines.append("| slice | n_pos | n_neg | DiffMean | LR |")
        lines.append("|---|---|---|---|---|")
        r = results[best["key"]]
        ordered = ["train", "val"]
        ordered += sorted([k for k in r if k.startswith("val_")])
        ordered += sorted([k for k in r if k.startswith("ood_")])
        for slice_name in ordered:
            if slice_name in ("layer", "position"):
                continue
            s = r.get(slice_name, {})
            if not isinstance(s, dict) or "n_pos" not in s:
                continue
            lines.append(f"| {slice_name} | {s['n_pos']} | {s['n_neg']} | "
                         f"{fmt(s.get('diffmean_auroc'))} | {fmt(s.get('lr_auroc'))} |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
