"""
Train linear probes on multiplication-pilot activations.

For each (layer, position) pair: stratified train/val/test split (70/15/15)
by operation × digits, standardize features, fit logistic regression with L2,
report AUROC. Then train a top-K combiner over (layer, position) pairs using
validation-set probabilities, and report a stratified AUROC breakdown.

Inputs:
  data/concepts/multiplication_pilot/activations_v1/metadata.jsonl
  data/concepts/multiplication_pilot/activations_v1/rollouts/<problem_id>.pt

Outputs:
  data/concepts/multiplication_pilot/probes_v1/per_pair_auroc.csv
  data/concepts/multiplication_pilot/probes_v1/slice_auroc.csv
  data/concepts/multiplication_pilot/probes_v1/summary.json

Usage:
  uv run python scripts/multiplication_train_probes.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def stratified_split(metadata, val_frac=0.15, test_frac=0.15, seed=42):
    """Stratify by (operation, digits). Returns dict problem_id -> 'train'/'val'/'test'."""
    import random
    rng = random.Random(seed)
    strata = defaultdict(list)
    for m in metadata:
        strata[(m["operation"], m["digits"])].append(m["problem_id"])

    splits = {}
    for key, ids in strata.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for i, pid in enumerate(ids):
            if i < n_test:
                splits[pid] = "test"
            elif i < n_test + n_val:
                splits[pid] = "val"
            else:
                splits[pid] = "train"
    return splits


def load_activations(metadata, act_dir, position):
    """Load (n_samples, n_layers, hidden_dim) tensor + labels + problem_ids for a position."""
    import torch
    xs, ys, ids = [], [], []
    for m in metadata:
        pt_path = act_dir / f"{m['problem_id']}.pt"
        if not pt_path.exists():
            continue
        data = torch.load(pt_path, weights_only=False)
        if position not in data:
            continue
        xs.append(data[position])  # (n_layers, hidden_dim) fp16
        ys.append(m["label_int"])
        ids.append(m["problem_id"])
    if not xs:
        return None, None, None
    X = torch.stack(xs, dim=0).float().numpy()  # (n_samples, n_layers, hidden_dim)
    y = torch.tensor(ys).numpy()
    return X, y, ids


def train_probe_one(X_train, y_train, X_val, y_val, X_test, y_test, C=1.0, seeds=(42, 123, 456, 789)):
    """Train an ensemble of logistic-regression probes; return mean val/test AUROC
    and the mean prediction probabilities on val/test for downstream combiner."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import numpy as np

    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)
    Xs = scaler.transform(X_test)

    val_probs_seeds, test_probs_seeds = [], []
    val_aurocs, test_aurocs = [], []
    for s in seeds:
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                 max_iter=2000, random_state=s)
        clf.fit(Xt, y_train)
        vp = clf.predict_proba(Xv)[:, 1]
        tp = clf.predict_proba(Xs)[:, 1]
        val_probs_seeds.append(vp)
        test_probs_seeds.append(tp)
        if len(set(y_val)) > 1:
            val_aurocs.append(roc_auc_score(y_val, vp))
        if len(set(y_test)) > 1:
            test_aurocs.append(roc_auc_score(y_test, tp))
    val_mean_arr = np.mean(val_probs_seeds, axis=0)
    test_mean_arr = np.mean(test_probs_seeds, axis=0)
    return {
        "val_auroc": float(np.mean(val_aurocs)) if val_aurocs else None,
        "test_auroc": float(np.mean(test_aurocs)) if test_aurocs else None,
        "val_probs": val_mean_arr,
        "test_probs": test_mean_arr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-dir",
                        default="data/concepts/multiplication_pilot/activations_v1")
    parser.add_argument("--output-dir",
                        default="data/concepts/multiplication_pilot/probes_v1")
    parser.add_argument("--positions", nargs="+",
                        default=["last_token", "end_of_reasoning",
                                 "first_answer_sentence_end", "answer_mean_pool",
                                 "reasoning_mean_pool"])
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of (layer, position) pairs in the combiner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="L2 reg inverse strength")
    args = parser.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    act_root = PROJECT_ROOT / args.activations_dir
    metadata_path = act_root / "metadata.jsonl"
    rollouts_dir = act_root / "rollouts"
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run extraction first.", file=sys.stderr)
        return 1

    metadata = [json.loads(l) for l in open(metadata_path) if l.strip()]
    target_layers = metadata[0]["target_layers"]
    print(f"Loaded metadata: {len(metadata)} samples, layers {target_layers}")

    splits = stratified_split(metadata, seed=args.seed)
    split_counts = Counter(splits.values())
    print(f"Splits: {dict(split_counts)}")

    # Per-split label balance check
    for sp in ("train", "val", "test"):
        labels = [m["label_int"] for m in metadata if splits[m["problem_id"]] == sp]
        pos = sum(labels)
        n = len(labels)
        print(f"  {sp}: n={n} pos={pos} ({pos/n:.1%})")

    # === Per (position, layer) probe training ===
    print(f"\nTraining probes at {len(target_layers)} layers × {len(args.positions)} positions...")
    rows = []
    val_probs_table = {}   # (pos, layer) -> array aligned to val set
    test_probs_table = {}  # similarly
    val_ids = [m["problem_id"] for m in metadata if splits[m["problem_id"]] == "val"]
    test_ids = [m["problem_id"] for m in metadata if splits[m["problem_id"]] == "test"]

    for pos in args.positions:
        X_full, y_full, ids = load_activations(metadata, rollouts_dir, pos)
        if X_full is None:
            print(f"  WARN: no activations for position {pos}, skipping")
            continue
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        train_idx = [id_to_idx[pid] for pid, sp in splits.items() if sp == "train" and pid in id_to_idx]
        val_idx = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
        test_idx = [id_to_idx[pid] for pid in test_ids if pid in id_to_idx]

        for li, layer in enumerate(target_layers):
            X = X_full[:, li, :]  # (n, hidden)
            res = train_probe_one(
                X[train_idx], y_full[train_idx],
                X[val_idx],   y_full[val_idx],
                X[test_idx],  y_full[test_idx],
                C=args.C,
            )
            rows.append({
                "position": pos,
                "layer": layer,
                "val_auroc": res["val_auroc"],
                "test_auroc": res["test_auroc"],
            })
            val_probs_table[(pos, layer)] = res["val_probs"]
            test_probs_table[(pos, layer)] = res["test_probs"]

    rows.sort(key=lambda r: (r["val_auroc"] if r["val_auroc"] is not None else -1), reverse=True)

    # Print top 10 and bottom 5
    print("\nTop 10 (layer, position) by VAL AUROC:")
    print(f"  {'position':<28} {'layer':>5}  {'val_auroc':>9}  {'test_auroc':>10}")
    for r in rows[:10]:
        va = f"{r['val_auroc']:.4f}" if r['val_auroc'] is not None else "  --  "
        ta = f"{r['test_auroc']:.4f}" if r['test_auroc'] is not None else "  --  "
        print(f"  {r['position']:<28} {r['layer']:>5}  {va:>9}  {ta:>10}")
    print("\nBottom 5:")
    for r in rows[-5:]:
        va = f"{r['val_auroc']:.4f}" if r['val_auroc'] is not None else "  --  "
        ta = f"{r['test_auroc']:.4f}" if r['test_auroc'] is not None else "  --  "
        print(f"  {r['position']:<28} {r['layer']:>5}  {va:>9}  {ta:>10}")

    # Save per-pair CSV
    csv_path = out_dir / "per_pair_auroc.csv"
    with open(csv_path, "w") as f:
        f.write("position,layer,val_auroc,test_auroc\n")
        for r in rows:
            f.write(f"{r['position']},{r['layer']},"
                    f"{r['val_auroc'] if r['val_auroc'] is not None else ''},"
                    f"{r['test_auroc'] if r['test_auroc'] is not None else ''}\n")
    print(f"\nWrote {csv_path}")

    # === Top-K layer combiner ===
    print(f"\nTraining top-{args.top_k} combiner (logistic regression on val probs)...")
    top_k = rows[:args.top_k]
    print("  Components:")
    for r in top_k:
        print(f"    {r['position']:<28} layer={r['layer']:>3} val_auroc={r['val_auroc']:.4f}")

    val_feat = np.stack([val_probs_table[(r["position"], r["layer"])] for r in top_k], axis=1)
    test_feat = np.stack([test_probs_table[(r["position"], r["layer"])] for r in top_k], axis=1)
    y_val = np.array([m["label_int"] for m in metadata if splits[m["problem_id"]] == "val"])
    y_test = np.array([m["label_int"] for m in metadata if splits[m["problem_id"]] == "test"])

    combiner = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
    combiner.fit(val_feat, y_val)
    val_combiner_probs = combiner.predict_proba(val_feat)[:, 1]
    test_combiner_probs = combiner.predict_proba(test_feat)[:, 1]
    val_combiner_auroc = roc_auc_score(y_val, val_combiner_probs) if len(set(y_val)) > 1 else None
    test_combiner_auroc = roc_auc_score(y_test, test_combiner_probs) if len(set(y_test)) > 1 else None
    print(f"  combiner val_auroc = {val_combiner_auroc:.4f}")
    print(f"  combiner test_auroc = {test_combiner_auroc:.4f}")

    # === Stratified slice AUROC on test set ===
    print("\nStratified test AUROC (using best single (layer, position)):")
    best = rows[0]
    best_test_probs = test_probs_table[(best["position"], best["layer"])]
    test_meta = [m for m in metadata if splits[m["problem_id"]] == "test"]

    def slice_auroc(filter_fn):
        idx = [i for i, m in enumerate(test_meta) if filter_fn(m)]
        if not idx:
            return None, 0, 0
        sub_y = y_test[idx]
        sub_p = best_test_probs[idx]
        if len(set(sub_y)) < 2:
            return None, len(idx), int(sub_y.sum())
        return roc_auc_score(sub_y, sub_p), len(idx), int(sub_y.sum())

    slice_rows = []
    for d in [3, 4, 5]:
        for op in ["mult", "add", "sub"]:
            auc, n, npos = slice_auroc(lambda m, d=d, op=op: m["digits"] == d and m["operation"] == op)
            slice_rows.append({"slice": f"{op}_d{d}", "n": n, "n_pos": npos, "auroc": auc})
    for src in ["pure_numerical", "wp_mult_direct", "wp_mult_indirect", "wp_add", "wp_sub"]:
        auc, n, npos = slice_auroc(lambda m, src=src: m["source"] == src)
        slice_rows.append({"slice": f"src:{src}", "n": n, "n_pos": npos, "auroc": auc})
    # uses_op_keyword effect (restricted to mult — adversarial questions vs not)
    for uok in [False, True]:
        auc, n, npos = slice_auroc(lambda m, uok=uok: m["operation"] == "mult" and m["uses_op_keyword"] == uok)
        slice_rows.append({"slice": f"mult_uses_op_keyword={uok}", "n": n, "n_pos": npos, "auroc": auc})

    print(f"  {'slice':<40} {'n':>4} {'n_pos':>6} {'auroc':>8}")
    for r in slice_rows:
        auc_s = f"{r['auroc']:.4f}" if r['auroc'] is not None else "  --  "
        print(f"  {r['slice']:<40} {r['n']:>4} {r['n_pos']:>6} {auc_s:>8}")

    slice_csv = out_dir / "slice_auroc.csv"
    with open(slice_csv, "w") as f:
        f.write("slice,n,n_pos,auroc\n")
        for r in slice_rows:
            f.write(f"{r['slice']},{r['n']},{r['n_pos']},"
                    f"{r['auroc'] if r['auroc'] is not None else ''}\n")
    print(f"\nWrote {slice_csv}")

    summary = {
        "n_samples": len(metadata),
        "splits": dict(split_counts),
        "best_single": {
            "position": best["position"],
            "layer": best["layer"],
            "val_auroc": best["val_auroc"],
            "test_auroc": best["test_auroc"],
        },
        "combiner": {
            "top_k": args.top_k,
            "components": [{"position": r["position"], "layer": r["layer"]} for r in top_k],
            "val_auroc": val_combiner_auroc,
            "test_auroc": test_combiner_auroc,
        },
        "slices": slice_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
