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
        clf = LogisticRegression(C=C, solver="lbfgs",
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
    parser.add_argument("--ood-activations-dir", default=None,
                        help="Optional second activations dir to score with the trained probes "
                             "(used for adversarial / OOD eval)")
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

    # === Score distribution by slice (test set, best single probe) ===
    # Within-slice AUROC is mostly useless here since our slices are dominated
    # by single-class (all-mult or all-non-mult). Mean probe score is the right
    # quantity — it tells us how confidently the probe scores each subgroup,
    # which is exactly what we need to diagnose surface-vs-deep signal.
    print("\nMean probe score by slice (best single probe, on TEST set):")
    best = rows[0]
    best_test_probs = test_probs_table[(best["position"], best["layer"])]
    test_meta = [m for m in metadata if splits[m["problem_id"]] == "test"]
    # Re-index test_meta to match the order of y_test / probs
    # (load_activations returns ids in metadata order, and we filtered to test_ids
    # which is also in metadata order — so positions align)

    def slice_stats(filter_fn):
        idx = [i for i, m in enumerate(test_meta) if filter_fn(m)]
        if not idx:
            return {"n": 0, "n_pos": 0, "mean_score": None, "min_score": None, "max_score": None}
        sub_y = y_test[idx]
        sub_p = best_test_probs[idx]
        return {
            "n": len(idx),
            "n_pos": int(sub_y.sum()),
            "mean_score": float(sub_p.mean()),
            "min_score": float(sub_p.min()),
            "max_score": float(sub_p.max()),
        }

    slice_rows = []
    for src in ["pure_numerical", "wp_mult_direct", "wp_mult_indirect", "wp_add", "wp_sub"]:
        s = slice_stats(lambda m, src=src: m["source"] == src)
        slice_rows.append({"slice": f"src:{src}", **s})
    for op in ["mult", "add", "sub"]:
        for d in [3, 4, 5]:
            s = slice_stats(lambda m, op=op, d=d: m["operation"] == op and m["digits"] == d)
            slice_rows.append({"slice": f"{op}_d{d}", **s})
    for uok in [False, True]:
        s = slice_stats(lambda m, uok=uok: m["operation"] == "mult" and m["uses_op_keyword"] == uok)
        slice_rows.append({"slice": f"mult_uses_op_keyword={uok}", **s})
    for uok in [False, True]:
        s = slice_stats(lambda m, uok=uok: m["operation"] != "mult" and m["uses_op_keyword"] == uok)
        slice_rows.append({"slice": f"non_mult_uses_op_keyword={uok}", **s})

    print(f"  {'slice':<38} {'n':>3} {'n_pos':>5}  {'mean':>6} {'min':>6} {'max':>6}")
    for r in slice_rows:
        ms = f"{r['mean_score']:.4f}" if r['mean_score'] is not None else "  --  "
        mn = f"{r['min_score']:.4f}" if r['min_score'] is not None else "  --  "
        mx = f"{r['max_score']:.4f}" if r['max_score'] is not None else "  --  "
        print(f"  {r['slice']:<38} {r['n']:>3} {r['n_pos']:>5}  {ms:>6} {mn:>6} {mx:>6}")

    slice_csv = out_dir / "slice_scores.csv"
    with open(slice_csv, "w") as f:
        f.write("slice,n,n_pos,mean_score,min_score,max_score\n")
        for r in slice_rows:
            f.write(f"{r['slice']},{r['n']},{r['n_pos']},"
                    f"{r['mean_score'] if r['mean_score'] is not None else ''},"
                    f"{r['min_score'] if r['min_score'] is not None else ''},"
                    f"{r['max_score'] if r['max_score'] is not None else ''}\n")
    print(f"\nWrote {slice_csv}")

    # === Leave-one-source-out generalization ===
    # Train on 4 sources, test on the 5th. If a probe trained without
    # wp_mult_indirect still scores it as positive, the operation feature
    # generalized. If it scores it ambiguously, the probe was source-specific.
    print("\nLeave-one-source-out generalization (best single (layer, position)):")
    print("  (train on 4 sources, predict on held-out source)")
    print(f"  {'held_out_source':<22} {'n':>4} {'n_pos':>5} {'auroc':>7} {'mean_pos':>8} {'mean_neg':>8}")

    loso_rows = []
    X_full, y_full, ids = load_activations(metadata, rollouts_dir, best["position"])
    if X_full is not None:
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        source_lookup = {m["problem_id"]: m["source"] for m in metadata}
        all_sources = ["pure_numerical", "wp_mult_direct", "wp_mult_indirect", "wp_add", "wp_sub"]
        from sklearn.preprocessing import StandardScaler
        for held_source in all_sources:
            train_ids = [pid for pid in ids if source_lookup.get(pid) != held_source]
            test_ids_l = [pid for pid in ids if source_lookup.get(pid) == held_source]
            tr_idx = [id_to_idx[p] for p in train_ids]
            te_idx = [id_to_idx[p] for p in test_ids_l]
            li = target_layers.index(best["layer"])
            X_tr = X_full[tr_idx, li, :]
            X_te = X_full[te_idx, li, :]
            y_tr = y_full[tr_idx]
            y_te = y_full[te_idx]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegression(C=args.C, solver="lbfgs", max_iter=2000, random_state=args.seed)
            clf.fit(scaler.transform(X_tr), y_tr)
            probs = clf.predict_proba(scaler.transform(X_te))[:, 1]
            auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else None
            mean_pos = float(probs[y_te == 1].mean()) if (y_te == 1).any() else None
            mean_neg = float(probs[y_te == 0].mean()) if (y_te == 0).any() else None
            n = len(te_idx); npos = int(y_te.sum())
            auc_s = f"{auc:.4f}" if auc is not None else "  --  "
            mp = f"{mean_pos:.4f}" if mean_pos is not None else "  --  "
            mn = f"{mean_neg:.4f}" if mean_neg is not None else "  --  "
            print(f"  {held_source:<22} {n:>4} {npos:>5}  {auc_s:>6} {mp:>8} {mn:>8}")
            loso_rows.append({"held_out_source": held_source, "n": n, "n_pos": npos,
                              "auroc": auc, "mean_pos": mean_pos, "mean_neg": mean_neg})

    loso_csv = out_dir / "loso_auroc.csv"
    with open(loso_csv, "w") as f:
        f.write("held_out_source,n,n_pos,auroc,mean_pos_score,mean_neg_score\n")
        for r in loso_rows:
            f.write(f"{r['held_out_source']},{r['n']},{r['n_pos']},"
                    f"{r['auroc'] if r['auroc'] is not None else ''},"
                    f"{r['mean_pos'] if r['mean_pos'] is not None else ''},"
                    f"{r['mean_neg'] if r['mean_neg'] is not None else ''}\n")
    print(f"\nWrote {loso_csv}")

    # === OOD / adversarial eval ===
    ood_block = None
    if args.ood_activations_dir:
        from sklearn.preprocessing import StandardScaler
        ood_root = PROJECT_ROOT / args.ood_activations_dir
        ood_meta_path = ood_root / "metadata.jsonl"
        ood_act_dir = ood_root / "rollouts"
        if not ood_meta_path.exists():
            print(f"\nWARN: OOD metadata not found at {ood_meta_path} — skipping OOD eval")
        else:
            print(f"\n=== OOD eval ({ood_meta_path}) ===")
            ood_metadata = [json.loads(l) for l in open(ood_meta_path) if l.strip()]
            print(f"Loaded {len(ood_metadata)} OOD samples")

            # Refit each top-K (position, layer) probe on TRAIN data only, retain
            # scaler + ensemble for inference on OOD activations.
            train_id_set = {pid for pid, sp in splits.items() if sp == "train"}
            cached_probes = {}  # (pos, layer) -> (scaler, [clf, clf, ...])
            for r in top_k:
                pos = r["position"]; layer = r["layer"]
                X_full, y_full, ids = load_activations(metadata, rollouts_dir, pos)
                id_to_idx = {pid: i for i, pid in enumerate(ids)}
                tr_idx = [id_to_idx[pid] for pid in train_id_set if pid in id_to_idx]
                li = target_layers.index(layer)
                X_tr = X_full[tr_idx, li, :]
                y_tr = y_full[tr_idx]
                scaler_p = StandardScaler().fit(X_tr)
                Xtr_s = scaler_p.transform(X_tr)
                clfs = []
                for s in (42, 123, 456, 789):
                    clf = LogisticRegression(C=args.C, solver="lbfgs", max_iter=2000, random_state=s)
                    clf.fit(Xtr_s, y_tr)
                    clfs.append(clf)
                cached_probes[(pos, layer)] = (scaler_p, clfs)

            # Score OOD: for each top-K probe, predict; build combiner feature.
            ood_probs_per_component = []  # list of (n_ood,) arrays in top_k order
            ood_ids_align = None
            for r in top_k:
                pos = r["position"]; layer = r["layer"]
                X_ood, y_ood, ids_ood = load_activations(ood_metadata, ood_act_dir, pos)
                if X_ood is None:
                    print(f"  WARN: no OOD activations for position {pos}")
                    ood_probs_per_component.append(None)
                    continue
                li = target_layers.index(layer)
                Xo = X_ood[:, li, :]
                scaler_p, clfs = cached_probes[(pos, layer)]
                Xo_s = scaler_p.transform(Xo)
                probs = np.mean([clf.predict_proba(Xo_s)[:, 1] for clf in clfs], axis=0)
                ood_probs_per_component.append(probs)
                if ood_ids_align is None:
                    ood_ids_align = ids_ood
                else:
                    assert ids_ood == ood_ids_align, "OOD activation order mismatch across positions"

            # Best single probe — score OOD
            best_pos = best["position"]; best_layer = best["layer"]
            X_ood_best, _, _ = load_activations(ood_metadata, ood_act_dir, best_pos)
            li_best = target_layers.index(best_layer)
            sc_best, clfs_best = cached_probes[(best_pos, best_layer)]
            Xo_best = sc_best.transform(X_ood_best[:, li_best, :])
            best_ood_probs = np.mean(
                [clf.predict_proba(Xo_best)[:, 1] for clf in clfs_best], axis=0)

            # Combiner — feature is the stack of top-K OOD probs
            ood_combiner_feat = np.stack(ood_probs_per_component, axis=1)
            combiner_ood_probs = combiner.predict_proba(ood_combiner_feat)[:, 1]

            ood_id_to_meta = {m["problem_id"]: m for m in ood_metadata}
            ord_meta = [ood_id_to_meta[pid] for pid in ood_ids_align]

            # === Robust pool-position benchmarks ===
            # Average probe predictions across all 15 layers within
            # reasoning_mean_pool and answer_mean_pool. These are more stable
            # than the val-best single probe and avoid cherry-picking by
            # val-set characteristics. Same train-set as the rest of the OOD
            # analysis (no peeking).
            pool_probs = {}
            for pool_pos in ["reasoning_mean_pool", "answer_mean_pool"]:
                X_id_all, y_id_all, ids_id_all = load_activations(
                    metadata, rollouts_dir, pool_pos)
                X_ood_all, _, _ = load_activations(
                    ood_metadata, ood_act_dir, pool_pos)
                if X_id_all is None or X_ood_all is None:
                    continue
                id_to_idx = {pid: i for i, pid in enumerate(ids_id_all)}
                tr_idx = [id_to_idx[pid] for pid in train_id_set
                          if pid in id_to_idx]
                per_layer_probs = []
                for li, layer in enumerate(target_layers):
                    X_tr = X_id_all[tr_idx, li, :]
                    y_tr = y_id_all[tr_idx]
                    sc = StandardScaler().fit(X_tr)
                    cl = LogisticRegression(
                        C=args.C, solver="lbfgs", max_iter=2000,
                        random_state=args.seed)
                    cl.fit(sc.transform(X_tr), y_tr)
                    per_layer_probs.append(
                        cl.predict_proba(sc.transform(X_ood_all[:, li, :]))[:, 1])
                if per_layer_probs:
                    pool_probs[pool_pos] = np.mean(per_layer_probs, axis=0)
                    print(f"  pool benchmark {pool_pos}: averaged across "
                          f"{len(per_layer_probs)} layers, mean={pool_probs[pool_pos].mean():.4f}")

            def ood_slice(filter_fn):
                idx = [i for i, m in enumerate(ord_meta) if filter_fn(m)]
                if not idx:
                    return None
                bp = np.array([best_ood_probs[i] for i in idx])
                cp = np.array([combiner_ood_probs[i] for i in idx])
                truth = np.array([1 if ord_meta[i].get("label") == "mult" else 0 for i in idx])
                has_both = len(set(truth)) > 1 and any(ord_meta[i].get("label") != "borderline" for i in idx)
                auc_best = roc_auc_score(truth, bp) if has_both else None
                auc_combiner = roc_auc_score(truth, cp) if has_both else None
                # Accuracy: fraction of rollouts where `correct` is True.
                # Critical for the evasion-regime slices, where we want to
                # measure the (probe_drop, accuracy_drop) Pareto.
                correctness = [ord_meta[i].get("correct") for i in idx]
                valid = [c for c in correctness if c is not None]
                accuracy = (sum(1 for c in valid if c) / len(valid)) if valid else None
                out = {
                    "n": len(idx),
                    "mean_best": float(bp.mean()),
                    "mean_combiner": float(cp.mean()),
                    "auc_best": auc_best,
                    "auc_combiner": auc_combiner,
                    "accuracy": accuracy,
                }
                # Robust pool benchmarks (the recommended headline metrics).
                for pos_name, probs in pool_probs.items():
                    sub = probs[idx]
                    out[f"mean_{pos_name}"] = float(sub.mean())
                    # AUROC if we have both classes in this slice
                    if has_both:
                        out[f"auc_{pos_name}"] = roc_auc_score(truth, sub)
                return out

            # Need source/label fields in metadata — they come from rollouts.jsonl
            # passed through extract_activations. The script writes 'label' and
            # 'label_int' as mult/non_mult. For adversarial set, we override with
            # the rollout's `label` (mult/non_mult/borderline). Check metadata.
            ood_rows = []
            # By adversarial type — the strongest signal
            adv_types = sorted({m.get("source", "?") for m in ord_meta})
            for t in adv_types:
                s = ood_slice(lambda m, t=t: m.get("source") == t)
                if s is None: continue
                ood_rows.append({"slice": f"type:{t}", **s})
            # By label
            for lbl in ["mult", "non_mult", "borderline"]:
                s = ood_slice(lambda m, lbl=lbl: m.get("label") == lbl)
                if s is None: continue
                ood_rows.append({"slice": f"label:{lbl}", **s})
            # Overall mult vs non_mult AUROC (exclude borderline)
            s = ood_slice(lambda m: m.get("label") in ("mult", "non_mult"))
            if s is not None:
                ood_rows.append({"slice": "mult_vs_non_mult (excludes borderline)", **s})

            # Robust pool columns dynamically — only present if data was loaded
            pool_cols = []
            for pos_name in ["reasoning_mean_pool", "answer_mean_pool"]:
                if pos_name in pool_probs:
                    pool_cols.append(pos_name)
            pool_short = {"reasoning_mean_pool": "mean_reason_pool",
                          "answer_mean_pool":    "mean_answer_pool"}

            header_extra = "".join(f" {pool_short[p]:>16}" for p in pool_cols)
            print(f"\n  {'slice':<46} {'n':>4} {'mean_best':>10} {'mean_comb':>10} "
                  f"{'auc_best':>10} {'auc_comb':>10} {'accuracy':>9}{header_extra}")
            for r in ood_rows:
                fmt = lambda v: f"{v:.4f}" if v is not None else "   --   "
                row_extra = "".join(
                    f" {fmt(r.get(f'mean_{p}')):>16}" for p in pool_cols)
                print(f"  {r['slice']:<46} {r['n']:>4} {fmt(r['mean_best']):>10} "
                      f"{fmt(r['mean_combiner']):>10} {fmt(r['auc_best']):>10} "
                      f"{fmt(r['auc_combiner']):>10} {fmt(r.get('accuracy')):>9}"
                      f"{row_extra}")

            ood_csv = out_dir / "ood_eval.csv"
            with open(ood_csv, "w") as fh:
                header = "slice,n,mean_best,mean_combiner,auc_best,auc_combiner,accuracy"
                for p in pool_cols:
                    header += f",mean_{p},auc_{p}"
                fh.write(header + "\n")
                for r in ood_rows:
                    line = (f"{r['slice']},{r['n']},{r['mean_best']},{r['mean_combiner']},"
                            f"{r['auc_best'] if r['auc_best'] is not None else ''},"
                            f"{r['auc_combiner'] if r['auc_combiner'] is not None else ''},"
                            f"{r.get('accuracy') if r.get('accuracy') is not None else ''}")
                    for p in pool_cols:
                        mp = r.get(f'mean_{p}')
                        ap = r.get(f'auc_{p}')
                        line += f",{mp if mp is not None else ''}"
                        line += f",{ap if ap is not None else ''}"
                    fh.write(line + "\n")
            print(f"\nWrote {ood_csv}")

            # === Full (layer, position) sweep ===
            # The earlier OOD block only scored the top-K probes + the single
            # best. If the evasion regime successfully suppresses the layer-8
            # signal while leaving other layers active, the headline stats hide
            # it. Re-score every (layer, position) probe on the OOD set so we
            # can see per-regime, per-probe variance.
            print(f"\n=== Full per-probe sweep on OOD (all {len(args.positions)} positions × {len(target_layers)} layers) ===")
            sweep_rows = []  # one row per (source, position, layer)
            # Per-position, load OOD activations once
            ood_X_by_pos = {}
            for pos in args.positions:
                X_ood, _, ids_ood = load_activations(ood_metadata, ood_act_dir, pos)
                if X_ood is None:
                    continue
                # The earlier OOD block asserted activation orders match across
                # positions, so ids_ood == ood_ids_align. Trust that here.
                ood_X_by_pos[pos] = X_ood

            # Per-position, refit a single (no ensemble) probe per layer on
            # in-distribution train data, predict OOD.
            for pos in args.positions:
                if pos not in ood_X_by_pos:
                    continue
                X_id, y_id, ids_id = load_activations(metadata, rollouts_dir, pos)
                id_to_idx = {pid: i for i, pid in enumerate(ids_id)}
                tr_idx = [id_to_idx[pid] for pid in train_id_set if pid in id_to_idx]
                X_ood_pos = ood_X_by_pos[pos]
                for li, layer in enumerate(target_layers):
                    X_tr = X_id[tr_idx, li, :]
                    y_tr = y_id[tr_idx]
                    scaler_p = StandardScaler().fit(X_tr)
                    clf = LogisticRegression(
                        C=args.C, solver="lbfgs", max_iter=2000,
                        random_state=args.seed)
                    clf.fit(scaler_p.transform(X_tr), y_tr)
                    probs_ood = clf.predict_proba(
                        scaler_p.transform(X_ood_pos[:, li, :]))[:, 1]
                    # Group by source — these are regime names for evasion
                    # data, adversarial-type names for the adversarial set, etc.
                    sources = sorted({m.get("source", "?") for m in ord_meta})
                    for source in sources:
                        sidx = [i for i, m in enumerate(ord_meta)
                                if m.get("source") == source]
                        if not sidx:
                            continue
                        sub = probs_ood[sidx]
                        sweep_rows.append({
                            "source": source,
                            "position": pos,
                            "layer": layer,
                            "n": len(sidx),
                            "mean": float(sub.mean()),
                            "min": float(sub.min()),
                            "max": float(sub.max()),
                        })

            sweep_csv = out_dir / "ood_per_probe.csv"
            with open(sweep_csv, "w") as fh:
                fh.write("source,position,layer,n,mean,min,max\n")
                for r in sweep_rows:
                    fh.write(f'{r["source"]},{r["position"]},{r["layer"]},'
                             f'{r["n"]},{r["mean"]},{r["min"]},{r["max"]}\n')
            print(f"Wrote {sweep_csv}  ({len(sweep_rows)} probe×source rows)")

            # Per-source summary across all probes: where does the probe see
            # least signal under each source?
            from collections import defaultdict as _dd
            per_source = _dd(list)
            for r in sweep_rows:
                per_source[r["source"]].append(r)
            print(f"\n  {'source':<42} {'mean_across':>11} {'min_probe':>10} "
                  f"{'min_at':<28} {'max_probe':>10}")
            for source in sorted(per_source):
                vals = per_source[source]
                means = [v["mean"] for v in vals]
                mean_all = sum(means) / len(means)
                min_v = min(vals, key=lambda v: v["mean"])
                max_v = max(vals, key=lambda v: v["mean"])
                min_at = f"{min_v['position']}@L{min_v['layer']}"
                print(f"  {source:<42} {mean_all:>11.4f} "
                      f"{min_v['mean']:>10.4f} {min_at:<28} {max_v['mean']:>10.4f}")

            # Per-source: how much does the probe drop vs baseline AT THE SAME
            # (layer, position)? Useful when baseline is a source and we're
            # comparing other sources (evasion regimes) to it.
            if any(s in per_source for s in ["regime_baseline"]):
                base = {(r["position"], r["layer"]): r["mean"]
                        for r in per_source.get("regime_baseline", [])}
                if base:
                    print(f"\n  per-source max drop vs baseline (at the (pos, layer) "
                          f"where the drop is largest):")
                    print(f"  {'source':<42} {'drop':>8} {'at':<28}")
                    for source in sorted(per_source):
                        if source == "regime_baseline":
                            continue
                        gaps = []
                        for r in per_source[source]:
                            b = base.get((r["position"], r["layer"]))
                            if b is None: continue
                            gaps.append((b - r["mean"], r["position"], r["layer"]))
                        if gaps:
                            d, p, l = max(gaps)
                            print(f"  {source:<42} {d:>8.4f}  "
                                  f"{p}@L{l}")

            # Per-sample OOD scores — useful for inspecting borderlines specifically
            per_sample = []
            for i, m in enumerate(ord_meta):
                rec = {
                    "problem_id": m["problem_id"],
                    "source": m.get("source", ""),
                    "label": m.get("label", ""),
                    "uses_op_keyword": m.get("uses_op_keyword", False),
                    "correct": m.get("correct", None),
                    "score_best": float(best_ood_probs[i]),
                    "score_combiner": float(combiner_ood_probs[i]),
                }
                for pos_name, probs in pool_probs.items():
                    rec[f"score_{pos_name}"] = float(probs[i])
                per_sample.append(rec)
            per_sample_path = out_dir / "ood_per_sample.jsonl"
            with open(per_sample_path, "w") as fh:
                for s in per_sample:
                    fh.write(json.dumps(s) + "\n")
            print(f"Wrote {per_sample_path}")

            ood_block = {"slices": ood_rows, "per_sample_path": str(per_sample_path)}

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
        "loso": loso_rows,
        "ood": ood_block,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
