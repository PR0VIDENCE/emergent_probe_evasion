"""
Calibration + tricky-false-positive analysis for the Stage 1 SyA probes.

Three things in one report:
  (1) Operating-point calibration: TPR @ FPR ∈ {0.01, 0.05, 0.10, 0.20} on OOD.
      Direct answer to "at a usable threshold, how many false positives does
      the probe make?" — what AUROC doesn't tell you.
  (2) Tricky false positives: the top-K highest-scoring NEGATIVES on OOD,
      with their question text + response excerpt. Shows which kinds of
      negatives (GA, pushback) the probe almost confuses for SyA.
  (3) Per-soft-positive rank: for each OOD SyA case (the 7 neutral/syc_strong
      capitulations the probe was NOT trained on), the probe's score and
      rank percentile within the OOD negative distribution.

By default, analyzes the top 3 probes by OOD-pooled DiffMean AUROC from
results.json. Override with --probe-keys.

Outputs in --output-dir (default <data_dir>/stage1_probes/calibration/):
  calibration_report.md       — human-readable summary
  tricky_false_positives.jsonl — full FP metadata for inspection
  per_positive_inspection.jsonl — score+rank for each OOD positive

Usage on RunPod:
  uv run python scripts/sycophancy_calibration_analysis.py \\
      --config configs/experiments/qa_probe_training_sycophancy.yaml \\
      --activations-dir data/concepts/sycophancy_qa_v2/stage1_activations \\
      --probes-dir data/concepts/sycophancy_qa_v2/stage1_probes
"""

import argparse
import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def assign_class(framing, judge_label):
    """Returns one of: 'SyA', 'GA', 'pushback', or None (excluded)."""
    if framing == "T3_user_wrong" and judge_label == "endorsed_incorrect":
        return "SyA"
    if framing == "T4_user_right" and judge_label == "endorsed_correct":
        return "GA"
    if framing == "T3_user_wrong" and judge_label == "endorsed_correct":
        return "pushback"
    return None


def parse_probe_key(key):
    """layer44_answer_mean_pool → (44, 'answer_mean_pool')"""
    if not key.startswith("layer"):
        raise ValueError(f"unrecognized probe key {key!r}")
    rest = key[len("layer"):]
    underscore = rest.find("_")
    layer = int(rest[:underscore])
    position = rest[underscore + 1:]
    return layer, position


def trunc(s, n=200):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + " [...]"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--probes-dir", required=True)
    parser.add_argument("--rollouts-glob", default=None,
                        help="Glob for labeled rollouts files. Default: <data_dir>/stage1_*/rollouts*labeled*.jsonl")
    parser.add_argument("--probe-keys", default=None,
                        help="Comma-separated probe keys (e.g. 'layer44_answer_mean_pool,layer60_reasoning_mean'). "
                             "Default: top 3 by OOD-pooled DiffMean AUROC.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-fps", type=int, default=10,
                        help="Show top-K tricky false positives per probe.")
    parser.add_argument("--fpr-points", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--train-pool-prompt", default="sycophancy_extreme")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)
    data_dir = PROJECT_ROOT / config["data_dir"]

    activations_dir = Path(args.activations_dir)
    if not activations_dir.is_absolute():
        activations_dir = PROJECT_ROOT / activations_dir
    probes_dir = Path(args.probes_dir)
    if not probes_dir.is_absolute():
        probes_dir = PROJECT_ROOT / probes_dir

    output_dir = Path(args.output_dir) if args.output_dir else probes_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe artifacts
    results_files = list(probes_dir.glob("results*.json"))
    if not results_files:
        raise FileNotFoundError(f"no results*.json under {probes_dir}")
    with open(results_files[0]) as f:
        results = json.load(f)
    directions = torch.load(probes_dir / "diffmean_directions.pt", weights_only=False)
    with open(probes_dir / "lr_probes.pkl", "rb") as f:
        lr_probes = pickle.load(f)

    # Pick probes to analyze
    if args.probe_keys:
        probe_keys = [k.strip() for k in args.probe_keys.split(",")]
    else:
        ranked = []
        for key, r in results.items():
            ood = r.get("ood_pooled", {}).get("diffmean_auroc")
            if ood is not None:
                ranked.append((key, ood))
        ranked.sort(key=lambda x: x[1], reverse=True)
        probe_keys = [k for k, _ in ranked[:3]]

    print(f"Analyzing probes: {probe_keys}")

    # Load extraction log
    log_path = activations_dir / "extraction_log.jsonl"
    log_entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                log_entries.append(json.loads(line))
    print(f"Extraction log: {len(log_entries)} rollouts")

    # Load original rollouts for response text (needed for FP inspection)
    if args.rollouts_glob:
        rollout_paths = sorted(glob.glob(args.rollouts_glob))
    else:
        pattern = str(data_dir / "stage1_*" / "rollouts*labeled*.jsonl")
        rollout_paths = sorted(glob.glob(pattern))
    print(f"Rollout files for response lookup: {len(rollout_paths)}")
    response_by_rid = {}
    for p in rollout_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                if "question_id" not in r: continue
                rid = f"{r['question_id']}_{r['framing']}_{r['system_prompt_id']}"
                response_by_rid[rid] = {
                    "user_text": r.get("user_text", ""),
                    "response": r.get("response", ""),
                    "thinking": r.get("thinking", ""),
                    "correct_answer": r.get("correct_answer", ""),
                    "incorrect_answer": r.get("incorrect_answer", ""),
                    "judge_reason": r.get("judge_reason", ""),
                }
    print(f"  loaded text for {len(response_by_rid)} rollouts")

    # Score every rollout at each probe (layer, position)
    # Skip loading more layers than we need from each .pt
    probe_specs = [parse_probe_key(k) for k in probe_keys]
    needed_layers = {layer for layer, _ in probe_specs}
    needed_positions = {pos for _, pos in probe_specs}

    print(f"\nScoring {len(log_entries)} rollouts at "
          f"{len(needed_layers)} layers × {len(needed_positions)} positions...")
    scores = {key: {} for key in probe_keys}  # probe_key → rollout_id → {"diffmean": s, "lr": s}
    n_loaded = n_missing_act = n_missing_pos = 0
    for entry in log_entries:
        rid = entry["rollout_id"]
        act_path = activations_dir / "activations" / f"{rid}.pt"
        if not act_path.exists():
            n_missing_act += 1
            continue
        acts = torch.load(act_path, weights_only=False, map_location="cpu")
        n_loaded += 1
        for key, (layer, position) in zip(probe_keys, probe_specs):
            if position not in acts or layer not in acts[position]:
                n_missing_pos += 1
                continue
            x = acts[position][layer].float().numpy()
            # DiffMean
            d = np.asarray(directions[key], dtype=np.float32)
            dm_score = float(x @ d)
            # LR ensemble (mean of decision_function across seeds)
            lr_score = float(np.mean([
                float(clf.decision_function(scaler.transform(x.reshape(1, -1)))[0])
                for scaler, clf in lr_probes[key]
            ]))
            scores[key][rid] = {"diffmean": dm_score, "lr": lr_score}
    print(f"  loaded {n_loaded} .pt files (missing: {n_missing_act}); "
          f"position misses: {n_missing_pos}")

    # Build the analysis tables
    fp_records_all = []
    pos_records_all = []
    md_lines = []
    md_lines.append("# SyA probe calibration + tricky-FP analysis")
    md_lines.append(f"\nTrain pool: `{args.train_pool_prompt}` (in-distribution)")
    md_lines.append(f"Probes analyzed: {', '.join(f'`{k}`' for k in probe_keys)}\n")

    for key in probe_keys:
        layer, position = parse_probe_key(key)
        md_lines.append(f"\n---\n\n## Probe: `{key}`\n")
        # Summary AUROC from results.json
        r_summary = results[key]
        ood = r_summary.get("ood_pooled", {})
        val = r_summary.get("val", {})
        md_lines.append(f"From training run: **val AUROC** = {val.get('diffmean_auroc', '—'):.3f} (DiffMean) "
                        f"/ {val.get('lr_auroc', '—'):.3f} (LR);  "
                        f"**ood_pooled AUROC** = {ood.get('diffmean_auroc', '—'):.3f} (DiffMean) "
                        f"/ {ood.get('lr_auroc', '—'):.3f} (LR)\n")

        # Bucket scores by (sp, class)
        # Class = SyA / GA / pushback / excluded
        buckets = defaultdict(lambda: {"diffmean": [], "lr": [], "ids": []})
        for entry in log_entries:
            rid = entry["rollout_id"]
            if rid not in scores[key]:
                continue
            cls = assign_class(entry["framing"], entry.get("label_judge"))
            if cls is None:
                continue
            sp = entry["system_prompt_id"]
            bucket = buckets[(sp, cls)]
            bucket["diffmean"].append(scores[key][rid]["diffmean"])
            bucket["lr"].append(scores[key][rid]["lr"])
            bucket["ids"].append(rid)

        # ---- OOD calibration (TPR @ FPR) ----
        # OOD positives: SyA from non-train prompts
        # OOD negatives: GA + pushback from non-train prompts
        ood_sps = sorted({sp for (sp, cls) in buckets if sp != args.train_pool_prompt})
        for scorer in ["diffmean", "lr"]:
            ood_pos_scores = []
            ood_neg_scores = []
            ood_neg_meta = []
            for (sp, cls), bucket in buckets.items():
                if sp == args.train_pool_prompt:
                    continue
                if cls == "SyA":
                    ood_pos_scores.extend(bucket[scorer])
                elif cls in ("GA", "pushback"):
                    for s, rid in zip(bucket[scorer], bucket["ids"]):
                        ood_neg_scores.append(s)
                        ood_neg_meta.append({"rollout_id": rid, "sp": sp, "class": cls})

            if not ood_pos_scores or not ood_neg_scores:
                continue

            ood_pos_scores = np.array(ood_pos_scores)
            ood_neg_scores = np.array(ood_neg_scores)

            md_lines.append(f"\n### TPR @ fixed FPR ({scorer})\n")
            md_lines.append(f"OOD pos: n={len(ood_pos_scores)}  |  OOD neg: n={len(ood_neg_scores)} (mix of GA + pushback)\n")
            md_lines.append("| FPR target | threshold | TPR | TP / total_pos | FP / total_neg |")
            md_lines.append("|---|---|---|---|---|")
            for fpr_target in args.fpr_points:
                threshold = np.quantile(ood_neg_scores, 1 - fpr_target)
                tp = int((ood_pos_scores >= threshold).sum())
                fp = int((ood_neg_scores >= threshold).sum())
                tpr = tp / len(ood_pos_scores)
                md_lines.append(f"| {fpr_target:.0%} | {threshold:.3f} | "
                                f"**{tpr:.0%}** | {tp}/{len(ood_pos_scores)} | "
                                f"{fp}/{len(ood_neg_scores)} ({100*fp/len(ood_neg_scores):.1f}%) |")

            # Also: TPR=1 threshold (catch ALL OOD positives) → what FPR does that cost?
            tpr1_thresh = ood_pos_scores.min()
            fpr_at_tpr1 = (ood_neg_scores >= tpr1_thresh).mean()
            md_lines.append(f"\n**Catching ALL {len(ood_pos_scores)} OOD positives "
                            f"(threshold = {tpr1_thresh:.3f}) costs FPR = {fpr_at_tpr1:.1%} "
                            f"({int((ood_neg_scores >= tpr1_thresh).sum())}/{len(ood_neg_scores)} negs flagged).**\n")

        # ---- Tricky false positives (DiffMean) ----
        all_negs = []
        for (sp, cls), bucket in buckets.items():
            if sp == args.train_pool_prompt:
                continue
            if cls in ("GA", "pushback"):
                for s, rid in zip(bucket["diffmean"], bucket["ids"]):
                    all_negs.append({"score": s, "rid": rid, "sp": sp, "class": cls})
        all_negs.sort(key=lambda x: -x["score"])

        md_lines.append(f"\n### Top {args.top_fps} tricky false positives (highest-scoring OOD negatives, DiffMean)\n")
        for i, neg in enumerate(all_negs[:args.top_fps], 1):
            rid = neg["rid"]
            text = response_by_rid.get(rid, {})
            log_entry = next((e for e in log_entries if e["rollout_id"] == rid), {})
            md_lines.append(f"\n**#{i}  score={neg['score']:.3f}  {neg['sp']} / {neg['class']}**  "
                            f"(qid={log_entry.get('question_id', '?')}, src={log_entry.get('source', '?')})\n")
            md_lines.append(f"- Q: {trunc(text.get('user_text', ''), 200)}")
            md_lines.append(f"- correct={text.get('correct_answer', '')!r}  "
                            f"incorrect={text.get('incorrect_answer', '')!r}")
            md_lines.append(f"- response: {trunc(text.get('response', ''), 300)}")
            md_lines.append(f"- judge: {log_entry.get('label_judge', '?')} — {trunc(text.get('judge_reason', ''), 150)}")
            fp_records_all.append({"probe_key": key, **neg, **text})

        # ---- Per-soft-positive inspection ----
        ood_pos_rids = []
        for (sp, cls), bucket in buckets.items():
            if sp == args.train_pool_prompt:
                continue
            if cls == "SyA":
                for s_dm, s_lr, rid in zip(bucket["diffmean"], bucket["lr"], bucket["ids"]):
                    ood_pos_rids.append({"rid": rid, "sp": sp, "diffmean": s_dm, "lr": s_lr})

        # Negatives for percentile reference
        neg_dm = np.sort([n["score"] for n in all_negs])  # ascending
        md_lines.append(f"\n### Per-soft-positive inspection: did the probe catch each OOD SyA case?\n")
        md_lines.append(f"`pct_of_negs_below` = percentile rank of this positive's score within the OOD negatives. "
                        f"100% = scored higher than ALL OOD negs (perfect detection).\n")
        md_lines.append("| # | prompt | qid | DM score | pct_of_negs_below | judge_reason snippet |")
        md_lines.append("|---|---|---|---|---|---|")
        ood_pos_rids.sort(key=lambda p: -p["diffmean"])
        for i, p in enumerate(ood_pos_rids, 1):
            log_entry = next((e for e in log_entries if e["rollout_id"] == p["rid"]), {})
            text = response_by_rid.get(p["rid"], {})
            pct = float((neg_dm < p["diffmean"]).mean() * 100) if len(neg_dm) else 0.0
            md_lines.append(f"| {i} | {p['sp']} | {log_entry.get('question_id', '?')} | "
                            f"{p['diffmean']:.3f} | {pct:.1f}% | "
                            f"{trunc(text.get('judge_reason', ''), 100)} |")
            pos_records_all.append({"probe_key": key, "pct_of_negs_below": pct, **p,
                                    "judge_reason": text.get("judge_reason", "")})

    # Write outputs
    report_path = output_dir / "calibration_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
    with open(output_dir / "tricky_false_positives.jsonl", "w") as f:
        for r in fp_records_all:
            f.write(json.dumps(r) + "\n")
    with open(output_dir / "per_positive_inspection.jsonl", "w") as f:
        for r in pos_records_all:
            f.write(json.dumps(r) + "\n")

    print(f"\nReports:")
    print(f"  {report_path}")
    print(f"  {output_dir / 'tricky_false_positives.jsonl'}")
    print(f"  {output_dir / 'per_positive_inspection.jsonl'}")


if __name__ == "__main__":
    main()
