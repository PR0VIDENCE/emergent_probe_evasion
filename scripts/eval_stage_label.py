"""
Stage 2 — label rollouts (probe_training or evasion).

  uv run python scripts/eval_stage_label.py --config <cfg> --concept trees --phase probe_training
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common, judging
from src.eval import mock as mock_mod


def run(config, concept, ctx, phase="probe_training"):
    stage = "labeling" if phase == "probe_training" else "evasion_labeling"
    if common.stage_done(config, concept, stage):
        print(f"  [{concept}] {stage}: skip (already complete)")
        return
    cc = config["_concept_configs"][concept]
    paths = common.stage_paths(config, concept)
    in_path = paths["pt_rollouts"] if phase == "probe_training" else paths["ev_rollouts"]
    out_path = paths["pt_labeled"] if phase == "probe_training" else paths["ev_labeled"]

    rollouts = common.read_jsonl(in_path)
    # Resume: keep already-labeled rollouts, only label the rest.
    existing = {r["rollout_id"]: r for r in common.read_jsonl(out_path)}
    to_label = [r for r in rollouts if r["rollout_id"] not in existing]

    if config.get("_mock"):
        for r in to_label:
            label = mock_mod.mock_judge_label(r, cc, phase)
            r["labels"] = {"judge_label": label, "judge_reason": "mock", "judge_model": "mock"}
    elif to_label:
        judging.label_rollouts(to_label, cc, config, phase,
                               progress=lambda i, n: print(f"    [{concept}/{stage}] judged {i}/{n}"))

    merged = list(existing.values()) + to_label
    # Preserve original order.
    order = {r["rollout_id"]: i for i, r in enumerate(rollouts)}
    merged.sort(key=lambda r: order.get(r["rollout_id"], 1e9))
    common.write_jsonl(out_path, merged)

    n_rel = sum(1 for r in merged if judging.is_behaviorally_relevant(r, cc))
    common.mark_stage(config, concept, stage,
                      {"n_labeled": len(merged), "n_behaviorally_relevant": n_rel})
    print(f"  [{concept}/{stage}] {len(merged)} labeled, {n_rel} behaviorally relevant")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--phase", default="probe_training", choices=["probe_training", "evasion"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    config = common.load_for_cli(args.config, args.fast, args.mock)
    run(config, args.concept, common.ModelContext(config, mock=args.mock), args.phase)


if __name__ == "__main__":
    main()
