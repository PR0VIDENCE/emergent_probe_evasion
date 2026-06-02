"""
Stage 3 — extract activations (probe_training or evasion).

  uv run python scripts/eval_stage_extract.py --config <cfg> --concept trees --phase probe_training
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common, pipeline


def run(config, concept, ctx, phase="probe_training"):
    stage = "extraction" if phase == "probe_training" else "evasion_extraction"
    if common.stage_done(config, concept, stage):
        print(f"  [{concept}] {stage}: skip (already complete)")
        return
    paths = common.stage_paths(config, concept)
    labeled = paths["pt_labeled"] if phase == "probe_training" else paths["ev_labeled"]
    acts_dir = paths["pt_acts"] if phase == "probe_training" else paths["ev_acts"]
    log = paths["pt_extract_log"] if phase == "probe_training" else paths["ev_extract_log"]
    n = pipeline.run_extraction(config, concept, ctx, phase, labeled, acts_dir, log)
    common.mark_stage(config, concept, stage, {"n_extracted": n})


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
