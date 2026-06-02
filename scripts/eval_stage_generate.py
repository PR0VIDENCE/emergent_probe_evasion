"""
Stage 1 — generate probe-training rollouts.

Reads the concept config's data handler, builds rollout requests, generates (or
copies prewritten completions), writes probe_training/rollouts.jsonl.

  uv run python scripts/eval_stage_generate.py --config configs/eval/qwq32b_fast.yaml --concept trees
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common, pipeline
from src.eval.data_handlers import build_probe_training_requests


def run(config, concept, ctx):
    if common.stage_done(config, concept, "generation"):
        print(f"  [{concept}] generation: skip (already complete)")
        return
    common.ensure_concept_dirs(config, concept)
    cc = config["_concept_configs"][concept]
    paths = common.stage_paths(config, concept)
    requests = build_probe_training_requests(cc, config, bool(config.get("fast")))
    n = pipeline.run_generation(config, concept, ctx, "probe_training",
                                requests, paths["pt_rollouts"])
    common.mark_stage(config, concept, "generation",
                      {"n_requests": len(requests), "n_new": n})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    config = common.load_for_cli(args.config, args.fast, args.mock)
    ctx = common.ModelContext(config, mock=args.mock)
    run(config, args.concept, ctx)


if __name__ == "__main__":
    main()
