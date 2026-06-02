"""
Stage 5 — generate evasion rollouts across regimes (universal + concept-specific).

Only runs for concepts whose probe passed validation. Builds the regime list
(universal regimes always; extra_regimes unless --fast), then the evasion
rollout requests, then generates.

  uv run python scripts/eval_stage_evasion.py --config <cfg> --concept trees
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common, pipeline
from src.eval.data_handlers import build_regime_list, build_evasion_requests


def run(config, concept, ctx):
    if common.stage_done(config, concept, "evasion_generation"):
        print(f"  [{concept}] evasion_generation: skip (already complete)")
        return
    common.ensure_concept_dirs(config, concept)
    cc = config["_concept_configs"][concept]
    paths = common.stage_paths(config, concept)
    fast = bool(config.get("fast"))
    regimes = build_regime_list(cc, config["_universal_regimes"], fast)
    requests = build_evasion_requests(cc, config, fast, regimes)
    n = pipeline.run_generation(config, concept, ctx, "evasion",
                                requests, paths["ev_rollouts"])
    common.mark_stage(config, concept, "evasion_generation",
                      {"n_regimes": len(regimes), "n_requests": len(requests), "n_new": n,
                       "regimes": [r["name"] for r in regimes]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    config = common.load_for_cli(args.config, args.fast, args.mock)
    run(config, args.concept, common.ModelContext(config, mock=args.mock))


if __name__ == "__main__":
    main()
