"""
Stage 6 — score evasion rollouts with the canonical probe + write the report.

  uv run python scripts/eval_stage_score.py --config <cfg> --concept trees
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common
from src.eval.scoring import score_and_report


def run(config, concept, ctx=None):
    if common.stage_done(config, concept, "scoring"):
        print(f"  [{concept}] scoring: skip (already complete)")
        return
    summary = score_and_report(config, concept)
    common.mark_stage(config, concept, "scoring",
                      {"n_regimes": len(summary["regimes"])})
    print(f"  [{concept}] scoring complete → {common.stage_paths(config, concept)['ev_report']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    config = common.load_for_cli(args.config, args.fast, args.mock)
    run(config, args.concept)


if __name__ == "__main__":
    main()
