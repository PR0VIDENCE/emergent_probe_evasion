"""
Stage 4 — train probes + validate against the 0.95 val-AUROC gate.

Writes probes/<position>/layer{N}_seed{S}.pt + scalers, combiner.json,
results.json, validation.json. Updates the concept's status.json with the
validation verdict (passed/failed + AUROC). Failed concepts are skipped at
evasion by the orchestrator.

  uv run python scripts/eval_stage_train_probes.py --config <cfg> --concept trees
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common
from src.eval.probes_train import train_and_validate


def run(config, concept, ctx=None):
    if common.stage_done(config, concept, "probe_validation"):
        print(f"  [{concept}] probe_training/validation: skip (already complete)")
        return common.read_status(config, concept).get("validation")
    result = train_and_validate(config, concept)
    # Record both stages + the validation verdict on the concept status.
    common.mark_stage(config, concept, "probe_training", {"trained": True})
    common.mark_stage(config, concept, "probe_validation",
                      {"passed": result["passed"], "val_auroc": result.get("val_auroc")})
    status = common.read_status(config, concept)
    status["validation"] = "passed" if result["passed"] else "failed"
    status["val_auroc"] = result.get("val_auroc")
    common.write_status(config, concept, status)
    verdict = "✅ passed" if result["passed"] else "❌ FAILED"
    print(f"  [{concept}] probe validation: {verdict} (val AUROC = {result.get('val_auroc')})")
    return status["validation"]


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
