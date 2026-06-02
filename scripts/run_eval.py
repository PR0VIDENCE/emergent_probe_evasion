"""
End-to-end probe-evasion eval orchestrator.

Runs every pipeline stage for each concept, in order, writing to a fresh
data/eval_runs/<run_id>/ tree. Resumable per concept (and per stage). A concept
whose probe fails validation is logged + skipped at evasion; one concept's
failure never crashes the whole run.

  uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml
  uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml --concepts trees,sycophancy --fast
  uv run python scripts/run_eval.py --config configs/eval/qwq32b_fast.yaml --mock   # no-GPU simulation
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.eval import common

import eval_stage_generate
import eval_stage_label
import eval_stage_extract
import eval_stage_train_probes
import eval_stage_evasion
import eval_stage_score
import eval_stage_dashboard


def run_concept(config, concept, ctx) -> str:
    """Run all stages for one concept. Returns a short status string."""
    print(f"\n{'='*70}\nCONCEPT: {concept}\n{'='*70}")

    # --- probe pipeline ---
    eval_stage_generate.run(config, concept, ctx)
    common.write_global_status(config)
    eval_stage_label.run(config, concept, ctx, phase="probe_training")
    eval_stage_extract.run(config, concept, ctx, phase="probe_training")
    common.write_global_status(config)
    validation = eval_stage_train_probes.run(config, concept)
    common.write_global_status(config)

    if validation != "passed":
        auroc = common.read_status(config, concept).get("val_auroc")
        print(f"❌ {concept}: probe failed validation (val AUROC = {auroc}) — skipping evasion")
        return "probe_failed"

    # --- evasion pipeline ---
    eval_stage_evasion.run(config, concept, ctx)
    common.write_global_status(config)
    eval_stage_label.run(config, concept, ctx, phase="evasion")
    eval_stage_extract.run(config, concept, ctx, phase="evasion")
    eval_stage_score.run(config, concept)
    common.write_global_status(config)
    return "complete"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concepts", default=None, help="Comma-separated subset.")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mock", action="store_true", help="No-GPU simulation backend.")
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args()

    config = common.load_for_cli(args.config, args.fast, args.mock, args.concepts)
    common.snapshot_config(config)
    ctx = common.ModelContext(config, mock=args.mock)

    print(f"Eval run: {config['_run_id']}  (model={config['_model_name']}, "
          f"fast={bool(config.get('fast'))}, mock={args.mock})")
    print(f"Concepts: {config['concepts']}")
    print(f"Output: {common.run_dir(config)}")

    t0 = time.time()
    outcomes = {}
    for concept in config["concepts"]:
        try:
            outcomes[concept] = run_concept(config, concept, ctx)
        except Exception as e:  # noqa: BLE001 — never let one concept crash the run
            import traceback
            traceback.print_exc()
            common.log_error(config, concept, "orchestrator",
                             f"{type(e).__name__}: {str(e)[:200]}")
            st = common.read_status(config, concept)
            st["validation"] = st.get("validation") or "error"
            st["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            common.write_status(config, concept, st)
            outcomes[concept] = "error"
        common.write_global_status(config)

    if not args.no_dashboard:
        print(f"\n{'='*70}\nBuilding dashboard\n{'='*70}")
        try:
            eval_stage_dashboard.run(config)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"Dashboard build failed: {e}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}\nRUN COMPLETE in {elapsed:.1f}min\n{'='*70}")
    for concept, outcome in outcomes.items():
        print(f"  {concept:<16} {outcome}")
    print(f"\nDashboard: {common.run_dir(config) / 'dashboard.html'}")


if __name__ == "__main__":
    main()
