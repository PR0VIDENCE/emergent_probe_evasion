#!/bin/bash
# End-to-end multiplication evasion pipeline.
#
# Runs the three stages in sequence:
#   1. multiplication_evasion.py          — generate rollouts under each regime
#   2. multiplication_extract_activations.py — forward-pass + extract per rollout
#   3. multiplication_train_probes.py     — train probe on original 250 + score
#                                            the evasion rollouts as an OOD slice
#
# All scripts are independently resumable, so re-running this pipeline picks up
# from wherever it left off (useful if the cluster reboots mid-job).
#
# Any extra args passed to this script are forwarded to the first stage
# (multiplication_evasion.py), e.g.:
#   ./scripts/run_multiplication_evasion_pipeline.sh --batch-size 8
#
# Headline output: data/concepts/multiplication_pilot/probes_v1/ood_eval.csv
# Look for rows with slice starting "type:regime_" — that's your Pareto table.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

EVASION_DIR="data/concepts/multiplication_pilot/evasion_v1"
ROLLOUTS_PATH="${EVASION_DIR}/qwq_local/rollouts.jsonl"
ACTIVATIONS_DIR="${EVASION_DIR}/activations"

stage() {
    echo
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

stage "[1/3] Generating evasion rollouts (4 regimes × 40 problems)"
uv run python scripts/multiplication_evasion.py "$@"

stage "[2/3] Extracting activations from evasion rollouts"
uv run python scripts/multiplication_extract_activations.py --rollouts "${ROLLOUTS_PATH}" --output-dir "${ACTIVATIONS_DIR}"

stage "[3/3] Scoring with trained probe + producing Pareto numbers"
uv run python scripts/multiplication_train_probes.py --ood-activations-dir "${ACTIVATIONS_DIR}"

echo
echo "============================================================"
echo "  Pipeline complete"
echo "============================================================"
echo "Per-regime Pareto: data/concepts/multiplication_pilot/probes_v1/ood_eval.csv"
echo "Per-rollout scores: data/concepts/multiplication_pilot/probes_v1/ood_per_sample.jsonl"
