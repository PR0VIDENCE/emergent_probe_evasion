#!/bin/bash
#
# End-to-end deception evasion pilot.
# Parallel structure to sycophancy_evasion_pipeline.sh.
#
#   1. Generate evasion rollouts (free-form, scenario as system prompt, 7 regimes)
#   2. Label with the deception judge (compares vs honest/deceptive references)
#   3. Re-extract activations under the ORIGINAL scenario context (Apollo-style)
#   4. Score against the canonical deception probe and write report
#
# Defaults:
#   N_SCENARIOS=50          stratified by category, paired across regimes
#   BATCH_SIZE=10           generation batch size (deception uses batch=1 internally
#                           since each rollout has a unique system prompt — scenario)
#   PROBE_KEY=layer44_answer_mean_pool
#   PROBES_DIR              Required env var or arg pointing at a trained deception probes dir
#
# Estimated runtime: ~5-8 hours for 50 scenarios × 7 regimes = 350 rollouts.
# (Slower than sycophancy because batch=1 — each rollout has a distinct scenario.)
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...
#   export PROBES_DIR=/path/to/deception_probes  # must contain diffmean_directions.pt
#   bash scripts/deception_evasion_pipeline.sh

set -e
set -u
set -o pipefail

CONFIG=configs/experiments/deception_prewritten.yaml
DATA_DIR=data/concepts/deception_prewritten_evasion
N_SCENARIOS=${N_SCENARIOS:-50}
BATCH_SIZE=${BATCH_SIZE:-10}
PROBE_KEY=${PROBE_KEY:-layer44_answer_mean_pool}
# Default to just baseline + technical_disclosure (cleanest contrast).
# Override with: REGIMES=baseline,awareness_only,strategy_omission_hedging bash ...
REGIMES=${REGIMES:-baseline,technical_disclosure}

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Labeling will fail."
    exit 1
fi
# Where train_probes_qa.py writes the deception probes — derived from
# configs/experiments/deception_prewritten.yaml's storage.base_dir, with /probes
# appended (see train_probes_qa.py line 517: probe_base_dir = data_dir/probes).
PROBES_DIR=${PROBES_DIR:-/workspace/deception_prewritten_data/probes}
if [ ! -d "$PROBES_DIR" ]; then
    echo "ERROR: PROBES_DIR not found: $PROBES_DIR"
    echo "  Expected layout: $PROBES_DIR/<position>/layer{N}_seed{S}.pt"
    echo "  If your probes are elsewhere, set: export PROBES_DIR=/path/to/probes"
    exit 1
fi

start=$(date +%s)
step() {
    s=$(date +%s)
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)]  $1"
    echo "================================================================"
}

N_REGIMES=$(echo "$REGIMES" | tr ',' '\n' | wc -l | tr -d ' ')
step "1/4: Generate deception-evasion rollouts (${N_SCENARIOS} scenarios × ${N_REGIMES} regimes: ${REGIMES})"
uv run python scripts/deception_evasion_pilot.py \
    --config "$CONFIG" \
    --n-scenarios "$N_SCENARIOS" \
    --regimes "$REGIMES" \
    --batch-size "$BATCH_SIZE"

step "2/4: Label with deception judge (Gemini Flash)"
uv run python scripts/deception_label.py \
    --rollouts "$DATA_DIR/rollouts.jsonl"

step "3/4: Re-extract activations under scenario context"
uv run python scripts/deception_evasion_extract.py \
    --config "$CONFIG" \
    --rollouts "$DATA_DIR/rollouts_labeled.jsonl"

step "4/4: Score with canonical deception probe + write report"
uv run python scripts/deception_evasion_analyze.py \
    --activations-dir "$DATA_DIR/activations_dir" \
    --probes-dir "$PROBES_DIR" \
    --evasion-dir "$DATA_DIR" \
    --probe-key "$PROBE_KEY"

total=$(( $(date +%s) - start ))
echo ""
echo "================================================================"
printf "Deception evasion pilot complete in %dh %dm %ds\n" $(( total / 3600 )) $(( (total % 3600) / 60 )) $(( total % 60 ))
echo "================================================================"
echo "Outputs:"
echo "  $DATA_DIR/rollouts.jsonl"
echo "  $DATA_DIR/rollouts_labeled.jsonl"
echo "  $DATA_DIR/activations_dir/"
echo "  $DATA_DIR/deception_evasion_report.md   ← main result"
echo "  $DATA_DIR/per_rollout_scores.jsonl"
