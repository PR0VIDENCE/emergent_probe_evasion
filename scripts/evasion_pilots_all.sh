#!/bin/bash
#
# Run BOTH evasion pilots (sycophancy + deception) sequentially.
# Model is loaded once per pipeline; can't truly parallelize on a single GPU
# without OOM risk, so we chain them.
#
# Required env:
#   OPENROUTER_API_KEY   for the LLM judge
#   PROBES_DIR           path to a trained DECEPTION probes dir (the sycophancy
#                        pipeline reads its probes from a hardcoded default).
#
# Optional env:
#   N_QUESTIONS / N_SCENARIOS    rollouts per regime per pipeline (default 50 each)
#   BATCH_SIZE                    generation batch size (default 10)
#   PROBE_KEY                     canonical probe (default layer44_answer_mean_pool)
#   SYCOPHANCY_PROBES_DIR         override the default canonical sycophancy probes
#                                 (default: data/concepts/sycophancy_qa_v2/stage1_probes_with_truncated)
#
# Estimated total runtime: ~10-15 hours on L40S.

set -e
set -u
set -o pipefail

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set."
    exit 1
fi
# Deception probes default location — train_probes_qa.py writes to
# storage.base_dir/probes; for deception this is /workspace/deception_prewritten_data/probes.
#
# DON'T export PROBES_DIR globally — the sycophancy pipeline also reads
# PROBES_DIR and would pick up the deception path. We pass it inline only
# when invoking the deception pipeline below.
DECEPTION_PROBES_DIR=${DECEPTION_PROBES_DIR:-${PROBES_DIR:-/workspace/deception_prewritten_data/probes}}
if [ ! -d "$DECEPTION_PROBES_DIR" ]; then
    echo "ERROR: deception probes directory not found: $DECEPTION_PROBES_DIR"
    echo "  Either train the deception probes first, or set"
    echo "  DECEPTION_PROBES_DIR=/path/to/probes (or PROBES_DIR — same thing here)."
    exit 1
fi
# Wipe PROBES_DIR from the env so the sycophancy pipeline uses its own default
# (data/concepts/sycophancy_qa_v2/stage1_probes_with_truncated).
unset PROBES_DIR

start=$(date +%s)

echo ""
echo "################################################################"
echo "#  PART 1/2: SYCOPHANCY EVASION PILOT"
echo "################################################################"
bash scripts/sycophancy_evasion_pipeline.sh

echo ""
echo "################################################################"
echo "#  PART 2/2: DECEPTION EVASION PILOT"
echo "################################################################"
PROBES_DIR="$DECEPTION_PROBES_DIR" bash scripts/deception_evasion_pipeline.sh

total=$(( $(date +%s) - start ))
echo ""
echo "================================================================"
printf "BOTH evasion pilots complete in %dh %dm %ds\n" $(( total / 3600 )) $(( (total % 3600) / 60 )) $(( total % 60 ))
echo "================================================================"
echo ""
echo "Sycophancy results:"
echo "  data/concepts/sycophancy_qa_v2/stage1_evasion_pilot/evasion_report.md"
echo ""
echo "Deception results:"
echo "  data/concepts/deception_prewritten_evasion/deception_evasion_report.md"
