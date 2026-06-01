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
if [ -z "${PROBES_DIR:-}" ]; then
    echo "ERROR: PROBES_DIR (deception probes dir) not set."
    echo "  This is needed for step 2 (deception). The sycophancy probes dir is"
    echo "  fixed by SYCOPHANCY_PROBES_DIR (default: stage1_probes_with_truncated)."
    exit 1
fi

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
bash scripts/deception_evasion_pipeline.sh

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
