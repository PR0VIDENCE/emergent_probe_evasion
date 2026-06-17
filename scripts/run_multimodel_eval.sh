#!/bin/bash
# Multi-model eval: GPT-OSS 120b + OLMo 3.1 32B Think
# Concepts: sycophancy, multiplication, deception
# Output: /mnt/d2/acp23ajh/probe_evasion/{gptoss120b_full,olmo3_32b_full}/
#
# Run inside tmux/screen for persistence:
#   tmux new -s eval
#   bash scripts/run_multimodel_eval.sh 2>&1 | tee /mnt/d2/acp23ajh/probe_evasion/run.log
#
# Fully resumable — re-running skips completed stages.
set -e

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo "Started: $(date)"

# --- API keys ---
export OPENROUTER_API_KEY=""

# --- Install / sync dependencies ---
echo "=== Syncing dependencies ==="
uv sync
uv pip install triton

# Patch transformers finegrained_fp8.py: torch 2.6 (cu124) lacks float8_e8m0fnu
# which transformers 4.57+ references unconditionally at import time. We don't
# use FP8 quantization, so guarding the import is safe.
FP8_FILE=".venv/lib/python3.12/site-packages/transformers/integrations/finegrained_fp8.py"
if [ -f "$FP8_FILE" ] && grep -q "^_UE8M0_SF_DTYPE = torch.float8_e8m0fnu$" "$FP8_FILE"; then
  echo "=== Patching transformers FP8 for torch 2.6 compatibility ==="
  sed -i 's/^_UE8M0_SF_DTYPE = torch.float8_e8m0fnu$/_UE8M0_SF_DTYPE = getattr(torch, "float8_e8m0fnu", None)/' "$FP8_FILE"
fi

# --- Ensure output directory exists ---
mkdir -p /mnt/d2/acp23ajh/probe_evasion

# --- Model 1: GPT-OSS 120b (harmony, MXFP4) ---
echo ""
echo "============================================================"
echo "MODEL 1: GPT-OSS 120b"
echo "============================================================"
# uv run python scripts/run_eval.py --config configs/eval/gptoss120b_full.yaml

# --- Model 2: OLMo 3.1 32B Think (think_tags, bf16) ---
echo ""
echo "============================================================"
echo "MODEL 2: OLMo 3.1 32B Think"
echo "============================================================"
# uv run python scripts/run_eval.py --config configs/eval/olmo3_32b_full.yaml


# --- Model 3: Qwen 32B Think (think_tags, bf16) ---
echo ""
echo "============================================================"
echo "MODEL 3: Qwen 32B Think"
echo "============================================================"
uv run python scripts/run_eval.py --config configs/eval/qwq32b_full.yaml

echo ""
echo "============================================================"
echo "Both model evaluations complete: $(date)"
echo "Results: /mnt/d2/acp23ajh/probe_evasion/{gptoss120b_full,olmo3_32b_full}/"
echo "============================================================"
