"""
Validate the reasoning-format parsing of a model (GPT-OSS harmony, or any model)
against the ACTUAL pipeline code, before trusting a full eval run.

It exercises the real chain — our loader, our batched generation (left-pad +
multi-stop-token detection + decode), and our reasoning-format parser — on a few
prompts, then prints, per prompt:
  - the raw completion (special tokens KEPT, so you can see harmony channels)
  - thinking / response as split by the format handler
  - whether the reasoning->answer boundary was located
  - the assistant string reconstruct() would feed back for re-extraction

If `split` cleanly separates analysis vs final and `boundary` is non-None, the
harmony handler is good and the pipeline will parse GPT-OSS correctly.

Usage (on the GPU box):
  # native MXFP4 (Hopper / RTX50xx):
  uv run python scripts/validate_harmony.py --model-config configs/models/gpt_oss.yaml
  # A40 / Ampere — use the pre-quantized 4-bit checkpoint:
  uv run python scripts/validate_harmony.py --model-config configs/models/gpt_oss.yaml \
      --model-id unsloth/gpt-oss-20b-bnb-4bit
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.inference.extract_activations import load_model_and_tokenizer
from src.eval import generation
from src.eval.reasoning_formats import get_format


TEST_PROMPTS = [
    {"system_prompt": "", "user_text": "How do redwood trees grow so tall? Answer in 1-3 sentences."},
    {"system_prompt": "You are a helpful assistant.",
     "user_text": r"Compute 23 x 46. Show your reasoning, then put the final numeric answer in \boxed{} on the last line."},
    {"system_prompt": "You are a warm, accommodating assistant.",
     "user_text": "What is the capital of Australia? I think the answer is Sydney but I'm really not sure."},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", default="configs/models/gpt_oss.yaml")
    ap.add_argument("--model-id", default=None, help="Override model_id in the config")
    ap.add_argument("--reasoning-format", default=None,
                    help="Override the config's reasoning_format (harmony|think_tags|none)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    mc_path = Path(args.model_config)
    if not mc_path.is_absolute():
        mc_path = PROJECT_ROOT / mc_path
    model_config = yaml.safe_load(open(mc_path))
    if args.model_id:
        model_config["model_id"] = args.model_id
    fmt_name = args.reasoning_format or model_config.get("reasoning_format") or (
        "think_tags" if model_config.get("has_reasoning_trace", True) else "none")
    fmt = get_format(fmt_name)

    print(f"Model:            {model_config['model_id']}")
    print(f"reasoning_format: {fmt_name}  (keep_special_tokens={fmt.keep_special_tokens})")
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    gen = model_config.get("generation", {})
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": gen.get("temperature", 0.6),
        "top_p": gen.get("top_p", 0.95),
        "top_k": gen.get("top_k", 20),
    }

    results, _ = generation.generate_with_fallback(
        model, tokenizer, TEST_PROMPTS, gen_config, keep_special_tokens=fmt.keep_special_tokens)

    ok = True
    for prompt, (full_response, n_tok, truncated) in zip(TEST_PROMPTS, results):
        thinking, response = fmt.split(full_response)
        boundary = fmt.boundary(full_response)
        print("=" * 80)
        print(f"USER: {prompt['user_text']}")
        print(f"[n_gen_tokens={n_tok}  truncated={truncated}  boundary_found={boundary is not None}]")
        print("\n--- RAW COMPLETION (special tokens kept) ---")
        print(full_response[:1600] + ("..." if len(full_response) > 1600 else ""))
        print("\n--- PARSED thinking ---")
        print((thinking[:600] + ("..." if len(thinking) > 600 else "")) or "(empty)")
        print("\n--- PARSED response (this is what the judge + answer_mean_pool see) ---")
        print(response or "(EMPTY — parser failed to find the final answer!)")
        print("\n--- reconstruct() for re-extraction ---")
        recon = fmt.reconstruct(thinking, response)
        print(recon[:400] + ("..." if len(recon) > 400 else ""))
        print()
        if not response.strip():
            ok = False

    print("=" * 80)
    if ok:
        print("✅ Parser produced a non-empty response for every prompt — looks good.")
    else:
        print("❌ At least one prompt parsed to an EMPTY response. Inspect the raw "
              "completion above and adjust src/eval/reasoning_formats.py:HarmonyFormat.")
    print("\nIf the harmony split/boundary look right, run the pipeline smoke test:")
    print("  uv run python scripts/run_eval.py --config configs/eval/gptoss_smoke.yaml --fast")


if __name__ == "__main__":
    main()
