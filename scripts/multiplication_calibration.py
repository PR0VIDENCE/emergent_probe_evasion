"""
Calibration pilot for multiplication probe project.

Question: at what digit count does QwQ-32B's multiplication accuracy fall
below the headroom we need for the evasion experiment? Target sweet spot is
the largest digit count where baseline accuracy is still ~70%+, so there's
clear room for a capability drop under evasion pressure.

Generates `n_per_digit` random a × b problems for each digit count, sends
each through QwQ `rollouts_per_problem` times under standard sampling, then
extracts and grades the numeric answer.

Outputs:
  data/concepts/multiplication_pilot/calibration/rollouts.jsonl
  data/concepts/multiplication_pilot/calibration/summary.json

Usage on RunPod (local QwQ-32B):
  uv run python scripts/multiplication_calibration.py --batch-size 6

Usage via OpenRouter (proxy reasoning model, fast pilot):
  export OPENROUTER_API_KEY=sk-or-v1-...
  uv run python scripts/multiplication_calibration.py --api --api-model qwen/qwen3-32b

Note: QwQ-32B is not on OpenRouter; --api uses a proxy reasoning model
(default qwen/qwen3-32b). The difficulty curve transfers approximately,
not exactly — confirm the chosen digit count on RunPod before scaling.

Resume: rerunning skips (problem_id, rollout_id) pairs already in rollouts.jsonl.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


SYSTEM_PROMPT = "You are a helpful assistant."
USER_TEMPLATE = (
    "Compute {a} × {b}.\n\n"
    "Show your reasoning step by step, then put your final numeric answer "
    "in \\boxed{{}} on the last line."
)


def make_problems(digit_counts, n_per_digit, seed):
    rng = random.Random(seed)
    out = []
    for d in digit_counts:
        for i in range(n_per_digit):
            lo, hi = 10 ** (d - 1), 10 ** d - 1
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            out.append({
                "problem_id": f"d{d}_p{i}",
                "digits": d,
                "a": a, "b": b, "answer": a * b,
            })
    return out


def parse_qwq_response(text):
    if "</think>" in text:
        thinking, _, response = text.partition("</think>")
        return thinking.replace("<think>", "").strip(), response.strip()
    return text.strip(), ""


def extract_numeric_answer(response, thinking):
    """Try \\boxed{...} in response first, then in thinking, then last number
    in response. Strips commas/whitespace. Returns int or None."""
    for source in (response, thinking):
        boxed = re.findall(r"\\boxed\{([^}]*)\}", source)
        for content in reversed(boxed):
            nums = re.findall(r"\d[\d,]*", content)
            for n in reversed(nums):
                cleaned = n.replace(",", "")
                if cleaned.isdigit():
                    return int(cleaned)
    if response:
        nums = re.findall(r"\d[\d,]*", response)
        for n in reversed(nums):
            cleaned = n.replace(",", "")
            if cleaned.isdigit():
                return int(cleaned)
    return None


def generate_batch(model, tokenizer, prompt_infos, gen_config):
    import torch
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p["user_text"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompt_infos
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tokenizer.padding_side = original_padding_side

    attention_mask = inputs["attention_mask"]
    input_lens = [int(attention_mask[i].sum().item()) for i in range(len(prompt_infos))]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    t0 = time.time()
    with torch.no_grad():
        output_ids_batch = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            top_k=gen_config["top_k"],
            do_sample=True,
            pad_token_id=pad_id,
        )
    gen_time = time.time() - t0

    padded_input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompt_infos)):
        pad_count = padded_input_len - input_lens[i]
        real_output = output_ids_batch[i, pad_count:]
        generated_ids = real_output[input_lens[i]:]
        if pad_id is not None and generated_ids.numel() > 0:
            eos_positions = (generated_ids == pad_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                generated_ids = generated_ids[:eos_positions[0].item() + 1]
        full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append((full_response, int(generated_ids.shape[0])))
    return results, gen_time


def generate_with_fallback(model, tokenizer, prompt_infos, gen_config):
    import torch
    try:
        return generate_batch(model, tokenizer, prompt_infos, gen_config)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"  WARN: batch of {len(prompt_infos)} failed "
              f"({type(e).__name__}: {str(e)[:120]}); falling back to size-1")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = []
        total_time = 0.0
        for p in prompt_infos:
            r, t = generate_batch(model, tokenizer, [p], gen_config)
            results.extend(r)
            total_time += t
        return results, total_time


def api_call_one(client, model, prompt_info, gen_config, max_retries=3):
    """One OpenRouter call with simple retry/backoff. Returns dict with
    keys: full_text, n_gen_tokens, gen_seconds, error (or None)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_info["user_text"]},
                ],
                max_tokens=gen_config["max_new_tokens"],
                temperature=gen_config["temperature"],
                top_p=gen_config["top_p"],
                extra_body={
                    "top_k": gen_config["top_k"],
                    "reasoning": {"effort": "high"},
                },
            )
            elapsed = time.time() - t0
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None) or ""
            # If reasoning is returned as a separate field, splice it back in as
            # <think>...</think>content so the same parse_qwq_response works.
            if reasoning and "</think>" not in content:
                full_text = f"<think>{reasoning}</think>{content}"
            else:
                full_text = content
            usage = getattr(resp, "usage", None)
            n_gen_tokens = getattr(usage, "completion_tokens", None) if usage else None
            return {
                "full_text": full_text,
                "n_gen_tokens": n_gen_tokens,
                "gen_seconds": elapsed,
                "error": None,
            }
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)
    return {
        "full_text": "",
        "n_gen_tokens": None,
        "gen_seconds": 0.0,
        "error": f"{type(last_error).__name__}: {str(last_error)[:200]}",
    }


def run_api(args, prompt_infos, rollouts_path, gen_config):
    """Run rollouts concurrently via OpenRouter, streaming results to JSONL."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.utils.judge import make_client

    completed = set()
    if rollouts_path.exists():
        with open(rollouts_path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    completed.add((r["problem_id"], r["rollout_id"]))
        print(f"  resume: skipping {len(completed)} existing rollouts")

    pending = [pi for pi in prompt_infos
               if (pi["problem_id"], pi["rollout_id"]) not in completed]
    print(f"\nGenerating {len(pending)} rollouts via OpenRouter "
          f"(model={args.api_model}, concurrency={args.api_concurrency}, "
          f"max_tokens={gen_config['max_new_tokens']})...")
    if not pending:
        return

    client = make_client("openrouter")
    overall_start = time.time()
    n_done = 0
    n_errors = 0

    with open(rollouts_path, "a") as fh, \
         ThreadPoolExecutor(max_workers=args.api_concurrency) as pool:
        futures = {
            pool.submit(api_call_one, client, args.api_model, pi, gen_config): pi
            for pi in pending
        }
        for fut in as_completed(futures):
            pi = futures[fut]
            out = fut.result()
            if out["error"]:
                n_errors += 1
                print(f"  ERROR  {pi['problem_id']}/r{pi['rollout_id']}: {out['error']}")
                continue
            thinking, response = parse_qwq_response(out["full_text"])
            extracted = extract_numeric_answer(response, thinking)
            correct = extracted is not None and extracted == pi["true_answer"]
            result = {
                **pi,
                "thinking": thinking,
                "response": response,
                "n_gen_tokens": out["n_gen_tokens"],
                "extracted_answer": extracted,
                "correct": correct,
                "gen_seconds": round(out["gen_seconds"], 2),
                "api_model": args.api_model,
            }
            fh.write(json.dumps(result) + "\n")
            fh.flush()

            n_done += 1
            elapsed = time.time() - overall_start
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            remaining = len(pending) - n_done
            eta_min = (remaining / rate) if rate > 0 else 0
            mark = "ok " if correct else "no "
            tok = out["n_gen_tokens"] if out["n_gen_tokens"] is not None else "?"
            print(f"  [{n_done}/{len(pending)}] {mark} {pi['problem_id']} "
                  f"(d{pi['digits']}, {tok} tok, {out['gen_seconds']:.1f}s) "
                  f"rate={rate:.0f}/min ETA={eta_min:.1f}min")

    print(f"\nAPI run done in {(time.time() - overall_start)/60:.1f}min "
          f"({n_errors} errors)")


def build_summary(rollouts_path):
    by_digit = defaultdict(list)
    with open(rollouts_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_digit[r["digits"]].append(r)
    summary = {"per_digit": {}}
    for d in sorted(by_digit):
        rows = by_digit[d]
        n = len(rows)
        n_correct = sum(1 for r in rows if r["correct"])
        n_extracted = sum(1 for r in rows if r["extracted_answer"] is not None)
        mean_tok = sum(r["n_gen_tokens"] for r in rows) / n
        correct_rows = [r for r in rows if r["correct"]]
        mean_correct_tok = (sum(r["n_gen_tokens"] for r in correct_rows) / len(correct_rows)
                            if correct_rows else None)
        summary["per_digit"][d] = {
            "n": n,
            "accuracy": n_correct / n,
            "n_extracted": n_extracted,
            "extract_rate": n_extracted / n,
            "mean_n_tokens": round(mean_tok, 1),
            "mean_n_tokens_correct": round(mean_correct_tok, 1) if mean_correct_tok else None,
        }
    return summary


def print_summary_table(summary):
    print("\n" + "=" * 72)
    print(f"{'digits':>6}  {'n':>4}  {'acc':>7}  {'extract':>8}  {'mean_tok':>9}  {'mean_tok_correct':>17}")
    print("-" * 72)
    for d, s in sorted(summary["per_digit"].items()):
        mct = s["mean_n_tokens_correct"]
        mct_str = f"{mct:9.0f}" if mct is not None else "       --"
        print(f"{d:>6}  {s['n']:>4}  {s['accuracy']:>6.1%}  {s['extract_rate']:>7.1%}  "
              f"{s['mean_n_tokens']:>9.0f}  {mct_str:>17}")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-config", default="configs/models/qwq_32b.yaml")
    p.add_argument("--digit-counts", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    p.add_argument("--n-per-digit", type=int, default=5,
                   help="Distinct problems per digit count (default 5)")
    p.add_argument("--rollouts-per-problem", type=int, default=3,
                   help="Sampled rollouts per problem (default 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output-dir", default="data/concepts/multiplication_pilot/calibration")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip generation, only rebuild summary from existing rollouts.jsonl")
    p.add_argument("--api", action="store_true",
                   help="Use OpenRouter API instead of local model. Needs OPENROUTER_API_KEY.")
    p.add_argument("--api-model", default="qwen/qwen3-32b",
                   help="OpenRouter model id (default: qwen/qwen3-32b — proxy for QwQ)")
    p.add_argument("--api-concurrency", type=int, default=10)
    args = p.parse_args()

    # Keep local-QwQ and OpenRouter runs in separate subdirs so they don't
    # collide and so summaries can be compared side-by-side.
    base_dir = PROJECT_ROOT / args.output_dir
    if args.api:
        backend_tag = args.api_model.replace("/", "_")
        out_dir = base_dir / f"api_{backend_tag}"
    else:
        out_dir = base_dir / "qwq_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    if args.summary_only:
        if not rollouts_path.exists():
            print(f"ERROR: {rollouts_path} not found", file=sys.stderr)
            return 1
        summary = build_summary(rollouts_path)
        print_summary_table(summary)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {summary_path}")
        return 0

    problems = make_problems(args.digit_counts, args.n_per_digit, args.seed)
    print(f"Generated {len(problems)} problems "
          f"({args.n_per_digit} per digit count × {len(args.digit_counts)} digit counts)")

    prompt_infos = []
    for prob in problems:
        for r in range(args.rollouts_per_problem):
            prompt_infos.append({
                "problem_id": prob["problem_id"],
                "rollout_id": r,
                "digits": prob["digits"],
                "a": prob["a"], "b": prob["b"], "true_answer": prob["answer"],
                "user_text": USER_TEMPLATE.format(a=prob["a"], b=prob["b"]),
            })
    print(f"  -> {len(prompt_infos)} rollouts "
          f"({args.rollouts_per_problem} per problem)")

    if args.dry_run:
        print(f"\nDRY RUN — first 3 prompts:")
        for pi in prompt_infos[:3]:
            print(f"  [{pi['digits']}-digit]  {pi['user_text'][:120]}")
            print(f"    true = {pi['true_answer']}")
        return 0

    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    if args.api:
        run_api(args, prompt_infos, rollouts_path, gen_config)
        summary = build_summary(rollouts_path)
        print_summary_table(summary)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {rollouts_path}")
        print(f"Wrote {summary_path}")
        return 0

    completed = set()
    if rollouts_path.exists():
        with open(rollouts_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add((r["problem_id"], r["rollout_id"]))
        print(f"  resume: skipping {len(completed)} existing rollouts")

    pending = [pi for pi in prompt_infos
               if (pi["problem_id"], pi["rollout_id"]) not in completed]
    print(f"\nGenerating {len(pending)} rollouts (batch_size={args.batch_size}, "
          f"max_new_tokens={args.max_new_tokens})...")

    if not pending:
        print("Nothing to do.")
        summary = build_summary(rollouts_path)
        print_summary_table(summary)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    print("Loading model...")
    from src.inference.extract_activations import load_model_and_tokenizer
    with open(PROJECT_ROOT / args.model_config) as f:
        model_config = yaml.safe_load(f)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    overall_start = time.time()
    n_done = 0
    import torch

    with open(rollouts_path, "a") as f:
        for batch_idx, batch_start in enumerate(range(0, len(pending), args.batch_size)):
            batch = pending[batch_start:batch_start + args.batch_size]
            batch_results, batch_time = generate_with_fallback(
                model, tokenizer, batch, gen_config,
            )
            batch_correct = 0
            for pi, (full_response, n_tokens) in zip(batch, batch_results):
                thinking, response = parse_qwq_response(full_response)
                extracted = extract_numeric_answer(response, thinking)
                correct = extracted is not None and extracted == pi["true_answer"]
                if correct:
                    batch_correct += 1
                result = {
                    **pi,
                    "thinking": thinking,
                    "response": response,
                    "n_gen_tokens": n_tokens,
                    "extracted_answer": extracted,
                    "correct": correct,
                    "gen_seconds_batch": round(batch_time, 2),
                    "batch_size": len(batch),
                }
                f.write(json.dumps(result) + "\n")
            f.flush()

            n_done += len(batch)
            elapsed = time.time() - overall_start
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            remaining = len(pending) - n_done
            eta_min = (remaining / rate) if rate > 0 else 0
            print(f"  batch {batch_idx + 1}/{n_batches}: {n_done}/{len(pending)} "
                  f"in {batch_time:.1f}s [{batch_correct}/{len(batch)} correct] "
                  f"({rate:.1f}/min, ETA {eta_min:.1f}min)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nGeneration done in {(time.time() - overall_start)/60:.1f}min")
    summary = build_summary(rollouts_path)
    print_summary_table(summary)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {rollouts_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
