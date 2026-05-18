"""
Pre-filter Sharma questions for Stage 1: keep only questions QwQ-32B can answer
correctly when asked bare under the `neutral` system prompt. Also drops
premise-rejection questions ("correct_answer" = No / None / N/A / Nothing /
Very few / etc.) where the SyA-vs-GA contrastive framing doesn't apply.

Two modes:

  --mode generate (default)
    Loads Sharma, drops premise-rejection, samples N candidate questions
    stratified by source, runs T1 (bare question) under `neutral`, saves to:
      data/concepts/sycophancy_qa_v2/stage0_pre_filter/rollouts.jsonl

    Followup: run scripts/sycophancy_label.py on the output to get judge labels.

  --mode select
    Reads stage0_pre_filter/rollouts_labeled.jsonl (after labeling) and writes:
      data/concepts/sycophancy_qa_v2/qwq_knowable.jsonl

    Contains only Sharma `base.question` rows where the T1+neutral judge label
    is endorsed_correct. This is the candidate pool for Stage 1.

Usage on RunPod:
  uv run python scripts/sycophancy_pre_filter.py --config configs/experiments/qa_probe_training_sycophancy.yaml --n-candidates 1500
  python scripts/sycophancy_label.py --rollouts data/concepts/sycophancy_qa_v2/stage0_pre_filter/rollouts.jsonl
  python scripts/sycophancy_pre_filter.py --config configs/experiments/qa_probe_training_sycophancy.yaml --mode select

Cost / time estimate (n_candidates=1500, batch_size=10):
  - GPU generation: ~75-90 min on L40S
  - Judge labeling: ~$0.07, ~5-10 min on Gemini Flash
  - Final yield: ~1100-1300 QwQ-knowable questions (75-85% of candidates)
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ---------------------------------------------------------------------------
# Premise-rejection filter — drop Sharma questions where `correct_answer` is
# a refusal-style placeholder. SyA-vs-GA training contrast doesn't apply to
# these (there's no positive answer to assert).
# ---------------------------------------------------------------------------

PREMISE_REJECT_EXACT = {
    s.lower() for s in [
        "No", "None", "N/A", "Nothing", "No one", "Nobody", "No comment",
        "Nothing in particular", "No recorded composer", "Nothing happens",
        "No place", "No consensus", "Very few",
        # Punctuated variants
        "No.", "None.", "Nothing.", "No one.", "Nobody.", "N/A.",
        "No comment.", "Nothing in particular.",
    ]
}
PREMISE_REJECT_PREFIXES = (
    "no ", "none ", "nothing ", "very few", "no one ", "nobody ",
)


def is_premise_reject(s: str) -> bool:
    s_low = s.lower().strip()
    if s_low in PREMISE_REJECT_EXACT:
        return True
    return any(s_low.startswith(prefix) for prefix in PREMISE_REJECT_PREFIXES)


# ---------------------------------------------------------------------------
# Sharma loader / sampler
# ---------------------------------------------------------------------------

def load_sharma_questions(path: Path) -> list[dict]:
    """Return one record per unique question (T1 = bare-question template)."""
    by_question = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tpl = rec["metadata"]["prompt_template"]
            if tpl != "{question}":  # we only need T1 records for pre-filter
                continue
            q = rec["base"]["question"]
            by_question[q] = rec
    return list(by_question.values())


def stratified_candidate_sample(records: list[dict], n_candidates: int,
                                seed: int) -> list[dict]:
    """Stratify by source, drop premise-rejection, sample up to n_candidates.

    Strategy: target an even per-source split, but if one source runs out of
    questions, redistribute the deficit to other sources with surplus.
    """
    pool = [r for r in records if not is_premise_reject(r["base"]["correct_answer"])]
    by_source = defaultdict(list)
    for r in pool:
        by_source[r["base"]["dataset"]].append(r)

    rng = random.Random(seed)
    for source in by_source:
        rng.shuffle(by_source[source])

    sources = sorted(by_source)
    target_per_source = n_candidates // len(sources)

    # First pass: take target from each, capped at availability
    take_per_source = {}
    for source in sources:
        take_per_source[source] = min(target_per_source, len(by_source[source]))

    # Redistribute deficit to sources with surplus
    deficit = n_candidates - sum(take_per_source.values())
    if deficit > 0:
        # Order surplus-rich sources by remaining capacity descending
        surplus_order = sorted(
            sources,
            key=lambda s: len(by_source[s]) - take_per_source[s],
            reverse=True,
        )
        for source in surplus_order:
            if deficit == 0:
                break
            remaining = len(by_source[source]) - take_per_source[source]
            extra = min(remaining, deficit)
            take_per_source[source] += extra
            deficit -= extra

    sampled = []
    for source in sources:
        sampled.extend(by_source[source][:take_per_source[source]])

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Generation helpers (lifted pattern from sycophancy_stage0_sanity.py)
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_qwq_response(text: str) -> tuple[str, str]:
    if "</think>" in text:
        thinking, _, response = text.partition("</think>")
        thinking = thinking.replace("<think>", "").strip()
        response = response.strip()
    else:
        thinking = text.strip()
        response = ""
    return thinking, response


def generate_batch(model, tokenizer, system_prompt, prompt_infos, gen_config):
    """Batched generation via left-padding. Returns (results, batch_seconds)
    where results is a list of (full_response, n_gen_tokens) per prompt info."""
    import torch
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": p["user_text"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompt_infos
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=4096)
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


def generate_with_fallback(model, tokenizer, system_prompt, prompt_infos, gen_config):
    import torch
    try:
        return generate_batch(model, tokenizer, system_prompt, prompt_infos, gen_config)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"  WARN: batch of {len(prompt_infos)} failed "
              f"({type(e).__name__}: {str(e)[:120]}); falling back to size-1")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = []
        total_time = 0.0
        for p in prompt_infos:
            r, t = generate_batch(model, tokenizer, system_prompt, [p], gen_config)
            results.extend(r)
            total_time += t
        return results, total_time


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def cmd_generate(args, config):
    data_dir = PROJECT_ROOT / config["data_dir"]
    sharma_path = data_dir / "sharma_answer.jsonl"
    out_dir = data_dir / "stage0_pre_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"

    print(f"Loading Sharma data: {sharma_path}")
    records = load_sharma_questions(sharma_path)
    print(f"  {len(records)} unique questions")

    n_dropped = sum(1 for r in records if is_premise_reject(r["base"]["correct_answer"]))
    print(f"  dropping {n_dropped} premise-rejection questions "
          f"({100*n_dropped/len(records):.1f}%)")

    sampled = stratified_candidate_sample(records, args.n_candidates, args.seed)
    by_source = Counter(r["base"]["dataset"] for r in sampled)
    print(f"\nStratified sample: {len(sampled)} candidates (seed={args.seed})")
    for s, n in sorted(by_source.items()):
        print(f"  {s}: {n}")

    # Build prompt_infos for generate_batch — uses the `{question}` template.
    prompts = []
    for rec in sampled:
        base = rec["base"]
        prompts.append({
            "source": base["dataset"],
            "question": base["question"],
            "framing": "T1_neutral",
            "user_text": rec["prompt"][0]["content"],  # already T1-formatted
            "correct_answer": base["correct_answer"],
            "incorrect_answer": base["incorrect_answer"],
            "long_correct_answer": base.get("long_correct_answer", ""),
            "aliases": base.get("answer", []),
        })

    if args.dry_run:
        print(f"\nDRY RUN — first 3 candidate prompts:")
        for p in prompts[:3]:
            print(f"  [{p['source']}]  {p['user_text'][:100]}")
            print(f"    correct={p['correct_answer']!r}  incorrect={p['incorrect_answer']!r}")
        return 0

    # Resume support
    completed = set()
    if rollouts_path.exists():
        with open(rollouts_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add((r["source"], r["question"]))
        print(f"  found {len(completed)} existing rollouts, will skip")

    pending = [p for p in prompts if (p["source"], p["question"]) not in completed]
    print(f"\nGenerating {len(pending)} T1+neutral rollouts (batch size {args.batch_size})...")

    # Load system prompts + model
    sys_prompts = load_yaml(data_dir / "system_prompts.yaml")
    neutral_sys = sys_prompts["prompts"]["neutral"]["content"]

    print("Loading model...")
    from src.inference.extract_activations import load_model_and_tokenizer
    model_config = load_yaml(PROJECT_ROOT / config["model_config"])
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s")

    gen_config = config["generation"]
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    overall_start = time.time()
    n_done = 0

    import torch

    with open(rollouts_path, "a") as f:
        for batch_idx, batch_start in enumerate(range(0, len(pending), args.batch_size)):
            batch = pending[batch_start:batch_start + args.batch_size]
            batch_results, batch_time = generate_with_fallback(
                model, tokenizer, neutral_sys, batch, gen_config,
            )
            for p, (full_response, n_tokens) in zip(batch, batch_results):
                thinking, response = parse_qwq_response(full_response)
                result = {
                    **p,
                    "system_prompt_id": "neutral",
                    "thinking": thinking,
                    "response": response,
                    "n_gen_tokens": n_tokens,
                    "gen_seconds_batch": round(batch_time, 2),
                    "batch_size": len(batch),
                    # No regex `label` field — we'll get the canonical label from
                    # the judge in --mode select.
                }
                f.write(json.dumps(result) + "\n")
            f.flush()

            n_done += len(batch)
            elapsed = time.time() - overall_start
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            remaining = len(pending) - n_done
            eta_min = (remaining / rate) if rate > 0 else 0
            print(f"  batch {batch_idx + 1}/{n_batches}: {n_done}/{len(pending)} "
                  f"in {batch_time:.1f}s ({rate:.1f}/min, ETA {eta_min:.1f}min)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. Output: {rollouts_path}")
    print(f"Next: python scripts/sycophancy_label.py --rollouts {rollouts_path}")
    print(f"Then: python scripts/sycophancy_pre_filter.py --config {args.config} --mode select")
    return 0


def cmd_select(args, config):
    data_dir = PROJECT_ROOT / config["data_dir"]
    out_dir = data_dir / "stage0_pre_filter"
    labeled_path = out_dir / "rollouts_labeled.jsonl"
    knowable_path = data_dir / "qwq_knowable.jsonl"

    if not labeled_path.exists():
        print(f"ERROR: labeled rollouts file not found: {labeled_path}", file=sys.stderr)
        print(f"  Run sycophancy_label.py on {out_dir / 'rollouts.jsonl'} first.",
              file=sys.stderr)
        return 1

    print(f"Reading labeled rollouts: {labeled_path}")
    labeled = [json.loads(l) for l in open(labeled_path) if l.strip()]
    print(f"  {len(labeled)} labeled rollouts")

    by_label = Counter(r["label_judge"] for r in labeled)
    print("\nLabel distribution:")
    for k in ["endorsed_correct", "endorsed_incorrect", "neither"]:
        n = by_label.get(k, 0)
        pct = 100 * n / len(labeled) if labeled else 0
        print(f"  {k:22s}  {n:4d}  ({pct:5.1f}%)")

    # Keep only endorsed_correct rollouts. Also stratify by source for later use.
    knowable = [r for r in labeled if r["label_judge"] == "endorsed_correct"]
    by_source = Counter(r["source"] for r in knowable)
    print(f"\nQwQ-knowable subset: {len(knowable)} questions "
          f"({100*len(knowable)/len(labeled):.1f}% of pool)")
    for s, n in sorted(by_source.items()):
        print(f"  {s}: {n}")

    # Output: minimal record per question for Stage 1 to load.
    with open(knowable_path, "w") as f:
        for r in knowable:
            out = {
                "source": r["source"],
                "question": r["question"],
                "correct_answer": r["correct_answer"],
                "incorrect_answer": r["incorrect_answer"],
                "long_correct_answer": r.get("long_correct_answer", ""),
                "aliases": r.get("aliases", []),
            }
            f.write(json.dumps(out) + "\n")
    print(f"\nWrote {knowable_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["generate", "select"], default="generate")
    p.add_argument("--n-candidates", type=int, default=1500,
                   help="(generate mode) number of candidate questions to sample. "
                        "Default 1500 yields ~1100-1300 QwQ-knowable.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="(generate mode) print sample prompts and exit")
    args = p.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)

    if args.mode == "generate":
        return cmd_generate(args, config)
    else:
        return cmd_select(args, config)


if __name__ == "__main__":
    sys.exit(main())
