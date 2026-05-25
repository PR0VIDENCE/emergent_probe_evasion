"""
Forward-pass re-extraction of activations from Stage 1 sycophancy rollouts.

For each labeled rollout (from any prior Stage 1 run — pilot, micro-pilot,
signal-topup, etc.), reconstructs the conversation under the `neutral` system
prompt (regardless of the prompt the rollout was originally generated under),
runs a single forward pass, and saves activations at 15 layers × 5 positions.

Why re-extract under `neutral`? See design decision (3) in the experiment YAML:
the probe should see the same context-text regardless of which system prompt
elicited the behavior, so it can't cheat by detecting prompt-text features.

Positions extracted per rollout:
  last_token                  — final token of the assistant response
  end_of_reasoning            — token at </think> boundary
  first_answer_sentence_end   — first sentence-end punctuation after </think>
  answer_mean_pool            — mean over tokens after </think>
  reasoning_mean              — mean over tokens INSIDE <think>...</think>

Layers: 15 evenly-spaced (every 4th of QwQ's 64) — from the YAML's target_layers.

Output per rollout:
  {output_dir}/activations/{rollout_id}.pt   — dict {position: {layer: tensor}}
  {output_dir}/extraction_log.jsonl          — one line metadata per rollout
where rollout_id = "{question_id}_{framing}_{original_system_prompt_id}"

Resumable: skips rollouts whose .pt file already exists.

Usage:
  uv run python scripts/sycophancy_extract_activations.py --config configs/experiments/qa_probe_training_sycophancy.yaml --rollouts data/concepts/sycophancy_qa_v2/stage1_train/rollouts_labeled.jsonl
  uv run python scripts/sycophancy_extract_activations.py --config <cfg> --rollouts <path> --rollouts <path2> ...
  uv run python scripts/sycophancy_extract_activations.py --config <cfg> --auto-discover  # finds all rollouts*_labeled.jsonl under data_dir
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.inference.extract_activations import (
    load_model_and_tokenizer,
    find_token_positions,
    extract_activations_at_positions,
)


NEUTRAL_PROMPT = "You are a helpful assistant."


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def rollout_id(r):
    """Stable per-rollout key. Including original system_prompt_id keeps the
    different elicitation conditions distinguishable as separate rollouts even
    after re-extraction under neutral."""
    return f"{r['question_id']}_{r['framing']}_{r['system_prompt_id']}"


def reconstruct_assistant_text(thinking, response):
    """Rebuild the original generation as a single assistant content string,
    with the <think>...</think> structure restored so find_token_positions can
    locate end_of_reasoning. Empty-response rollouts (max_new_tokens hit during
    thinking) come back as the thinking trace alone, no </think>."""
    thinking = (thinking or "").strip()
    response = (response or "").strip()
    if thinking and response:
        return f"<think>\n{thinking}\n</think>\n\n{response}"
    if thinking:
        # Truncated mid-thinking — no </think> tag. The extractor's fallback
        # makes answer positions collapse to last_token (acceptable).
        return f"<think>\n{thinking}"
    return response


def load_rollouts(paths):
    """Load and dedupe rollouts across multiple jsonl files, keying on rollout_id."""
    seen = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = rollout_id(r)
                if key not in seen:
                    seen[key] = r
                    seen[key]["_source_file"] = str(p)
    return list(seen.values())


def discover_rollout_files(data_dir):
    """Find rollouts*labeled*.jsonl files under any stage1_* subdir.

    Restricted to stage1_* because earlier Stage 0 rollouts predate the
    question_id schema and would fail downstream — they're not relevant for
    Stage 1 probe training anyway.
    """
    files = []
    for sub in sorted(data_dir.glob("stage1_*")):
        if sub.is_dir():
            files.extend(sorted(sub.glob("rollouts*labeled*.jsonl")))
    return files


def process_rollout(r, model, tokenizer, target_layers):
    """Re-extract activations for one rollout under the neutral system prompt."""
    assistant_text = reconstruct_assistant_text(r.get("thinking", ""), r.get("response", ""))
    if not assistant_text.strip():
        return None  # nothing to extract

    messages = [
        {"role": "system", "content": NEUTRAL_PROMPT},
        {"role": "user", "content": r["user_text"]},
    ]

    # Tokenize prompt-only (system + user + generation marker) to find the
    # boundary where the assistant tokens begin.
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_len = len(prompt_ids)

    # Tokenize the full conversation (system + user + assistant).
    full_messages = messages + [{"role": "assistant", "content": assistant_text}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False,
    )["input_ids"].to(model.device)

    if full_ids.shape[1] <= input_len:
        return None  # degenerate: no assistant tokens after tokenization

    positions = find_token_positions(full_ids, input_len, tokenizer)

    # Add reasoning_mean: mean pool over the <think>...</think> tokens.
    # answer_start is the first token AFTER </think>, so reasoning lives in
    # [input_len, answer_start - 1]. If answer_start fell back to last_token
    # (no </think> found — truncated thinking), pool over the whole assistant
    # span [input_len, last_token].
    answer_start = positions.get("answer_start")
    last_token = positions["last_token"]
    if answer_start is not None and answer_start > input_len:
        reasoning_end = answer_start - 1
    else:
        reasoning_end = last_token
    reasoning_start = input_len
    extra_pools = {"reasoning_mean": (reasoning_start, reasoning_end)}

    t0 = time.time()
    activations = extract_activations_at_positions(
        full_ids,
        model,
        target_layers,
        positions,
        answer_start=positions.get("answer_start"),
        answer_end=positions.get("answer_end"),
        extra_mean_pools=extra_pools,
    )
    extract_time = time.time() - t0

    meta = {
        "rollout_id": rollout_id(r),
        "question_id": r["question_id"],
        "framing": r["framing"],
        "system_prompt_id": r["system_prompt_id"],
        "source": r["source"],
        "label_judge": r.get("label_judge"),
        # is_truncated: response field was empty (model hit max_new_tokens during
        # the <think> trace and never produced an answer). Activations at the
        # answer position are mid-deliberation noise — downstream training and
        # eval filter these by default.
        "is_truncated": not (r.get("response") or "").strip(),
        "seq_len": int(full_ids.shape[1]),
        "input_len": int(input_len),
        "assistant_tokens": int(full_ids.shape[1] - input_len),
        "positions": {k: int(v) for k, v in positions.items()},
        "reasoning_range": [int(reasoning_start), int(reasoning_end)],
        "extract_seconds": round(extract_time, 2),
        "_source_file": r.get("_source_file"),
    }
    return activations, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollouts", action="append", default=None,
                        help="Path to a rollouts_labeled.jsonl file. Can be passed multiple times.")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover all rollouts*labeled*.jsonl under data_dir")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write activations + log. Defaults to data_dir/stage1_activations/")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap processed rollouts (for smoke tests)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan, count rollouts, and exit (no model load)")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)
    data_dir = PROJECT_ROOT / config["data_dir"]
    target_layers = config["target_layers"]

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "stage1_activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "extraction_log.jsonl"

    # Resolve rollout sources.
    if args.auto_discover:
        rollout_paths = discover_rollout_files(data_dir)
    elif args.rollouts:
        rollout_paths = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.rollouts]
    else:
        raise SystemExit("Must pass --rollouts <path> (repeatable) or --auto-discover")

    print(f"Rollout files ({len(rollout_paths)}):")
    for p in rollout_paths:
        print(f"  {p}")

    rollouts = load_rollouts(rollout_paths)
    print(f"\nLoaded {len(rollouts)} unique rollouts (deduped on rollout_id)")

    # Resume: skip rollouts whose .pt already exists.
    pending = []
    skipped = 0
    for r in rollouts:
        rid = rollout_id(r)
        if (activations_dir / f"{rid}.pt").exists():
            skipped += 1
            continue
        pending.append(r)
    print(f"  already extracted: {skipped}  |  pending: {len(pending)}")

    if args.limit:
        pending = pending[:args.limit]
        print(f"  --limit {args.limit} → processing first {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Class breakdown — useful before committing to a long run.
    from collections import Counter
    by_label = Counter(r.get("label_judge", "unlabeled") for r in pending)
    by_sp = Counter(r["system_prompt_id"] for r in pending)
    by_framing = Counter(r["framing"] for r in pending)
    print(f"\nPending breakdown:")
    print(f"  by label_judge: {dict(by_label)}")
    print(f"  by system_prompt_id (original): {dict(by_sp)}")
    print(f"  by framing: {dict(by_framing)}")
    print(f"  target_layers ({len(target_layers)}): {target_layers}")

    if args.dry_run:
        print("\nDRY RUN — exiting without model load")
        return

    # Load model.
    print("\nLoading model...")
    model_config = load_yaml(PROJECT_ROOT / config["model_config"])
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s")

    overall_start = time.time()
    n_ok = 0
    n_skipped_empty = 0
    n_errors = 0
    with open(log_path, "a") as logf:
        for i, r in enumerate(pending):
            rid = rollout_id(r)
            try:
                result = process_rollout(r, model, tokenizer, target_layers)
            except Exception as e:
                n_errors += 1
                err_path = output_dir / "errors.jsonl"
                with open(err_path, "a") as ef:
                    ef.write(json.dumps({
                        "rollout_id": rid,
                        "error": f"{type(e).__name__}: {e}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }) + "\n")
                print(f"  [{i+1}/{len(pending)}] ERROR on {rid}: {type(e).__name__}: {str(e)[:120]}")
                continue

            if result is None:
                n_skipped_empty += 1
                continue

            activations, meta = result
            torch.save(activations, activations_dir / f"{rid}.pt")
            logf.write(json.dumps(meta) + "\n")
            logf.flush()
            n_ok += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(pending):
                elapsed = time.time() - overall_start
                rate = (n_ok + n_skipped_empty) / elapsed * 60 if elapsed > 0 else 0
                remaining = len(pending) - (i + 1)
                eta_min = remaining / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] ok={n_ok} empty_skip={n_skipped_empty} err={n_errors} "
                      f"({rate:.1f}/min, ETA {eta_min:.1f}min) "
                      f"last: {rid} seq_len={meta['seq_len']} t={meta['extract_seconds']:.1f}s")

            if torch.cuda.is_available() and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

    print(f"\nDONE in {(time.time() - overall_start)/60:.1f}min")
    print(f"  extracted:        {n_ok}")
    print(f"  skipped (empty):  {n_skipped_empty}")
    print(f"  errors:           {n_errors}")
    print(f"  activations dir:  {activations_dir}")
    print(f"  log:              {log_path}")


if __name__ == "__main__":
    main()
