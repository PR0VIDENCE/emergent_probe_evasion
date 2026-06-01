"""
Forward-pass re-extraction of activations from deception evasion rollouts.

Like sycophancy_extract_activations.py but the re-extraction context is the
ORIGINAL scenario (not a fixed neutral prompt). This matches how the Apollo-
style deception probes were trained: system=scenario, user=question,
assistant=completion. We strip the evasion regime addendum so the probe sees
the same context shape it saw at training time — what changes between
rollouts is just the (free-form) assistant content.

Positions extracted per rollout:
  last_token, first_answer_sentence_end, answer_mean_pool
(Apollo-style — no end_of_reasoning since the trained probes didn't use it.
We do add reasoning_mean since the model uses <think> tags here.)

Output:
  {output_dir}/activations/{rollout_id}.pt   — {position: {layer: tensor}}
  {output_dir}/extraction_log.jsonl          — one line per rollout

where rollout_id = "deception_evasion_{regime}_{pair_id}".

Resumable: skips rollouts whose .pt already exists.

Usage:
  uv run python scripts/deception_evasion_extract.py \\
      --config configs/experiments/deception_prewritten.yaml \\
      --rollouts data/concepts/deception_prewritten_evasion/rollouts_labeled.jsonl
"""

import argparse
import json
import sys
import time
from collections import Counter
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


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_rollout_id(r):
    """Stable rollout key for deception evasion."""
    return f"deception_evasion_{r['regime']}_{r['pair_id']:04d}"


def reconstruct_assistant_text(thinking, response):
    """Rebuild full assistant content with <think>...</think>, matching how
    QwQ would natively output it. Truncated cases (no </think>) get only the
    thinking head; find_token_positions falls back to last_token."""
    thinking = (thinking or "").strip()
    response = (response or "").strip()
    if thinking and response:
        return f"<think>\n{thinking}\n</think>\n\n{response}"
    if thinking:
        return f"<think>\n{thinking}"
    return response


def process_rollout(r, model, tokenizer, target_layers):
    """Re-extract activations under the ORIGINAL scenario context (no evasion addendum)."""
    assistant_text = reconstruct_assistant_text(r.get("thinking", ""), r.get("response", ""))
    if not assistant_text.strip():
        return None

    # Re-extraction context: just the scenario as system (matches Apollo probe training)
    messages = [
        {"role": "system", "content": r["scenario"]},
        {"role": "user", "content": r["question"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_len = len(prompt_ids)

    # Optionally include the answer_prefix to match Apollo's structure
    answer_prefix = r.get("answer_prefix", "").strip()
    if answer_prefix and not assistant_text.startswith(answer_prefix):
        assistant_content = f"{answer_prefix} {assistant_text}"
    else:
        assistant_content = assistant_text

    full_messages = messages + [{"role": "assistant", "content": assistant_content}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    if full_ids.shape[1] <= input_len:
        return None

    positions = find_token_positions(full_ids, input_len, tokenizer)

    # Reasoning mean (for QwQ generations that have <think> tags)
    answer_start = positions.get("answer_start")
    last_token = positions["last_token"]
    if answer_start is not None and answer_start > input_len:
        reasoning_end = answer_start - 1
    else:
        reasoning_end = last_token
    extra_pools = {"reasoning_mean": (input_len, reasoning_end)}

    t0 = time.time()
    activations = extract_activations_at_positions(
        full_ids, model, target_layers, positions,
        answer_start=positions.get("answer_start"),
        answer_end=positions.get("answer_end"),
        extra_mean_pools=extra_pools,
    )
    extract_time = time.time() - t0

    meta = {
        "rollout_id": make_rollout_id(r),
        "pair_id": r["pair_id"],
        "regime": r["regime"],
        "category": r.get("category", "other"),
        "system_prompt_id": r["system_prompt_id"],
        "label_judge": r.get("label_judge"),
        "is_truncated": not (r.get("response") or "").strip(),
        "seq_len": int(full_ids.shape[1]),
        "input_len": int(input_len),
        "assistant_tokens": int(full_ids.shape[1] - input_len),
        "positions": {k: int(v) for k, v in positions.items()},
        "extract_seconds": round(extract_time, 2),
    }
    return activations, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollouts", required=True,
                        help="Labeled deception-evasion rollouts (.jsonl)")
    parser.add_argument("--output-dir", default=None,
                        help="Default: <rollouts_parent>/activations_dir/")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)
    target_layers = config["target_layers"]

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.is_absolute():
        rollouts_path = PROJECT_ROOT / rollouts_path

    output_dir = Path(args.output_dir) if args.output_dir else rollouts_path.parent / "activations_dir"
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "extraction_log.jsonl"

    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    print(f"Loaded {len(rollouts)} rollouts from {rollouts_path}")

    pending = []
    skipped = 0
    for r in rollouts:
        rid = make_rollout_id(r)
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

    by_label = Counter(r.get("label_judge", "unlabeled") for r in pending)
    by_regime = Counter(r["regime"] for r in pending)
    print(f"\nPending breakdown:")
    print(f"  by label_judge: {dict(by_label)}")
    print(f"  by regime:      {dict(by_regime)}")
    print(f"  target_layers ({len(target_layers)}): {target_layers}")

    if args.dry_run:
        print("\nDRY RUN — exiting before model load")
        return

    print("\nLoading model...")
    model_config = load_yaml(PROJECT_ROOT / config["model_config"])
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s")

    overall_start = time.time()
    n_ok = n_err = n_empty = 0
    with open(log_path, "a") as logf:
        for i, r in enumerate(pending):
            rid = make_rollout_id(r)
            try:
                result = process_rollout(r, model, tokenizer, target_layers)
            except Exception as e:
                n_err += 1
                err_path = output_dir / "errors.jsonl"
                with open(err_path, "a") as ef:
                    ef.write(json.dumps({"rollout_id": rid, "error": f"{type(e).__name__}: {e}",
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
                print(f"  [{i+1}/{len(pending)}] ERROR on {rid}: {type(e).__name__}: {str(e)[:120]}")
                continue
            if result is None:
                n_empty += 1
                continue
            activations, meta = result
            torch.save(activations, activations_dir / f"{rid}.pt")
            logf.write(json.dumps(meta) + "\n")
            logf.flush()
            n_ok += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(pending):
                elapsed = time.time() - overall_start
                rate = (n_ok + n_empty) / elapsed * 60 if elapsed > 0 else 0
                remaining = len(pending) - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] ok={n_ok} empty={n_empty} err={n_err}  "
                      f"({rate:.1f}/min, ETA {eta:.1f}min)  last seq_len={meta['seq_len']}")
            if torch.cuda.is_available() and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

    print(f"\nDONE in {(time.time() - overall_start)/60:.1f}min")
    print(f"  extracted={n_ok}, empty_skip={n_empty}, errors={n_err}")
    print(f"  activations: {activations_dir}")
    print(f"  log:         {log_path}")


if __name__ == "__main__":
    main()
