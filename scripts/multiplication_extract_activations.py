"""
Forward-pass-only activation extraction for the multiplication pilot rollouts.

For each rollout (system + user + QwQ's already-emitted assistant message
including <think>...</think>), reconstructs the full chat sequence, runs a
single forward pass with hooks at 15 layers, and extracts activations at 5
positions: last_token, end_of_reasoning, first_answer_sentence_end,
answer_mean_pool, reasoning_mean_pool.

Output: one .pt per rollout containing {position_name: tensor(n_layers, hidden_dim)}
in fp16, plus a metadata.jsonl with labels and provenance.

Usage on cluster:
  uv run python scripts/multiplication_extract_activations.py

Resume: if metadata.jsonl already has an entry for a problem_id, that rollout
is skipped.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


DEFAULT_TARGET_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
SYSTEM_PROMPT = "You are a helpful assistant."


def compute_positions_qwq(full_ids, completion_start, tokenizer):
    """Positions for a QwQ rollout with <think>...</think> followed by answer.

    Returns dict with keys: last_token, end_of_reasoning, first_answer_sentence_end,
    answer_start, answer_end, reasoning_start, reasoning_end. All are absolute
    indices into full_ids (or fall back to last_token if undetectable).
    """
    seq_len = full_ids.shape[1]
    positions = {
        "last_token": seq_len - 1,
        "answer_end": seq_len - 1,
    }

    completion_ids = full_ids[0, completion_start:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    think_end_match = re.search(r"</think>", completion_text)

    if think_end_match:
        text_before = completion_text[:think_end_match.start()]
        tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
        think_end_offset = len(tokens_before)
        think_tag_tokens = tokenizer.encode("</think>", add_special_tokens=False)

        end_of_reasoning_offset = think_end_offset + len(think_tag_tokens) - 1
        positions["end_of_reasoning"] = min(
            completion_start + end_of_reasoning_offset, seq_len - 1)
        positions["reasoning_start"] = completion_start
        positions["reasoning_end"] = positions["end_of_reasoning"]

        answer_start_offset = think_end_offset + len(think_tag_tokens)
        text_after = completion_text[think_end_match.end():]
        stripped = text_after.lstrip()
        ws_len = len(text_after) - len(stripped)
        if ws_len > 0:
            ws_tokens = tokenizer.encode(text_after[:ws_len], add_special_tokens=False)
            answer_start_offset += len(ws_tokens)
        positions["answer_start"] = min(completion_start + answer_start_offset, seq_len - 1)

        sent_match = re.search(r"[.!?]", text_after)
        if sent_match:
            text_to_sent = text_after[:sent_match.end()]
            sent_tokens = tokenizer.encode(text_to_sent, add_special_tokens=False)
            sent_offset = answer_start_offset + len(sent_tokens) - 1
            positions["first_answer_sentence_end"] = min(
                completion_start + sent_offset, seq_len - 1)
        else:
            positions["first_answer_sentence_end"] = positions["last_token"]
    else:
        positions["end_of_reasoning"] = positions["last_token"]
        positions["reasoning_start"] = completion_start
        positions["reasoning_end"] = positions["last_token"]
        positions["answer_start"] = completion_start
        positions["first_answer_sentence_end"] = positions["last_token"]

    return positions


def process_rollout(rollout, model, tokenizer, target_layers):
    """Tokenize, forward-pass, extract activations. Returns (pos_to_stacked_tensor, positions, seq_len)."""
    import torch
    from src.inference.extract_activations import extract_activations_at_positions

    user_msg = rollout["user_text"]
    thinking = rollout.get("thinking", "") or ""
    response = rollout.get("response", "") or ""
    if thinking:
        assistant_msg = f"<think>{thinking}</think>{response}"
    else:
        assistant_msg = response

    # Tokenize the (system, user) prompt to find where the assistant content begins
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_start = len(prompt_ids)

    full_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_msg},
         {"role": "assistant", "content": assistant_msg}],
        tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    positions = compute_positions_qwq(full_ids, completion_start, tokenizer)

    extra_mean_pools = {}
    if positions["reasoning_start"] < positions["reasoning_end"]:
        extra_mean_pools["reasoning_mean_pool"] = (
            positions["reasoning_start"], positions["reasoning_end"])

    point_positions = {
        "last_token": positions["last_token"],
        "end_of_reasoning": positions["end_of_reasoning"],
        "first_answer_sentence_end": positions["first_answer_sentence_end"],
    }

    activations = extract_activations_at_positions(
        full_ids, model, target_layers, point_positions,
        answer_start=positions["answer_start"],
        answer_end=positions["answer_end"],
        extra_mean_pools=extra_mean_pools,
    )

    # Stack {layer_idx: tensor(hidden,)} -> tensor(n_layers, hidden) per position
    pos_to_stacked = {}
    for pos_name, layer_acts in activations.items():
        pos_to_stacked[pos_name] = torch.stack(
            [layer_acts[l] for l in target_layers], dim=0)

    return pos_to_stacked, positions, int(full_ids.shape[1])


def resolve_rollouts_path(rollouts_arg):
    """Handle the case where the file was downloaded with a numeric suffix
    (e.g., 'rollouts (4).jsonl')."""
    p = PROJECT_ROOT / rollouts_arg
    if p.exists():
        return p
    candidates = sorted(p.parent.glob("rollouts*.jsonl"))
    if candidates:
        return candidates[0]
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts",
                        default="data/concepts/multiplication_pilot/dataset_v1/qwq_local/rollouts.jsonl",
                        help="Path to rollouts JSONL. Falls back to rollouts*.jsonl glob.")
    parser.add_argument("--output-dir",
                        default="data/concepts/multiplication_pilot/activations_v1")
    parser.add_argument("--model-config", default="configs/models/qwq_32b.yaml")
    parser.add_argument("--target-layers", type=int, nargs="+",
                        default=DEFAULT_TARGET_LAYERS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rollouts_path = resolve_rollouts_path(args.rollouts)
    if not rollouts_path.exists():
        print(f"ERROR: rollouts not found near {rollouts_path}", file=sys.stderr)
        return 1
    print(f"Reading rollouts from: {rollouts_path}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = out_dir / "rollouts"
    act_dir.mkdir(exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"

    rollouts = [json.loads(l) for l in open(rollouts_path) if l.strip()]
    if args.limit:
        rollouts = rollouts[:args.limit]
    print(f"Loaded {len(rollouts)} rollouts")

    completed = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["problem_id"])
        print(f"  resume: {len(completed)} already extracted")

    pending = [r for r in rollouts if r["problem_id"] not in completed]
    print(f"  {len(pending)} pending")
    if not pending:
        return 0

    print("Loading model...")
    from src.inference.extract_activations import load_model_and_tokenizer
    with open(PROJECT_ROOT / args.model_config) as f:
        model_config = yaml.safe_load(f)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  loaded in {time.time() - t0:.1f}s")

    import torch

    print(f"\nExtracting at layers {args.target_layers} (n={len(args.target_layers)})...\n")
    overall_start = time.time()
    n_done = 0
    n_errors = 0

    with open(metadata_path, "a") as meta_f:
        for r in pending:
            t_rollout = time.time()
            try:
                pos_to_tensor, positions, seq_len = process_rollout(
                    r, model, tokenizer, args.target_layers)
            except Exception as e:
                n_errors += 1
                print(f"  ERROR {r['problem_id']}: {type(e).__name__}: {str(e)[:160]}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            out_path = act_dir / f"{r['problem_id']}.pt"
            torch.save(pos_to_tensor, out_path)

            # Prefer explicit label from rollout (adversarial set has "borderline").
            # Fall back to operation-derived label for the original 250.
            explicit_label = r.get("label")
            if explicit_label in ("mult", "non_mult", "borderline"):
                label = explicit_label
            else:
                label = "mult" if r["operation"] == "mult" else "non_mult"

            meta = {
                "problem_id": r["problem_id"],
                "operation": r["operation"],
                "digits": r["digits"],
                "source": r.get("source", "unknown"),
                "phrasing_style": r.get("phrasing_style", "unknown"),
                "uses_op_keyword": bool(r.get("uses_op_keyword", False)),
                "adversarial_type": r.get("adversarial_type"),
                "correct": bool(r["correct"]),
                "n_gen_tokens": r["n_gen_tokens"],
                "seq_len": seq_len,
                "positions": {k: int(v) for k, v in positions.items()},
                "label": label,
                "label_int": 1 if label == "mult" else 0,
                "target_layers": list(args.target_layers),
            }
            meta_f.write(json.dumps(meta) + "\n")
            meta_f.flush()

            n_done += 1
            elapsed_total = time.time() - overall_start
            rate = n_done / elapsed_total * 60 if elapsed_total > 0 else 0
            remaining_n = len(pending) - n_done
            eta = (remaining_n / rate) if rate > 0 else 0
            print(f"  [{n_done}/{len(pending)}] {r['problem_id']:<38} "
                  f"seq={seq_len:>5} ({time.time() - t_rollout:.1f}s) "
                  f"rate={rate:.0f}/min ETA={eta:.1f}min")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone in {(time.time() - overall_start)/60:.1f}min "
          f"({n_done} extracted, {n_errors} errors)")
    print(f"Activations: {act_dir}")
    print(f"Metadata:    {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
