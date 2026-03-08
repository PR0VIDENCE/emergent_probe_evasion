"""
Extract activations from prewritten completions via forward pass (no generation).

Apollo-style approach: for each scenario, the prompt context is identical —
only the completion text differs between deceptive and honest variants.
This removes the instruction-leakage confound present in the generate-and-extract
pipeline where "Be deceptive" vs "Be honest" instructions differ.

Supports two datasets:
  --dataset roleplaying   371 pairs from Apollo roleplaying scenarios
  --dataset ai_liar       24 pairs from "How to Catch an AI Liar"

Neutral control mode (--neutral-control): pairs honest completions from different
scenarios so both tree and non_tree get honest text. Probe should score ~0.5 AUROC.

Output format is identical to generate_and_extract.py (generation JSONs, activation
.pt files, generation_log.jsonl) so downstream train/eval scripts work unchanged.

Usage:
    python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten.yaml --dataset roleplaying
    python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten_ai_liar.yaml --dataset ai_liar
    python scripts/extract_prewritten_activations.py --config configs/experiments/deception_prewritten.yaml --dataset roleplaying --neutral-control --output-dir /workspace/deception_prewritten_neutral
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.extract_activations import (
    load_model_and_tokenizer,
    extract_activations_at_positions,
)
from src.utils.logging import setup_logging

# Reuse save helpers from generate_and_extract
from scripts.generate_and_extract import (
    save_generation,
    save_activations,
    load_completed_ids,
)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_scenarios(scenarios_path: str) -> List[dict]:
    """
    Load pre-converted scenarios from scenarios.yaml.

    These files are generated locally by convert_prewritten_data.py from the
    gitignored deception-detection/ repo, then committed so they're available
    on RunPod.

    Returns list of scenario dicts (format depends on dataset type).
    """
    with open(scenarios_path, "r") as f:
        data = yaml.safe_load(f)

    scenarios = data["scenarios"]
    print(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")
    return scenarios


def build_roleplaying_prompts(pairs: List[dict], neutral_control: bool = False) -> List[dict]:
    """
    Build prompt list from roleplaying pairs.

    Each pair produces two prompts. With neutral_control=True, both variants
    get honest completions (from different scenarios).

    Returns list of prompt dicts with keys:
        prompt_id, label, global_pair_id, category, question_type,
        messages (list of role/content dicts), completion_text
    """
    prompts = []

    if neutral_control:
        # Pair honest completions from adjacent scenarios
        for i, pair in enumerate(pairs):
            partner_idx = (i + 1) % len(pairs)
            partner = pairs[partner_idx]

            # "tree" gets honest completion from this scenario
            prompts.append({
                "prompt_id": f"tree_{pair['pair_id']:04d}",
                "label": "tree",
                "global_pair_id": pair["pair_id"],
                "category": pair["category"],
                "question_type": "roleplaying_neutral",
                "messages": [
                    {"role": "system", "content": pair["scenario"]},
                    {"role": "user", "content": pair["question"]},
                ],
                "answer_prefix": pair["answer_prefix"],
                "completion_text": pair["honest_completion"],
            })

            # "non_tree" gets honest completion from partner scenario
            prompts.append({
                "prompt_id": f"non_tree_{pair['pair_id']:04d}",
                "label": "non_tree",
                "global_pair_id": pair["pair_id"],
                "category": pair["category"],
                "question_type": "roleplaying_neutral",
                "messages": [
                    {"role": "system", "content": pair["scenario"]},
                    {"role": "user", "content": pair["question"]},
                ],
                "answer_prefix": pair["answer_prefix"],
                "completion_text": partner["honest_completion"],
            })
    else:
        for pair in pairs:
            # Deceptive (tree)
            prompts.append({
                "prompt_id": f"tree_{pair['pair_id']:04d}",
                "label": "tree",
                "global_pair_id": pair["pair_id"],
                "category": pair["category"],
                "question_type": "roleplaying",
                "messages": [
                    {"role": "system", "content": pair["scenario"]},
                    {"role": "user", "content": pair["question"]},
                ],
                "answer_prefix": pair["answer_prefix"],
                "completion_text": pair["deceptive_completion"],
            })

            # Honest (non_tree)
            prompts.append({
                "prompt_id": f"non_tree_{pair['pair_id']:04d}",
                "label": "non_tree",
                "global_pair_id": pair["pair_id"],
                "category": pair["category"],
                "question_type": "roleplaying",
                "messages": [
                    {"role": "system", "content": pair["scenario"]},
                    {"role": "user", "content": pair["question"]},
                ],
                "answer_prefix": pair["answer_prefix"],
                "completion_text": pair["honest_completion"],
            })

    return prompts


def build_ai_liar_prompts(pairs: List[dict]) -> List[dict]:
    """
    Build prompt list from AI Liar pairs.

    Same answer text in both — deception comes from the system instruction context.
    """
    prompts = []

    for pair in pairs:
        q_prefix = pair["question_prefix"]
        a_prefix = pair["answer_prefix"]
        question_text = f"{q_prefix}{pair['question']}"

        # Deceptive (tree): deceive_instruction as system
        prompts.append({
            "prompt_id": f"tree_{pair['pair_id']:04d}",
            "label": "tree",
            "global_pair_id": pair["pair_id"],
            "category": pair["category"],
            "question_type": "ai_liar",
            "messages": [
                {"role": "system", "content": pair["deceive_instruction"]},
                {"role": "user", "content": question_text},
            ],
            "answer_prefix": a_prefix,
            "completion_text": f"{a_prefix}{pair['answer']}",
        })

        # Honest (non_tree): normal_instruction as system
        prompts.append({
            "prompt_id": f"non_tree_{pair['pair_id']:04d}",
            "label": "non_tree",
            "global_pair_id": pair["pair_id"],
            "category": pair["category"],
            "question_type": "ai_liar",
            "messages": [
                {"role": "system", "content": pair["normal_instruction"]},
                {"role": "user", "content": question_text},
            ],
            "answer_prefix": a_prefix,
            "completion_text": f"{a_prefix}{pair['answer']}",
        })

    return prompts


def find_first_sentence_end(token_ids: torch.Tensor, start_pos: int, tokenizer) -> Optional[int]:
    """Find the first sentence-ending punctuation token (.!?) at or after start_pos."""
    for pos in range(start_pos, token_ids.shape[1]):
        token_text = tokenizer.decode(token_ids[0, pos:pos + 1])
        if any(ch in token_text for ch in ".!?"):
            return pos
    return None


def compute_positions(full_ids: torch.Tensor, completion_start: int, tokenizer) -> dict:
    """
    Compute extraction positions for a prewritten completion.

    No end_of_reasoning (no <think> tags in prewritten text).

    Returns dict with:
        last_token, first_answer_sentence_end, answer_start, answer_end
    """
    seq_len = full_ids.shape[1]

    positions = {
        "last_token": seq_len - 1,
        "first_answer_sentence_end": None,
        "answer_start": completion_start,
        "answer_end": seq_len - 1,
    }

    # Find first sentence-ending punctuation in the completion
    sent_end = find_first_sentence_end(full_ids, completion_start, tokenizer)
    if sent_end is not None:
        positions["first_answer_sentence_end"] = sent_end
    else:
        positions["first_answer_sentence_end"] = positions["last_token"]

    return positions


def process_prompt(
    prompt_info: dict,
    model,
    tokenizer,
    target_layers: List[int],
    output_dir: str,
) -> dict:
    """
    Process a single prewritten prompt: tokenize, forward pass, extract, save.

    Returns dict with metadata for the generation log.
    """
    messages = prompt_info["messages"]
    completion_text = prompt_info["completion_text"]
    answer_prefix = prompt_info.get("answer_prefix", "")

    # Build the assistant completion including the answer_prefix
    if answer_prefix:
        assistant_content = f"{answer_prefix} {completion_text}"
    else:
        assistant_content = completion_text

    # Tokenize prompt (system + user) without assistant to find completion boundary
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_start = len(prompt_ids)

    # Tokenize full sequence (system + user + assistant)
    full_messages = messages + [{"role": "assistant", "content": assistant_content}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    # Compute positions
    positions = compute_positions(full_ids, completion_start, tokenizer)

    # Forward pass with activation extraction
    t0 = time.time()
    activations = extract_activations_at_positions(
        full_ids,
        model,
        target_layers,
        positions,
        answer_start=positions["answer_start"],
        answer_end=positions["answer_end"],
    )
    extract_time = time.time() - t0

    # Build prompt_info dict compatible with save_generation
    save_info = {
        "prompt_id": prompt_info["prompt_id"],
        "question": prompt_info["messages"][-1]["content"],  # user message
        "label": prompt_info["label"],
        "global_pair_id": prompt_info["global_pair_id"],
        "question_type": prompt_info.get("question_type", "unknown"),
        "domain": prompt_info.get("category", "unknown"),
        "elicitation_style": "default",
        "similarity": True,
        "group": prompt_info.get("category", ""),
    }

    # Save generation JSON
    save_generation(
        save_info,
        full_text,
        "",  # no thinking trace
        assistant_content,
        positions,
        full_ids.shape[1] - completion_start,  # num "generated" tokens
        extract_time,
        output_dir,
    )

    # Save activation .pt files
    save_activations(save_info, activations, output_dir)

    return {
        "prompt_id": prompt_info["prompt_id"],
        "label": prompt_info["label"],
        "global_pair_id": prompt_info["global_pair_id"],
        "category": prompt_info.get("category", ""),
        "seq_len": full_ids.shape[1],
        "completion_start": completion_start,
        "completion_tokens": full_ids.shape[1] - completion_start,
        "positions": {k: int(v) for k, v in positions.items()},
        "extract_time": round(extract_time, 2),
    }


def generate_split_assignment(pairs: List[dict], output_path: str, all_test: bool = False):
    """
    Generate a split_assignment.yaml file.

    If all_test=True, all pairs go to test (for AI Liar cross-dataset eval).
    Otherwise, 80/10/10 train/val/test split.
    """
    import random

    assignments = {}

    if all_test:
        for pair in pairs:
            assignments[pair["pair_id"]] = "test"
        strategy = {"note": "All pairs assigned to test for cross-dataset evaluation"}
    else:
        random.seed(42)
        ids = [p["pair_id"] for p in pairs]
        random.shuffle(ids)
        n_train = int(len(ids) * 0.8)
        n_val = int(len(ids) * 0.1)
        for pid in sorted(ids[:n_train]):
            assignments[pid] = "train"
        for pid in sorted(ids[n_train:n_train + n_val]):
            assignments[pid] = "val"
        for pid in sorted(ids[n_train + n_val:]):
            assignments[pid] = "test"
        strategy = {"split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1}}

    split_data = {
        "description": "Pre-assigned splits for prewritten deception probes",
        "seed": 42,
        "strategy": strategy,
        "assignments": assignments,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(split_data, f, default_flow_style=False, sort_keys=False)

    # Count per split
    counts = {}
    for split in assignments.values():
        counts[split] = counts.get(split, 0) + 1
    print(f"  Split assignment saved to {output_path}")
    for split_name, count in sorted(counts.items()):
        print(f"    {split_name}: {count} pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from prewritten completions (no generation)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--dataset", type=str, required=True, choices=["roleplaying", "ai_liar"],
                        help="Which dataset to process")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs to process (for testing)")
    parser.add_argument("--neutral-control", action="store_true",
                        help="Neutral control mode: pair honest completions from different scenarios")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files")
    args = parser.parse_args()

    # Resolve paths
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)
    config = load_config(args.config)
    model_config_path = resolve_path(config["model_config"])
    model_config = load_config(model_config_path)

    output_dir = args.output_dir or config["storage"]["base_dir"]
    log_dir = args.log_dir or os.path.join(output_dir, "logs")
    setup_logging(log_dir, "extract_prewritten")

    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations", "tree"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations", "non_tree"), exist_ok=True)

    # Save config snapshot
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load dataset and build prompts
    print("=" * 60)
    print("Step 1: Loading dataset")
    print("=" * 60)

    # Load scenarios from committed data_dir (generated by convert_prewritten_data.py)
    data_dir = resolve_path(config["data_dir"])
    scenarios_path = os.path.join(data_dir, "scenarios.yaml")
    pairs = load_scenarios(scenarios_path)

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"  Limited to {args.max_pairs} pairs")

    if args.dataset == "roleplaying":
        prompts = build_roleplaying_prompts(pairs, neutral_control=args.neutral_control)

        # Generate split assignment if not in neutral-control mode
        if not args.neutral_control:
            split_path = os.path.join(data_dir, "split_assignment.yaml")
            if not os.path.exists(split_path):
                generate_split_assignment(pairs, split_path, all_test=False)

    elif args.dataset == "ai_liar":
        prompts = build_ai_liar_prompts(pairs)

        # Generate split assignment (all test for cross-dataset eval)
        split_path = os.path.join(data_dir, "split_assignment.yaml")
        if not os.path.exists(split_path):
            generate_split_assignment(pairs, split_path, all_test=True)

    mode_str = " (neutral control)" if args.neutral_control else ""
    print(f"  Dataset: {args.dataset}{mode_str}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Prompts: {len(prompts)} ({len(prompts)} forward passes)")

    # Check for completed work (resumability)
    log_path = os.path.join(output_dir, "prompts", "generation_log.jsonl")
    completed_ids = load_completed_ids(log_path)
    remaining = [p for p in prompts if p["prompt_id"] not in completed_ids]

    if completed_ids:
        print(f"  Already completed: {len(completed_ids)}")
        print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("  All prompts already processed. Nothing to do.")
        return

    # Load model
    print("\n" + "=" * 60)
    print("Step 2: Loading model")
    print("=" * 60)
    print(f"  Model: {model_config['model_id']}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Process prompts
    print("\n" + "=" * 60)
    print("Step 3: Forward pass extraction")
    print("=" * 60)
    print(f"  Target layers: {target_layers}")
    print(f"  Token positions: {config.get('token_positions', ['last_token', 'first_answer_sentence_end', 'answer_mean_pool'])}")
    print(f"  Remaining: {len(remaining)} prompts")

    total_done = 0
    total_tokens = 0
    start_time = time.time()

    for i, prompt_info in enumerate(remaining):
        try:
            log_entry = process_prompt(
                prompt_info, model, tokenizer, target_layers, output_dir,
            )

            # Append to log
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                f.flush()

            total_done += 1
            total_tokens += log_entry["seq_len"]

            # Progress report every 50 prompts
            if total_done % 50 == 0 or total_done == len(remaining):
                elapsed = time.time() - start_time
                rate = total_done / elapsed * 3600
                remaining_count = len(remaining) - total_done
                eta = remaining_count / (total_done / elapsed) if total_done > 0 else 0
                print(f"  {total_done}/{len(remaining)} done "
                      f"({rate:.0f}/hr, ETA: {eta / 60:.1f}min, "
                      f"last: {log_entry['prompt_id']} "
                      f"seq_len={log_entry['seq_len']} "
                      f"t={log_entry['extract_time']:.1f}s)")

        except Exception as e:
            print(f"  ERROR on {prompt_info['prompt_id']}: {e}")
            error_entry = {
                "prompt_id": prompt_info["prompt_id"],
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            error_log_path = os.path.join(output_dir, "prompts", "errors.jsonl")
            with open(error_log_path, "a") as f:
                f.write(json.dumps(error_entry) + "\n")

        # Clear CUDA cache periodically
        if total_done % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Processed: {total_done} prompts")
    print(f"  Total tokens processed: {total_tokens}")
    print(f"  Total time: {total_time / 60:.1f}min")
    print(f"  Effective rate: {total_done / total_time * 3600:.0f} prompts/hr")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
