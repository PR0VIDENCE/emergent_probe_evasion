"""
Extract activations from stitched reasoning traces + completions.

Takes prewritten reasoning traces (from generate_reasoning_traces.py or
handwritten/Claude-generated) and pairs them with fixed deceptive/honest
completions. Runs forward passes and extracts activations at answer
positions ONLY (not reasoning positions).

This tests whether the content of the reasoning trace affects probe
activations in the generation, even though the generation text is identical.

Extraction positions (all within the answer, after </think>):
  - last_token
  - first_answer_sentence_end
  - answer_mean_pool

Usage:
    uv run python scripts/extract_reasoning_trace_activations.py --config configs/experiments/deception_prewritten.yaml --traces data/concepts/reasoning_traces/qwq_traces.yaml
    uv run python scripts/extract_reasoning_trace_activations.py --config configs/experiments/deception_prewritten.yaml --traces data/concepts/reasoning_traces/claude_traces_batch1.yaml
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.extract_activations import (
    load_model_and_tokenizer,
    extract_activations_at_positions,
)
from src.utils.logging import setup_logging
from scripts.generate_and_extract import save_activations, load_completed_ids

CONDITIONS_DECEPTIVE = [
    "naked_deception", "self_deceptive", "procedural", "random", "empty",
    "honest_reasoning_deceptive", "domain_shift", "compartmentalized",
    "moral_reframing", "abstracted", "empathy_anchored",
]
CONDITIONS_HONEST = ["honest_control"]
ALL_CONDITIONS = CONDITIONS_DECEPTIVE + CONDITIONS_HONEST


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_scenarios(path: str) -> List[dict]:
    with open(path, "r") as f:
        return yaml.safe_load(f)["scenarios"]


def load_traces(path: str) -> Dict[int, dict]:
    """Load reasoning traces, indexed by pair_id."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {t["pair_id"]: t for t in data["traces"]}


def find_first_sentence_end(token_ids, start_pos, tokenizer):
    for pos in range(start_pos, token_ids.shape[1]):
        token_text = tokenizer.decode(token_ids[0, pos:pos + 1])
        if any(ch in token_text for ch in ".!?"):
            return pos
    return None


def build_messages_with_reasoning(scenario, reasoning_text, completion_text, answer_prefix):
    """Build chat messages with <think> reasoning + completion as assistant content."""
    if reasoning_text:
        assistant_content = f"<think>\n{reasoning_text}\n</think>\n{answer_prefix} {completion_text}"
    else:
        # Empty reasoning condition
        assistant_content = f"<think>\n</think>\n{answer_prefix} {completion_text}"

    messages = [
        {"role": "system", "content": scenario["scenario"]},
        {"role": "user", "content": scenario["question"]},
    ]
    return messages, assistant_content


def process_single(
    prompt_id: str,
    label: str,
    messages: list,
    assistant_content: str,
    model,
    tokenizer,
    target_layers: List[int],
    output_dir: str,
    metadata: dict,
) -> dict:
    """Tokenize, forward pass, extract activations at answer positions only."""

    # Tokenize prompt (system + user) to find where assistant starts
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Tokenize full sequence
    full_messages = messages + [{"role": "assistant", "content": assistant_content}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    seq_len = full_ids.shape[1]

    # Find </think> boundary to get answer_start
    assistant_start = len(prompt_ids)
    generated_text = tokenizer.decode(full_ids[0, assistant_start:], skip_special_tokens=False)
    think_end_match = re.search(r"</think>", generated_text)

    if think_end_match:
        text_before = generated_text[:think_end_match.start()]
        tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
        think_tag_tokens = tokenizer.encode("</think>", add_special_tokens=False)

        answer_start_offset = len(tokens_before) + len(think_tag_tokens)
        # Skip whitespace after </think>
        text_after = generated_text[think_end_match.end():]
        stripped = text_after.lstrip()
        ws_chars = len(text_after) - len(stripped)
        if ws_chars > 0:
            ws_tokens = tokenizer.encode(text_after[:ws_chars], add_special_tokens=False)
            answer_start_offset += len(ws_tokens)

        answer_start = assistant_start + answer_start_offset
    else:
        # No think tags (shouldn't happen but fallback)
        answer_start = assistant_start

    # Compute positions (answer-only)
    positions = {
        "last_token": seq_len - 1,
        "first_answer_sentence_end": None,
        "answer_start": min(answer_start, seq_len - 1),
        "answer_end": seq_len - 1,
    }

    sent_end = find_first_sentence_end(full_ids, answer_start, tokenizer)
    positions["first_answer_sentence_end"] = sent_end if sent_end is not None else positions["last_token"]

    # Forward pass
    t0 = time.time()
    activations = extract_activations_at_positions(
        full_ids, model, target_layers, positions,
        answer_start=positions["answer_start"],
        answer_end=positions["answer_end"],
    )
    extract_time = time.time() - t0

    # Save activations
    save_info = {"prompt_id": prompt_id, "label": label}
    save_activations(save_info, activations, output_dir)

    # Save generation JSON
    gen_dir = os.path.join(output_dir, "generations", label)
    os.makedirs(gen_dir, exist_ok=True)
    gen_data = {
        "prompt_id": prompt_id,
        "label": label,
        "global_pair_id": metadata.get("pair_id"),
        "condition": metadata.get("condition"),
        "category": metadata.get("category"),
        "question": messages[-1]["content"],
        "reasoning_trace": metadata.get("reasoning_text", ""),
        "answer": metadata.get("completion_text", ""),
        "full_text": full_text,
        "seq_len": seq_len,
        "answer_start": answer_start,
        "token_positions": {k: int(v) for k, v in positions.items()},
        "extract_time": round(extract_time, 2),
    }
    with open(os.path.join(gen_dir, f"{prompt_id}.json"), "w") as f:
        json.dump(gen_data, f, indent=2)

    return {
        "prompt_id": prompt_id,
        "label": label,
        "global_pair_id": metadata.get("pair_id"),
        "condition": metadata.get("condition"),
        "category": metadata.get("category"),
        "seq_len": seq_len,
        "answer_start": answer_start,
        "completion_tokens": seq_len - answer_start,
        "positions": {k: int(v) for k, v in positions.items()},
        "extract_time": round(extract_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from stitched reasoning traces + completions"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--traces", type=str, required=True,
                        help="Path to reasoning traces YAML")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Which conditions to extract (default: all present in traces file)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore generation log and rerun all forward passes")
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    config = load_config(resolve_path(args.config))
    model_config = load_config(resolve_path(config["model_config"]))

    output_dir = args.output_dir or os.path.join(config["storage"]["base_dir"], "reasoning_traces")
    log_dir = args.log_dir or os.path.join(output_dir, "logs")
    setup_logging(log_dir, "extract_reasoning_traces")

    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))

    # Load data
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)

    data_dir = resolve_path(config["data_dir"])
    scenarios = load_scenarios(os.path.join(data_dir, "scenarios.yaml"))
    traces = load_traces(resolve_path(args.traces))

    # Filter to scenarios that have traces
    scenarios = [s for s in scenarios if s["pair_id"] in traces]
    if args.max_pairs:
        scenarios = scenarios[:args.max_pairs]

    # Determine which conditions to run
    sample_trace = next(iter(traces.values()))
    available_conditions = [c for c in ALL_CONDITIONS if c in sample_trace]
    conditions = args.conditions or available_conditions
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Conditions: {conditions}")
    print(f"  Total forward passes: {len(scenarios) * len(conditions)}")

    # Setup output dirs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations", "tree"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations", "non_tree"), exist_ok=True)

    # Check for completed work
    log_path = os.path.join(output_dir, "prompts", "generation_log.jsonl")
    completed_ids = load_completed_ids(log_path)
    print(f"  Already completed: {len(completed_ids)}")

    # Load model
    print(f"\n  Loading model: {model_config['model_id']}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Process
    print("\n" + "=" * 60)
    print("Step 2: Forward pass extraction")
    print("=" * 60)

    total_done = 0
    start_time = time.time()

    for scenario in scenarios:
        pid = scenario["pair_id"]
        trace_data = traces[pid]

        for condition in conditions:
            if condition not in trace_data:
                continue

            reasoning_text = trace_data[condition]

            # Determine label and completion
            if condition in CONDITIONS_HONEST:
                label = "non_tree"
                completion_text = scenario["honest_completion"]
            else:
                label = "tree"
                completion_text = scenario["deceptive_completion"]

            prompt_id = f"{label}_{pid:04d}_{condition}"

            if prompt_id in completed_ids:
                continue

            try:
                messages, assistant_content = build_messages_with_reasoning(
                    scenario, reasoning_text, completion_text, scenario["answer_prefix"]
                )

                log_entry = process_single(
                    prompt_id, label, messages, assistant_content,
                    model, tokenizer, target_layers, output_dir,
                    metadata={
                        "pair_id": pid,
                        "condition": condition,
                        "category": scenario["category"],
                        "reasoning_text": reasoning_text,
                        "completion_text": completion_text,
                    },
                )

                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()

                total_done += 1

                if total_done % 20 == 0 or total_done == 1:
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed * 3600
                    print(f"  {total_done} done ({rate:.0f}/hr, "
                          f"last: {prompt_id} seq={log_entry['seq_len']} "
                          f"ans_start={log_entry['answer_start']} "
                          f"t={log_entry['extract_time']:.1f}s)")

            except Exception as e:
                print(f"  ERROR {prompt_id}: {e}")
                error_path = os.path.join(output_dir, "prompts", "errors.jsonl")
                with open(error_path, "a") as f:
                    f.write(json.dumps({"prompt_id": prompt_id, "error": str(e)}) + "\n")

            if total_done % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE: {total_done} forward passes in {total_time/60:.1f}min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
