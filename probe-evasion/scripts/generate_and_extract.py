"""
Generate QwQ-32B responses and extract multi-position activations.

For each contrastive question pair:
1. Format question as chat prompt
2. Generate full response with model.generate() (batched for throughput)
3. Find token positions (last_token, end_of_reasoning, first_answer_sentence_end, answer range)
4. Run one forward pass on the full output to extract activations at all positions
5. Save generation JSON + activation .pt files

Batched generation: prompts are grouped into batches (default 10) and generated
together. On L40S (48GB), KV cache for 10 sequences at 2048 tokens is ~5.4GB,
well within the ~28GB available after the 18GB quantized model. This gives ~10x
throughput on the memory-bandwidth-bound decode phase.

Extraction forward passes remain sequential (different sequence lengths).

Resumable: reads generation_log.jsonl on startup, skips completed prompt_ids.

Usage:
    python scripts/generate_and_extract.py \
        --config configs/experiments/qa_probe_training.yaml

    # Custom batch size:
    python scripts/generate_and_extract.py \
        --config configs/experiments/qa_probe_training.yaml \
        --batch-size 20

    # Quick test with 2 pairs:
    python scripts/generate_and_extract.py \
        --config configs/experiments/qa_probe_training.yaml \
        --max-pairs 2
"""

import argparse
import json
import os
import re
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
    find_token_positions, extract_activations_at_positions,
)
from src.utils.logging import setup_logging

# Elicitation style instructions for generation diversity
ELICITATION_INSTRUCTIONS = {
    "default": "Answer in 1-3 sentences.",
    "concise": "Be incredibly concise. Use 1-2 sentences maximum.",
    "verbose": "Be extremely verbose and detailed. Elaborate at length.",
    "eli5": "Explain as if to a 5-year-old. Use simple words.",
    "numbered_list": "Structure your answer as a numbered list.",
    "explicit_mention": "Answer without directly naming the core concept. Answer in 1-3 sentences.",
    "no_keyword": "Answer without using the word 'tree' or 'trees'. Answer in 1-3 sentences.",
    "academic": "Answer in formal academic style with precise terminology.",
    "casual": "Answer casually, as if chatting with a friend.",
}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_contrastive_pairs(data_dir: str) -> List[dict]:
    """
    Load all contrastive pairs from YAML batch files.

    Args:
        data_dir: Directory containing contrastive_batch_*.yaml files.

    Returns:
        List of pair dicts with global_pair_id added.
    """
    data_path = Path(data_dir)
    batch_files = sorted(data_path.glob("contrastive_batch_*.yaml"))

    if not batch_files:
        raise FileNotFoundError(f"No contrastive_batch_*.yaml files found in {data_dir}")

    all_pairs = []
    global_id = 0

    for batch_file in batch_files:
        with open(batch_file, "r") as f:
            data = yaml.safe_load(f)

        pairs = data.get("pairs", [])
        batch_name = batch_file.stem

        for pair in pairs:
            # Use pair_id if present (v2 data), otherwise auto-increment (v1 compat)
            pair["global_pair_id"] = pair.get("pair_id", global_id)
            pair["source_batch"] = batch_name
            all_pairs.append(pair)
            global_id += 1

    print(f"Loaded {len(all_pairs)} contrastive pairs from {len(batch_files)} batch files")
    return all_pairs


def build_prompt_list(pairs: List[dict]) -> List[dict]:
    """
    Expand contrastive pairs into a flat list of prompts.

    Each pair produces two prompts: one tree, one non-tree.

    Returns:
        List of dicts with keys: prompt_id, question, label, global_pair_id, question_type, domain
    """
    prompts = []
    for pair in pairs:
        gid = pair["global_pair_id"]

        elicitation_style = pair.get("elicitation_style", "default")
        similarity = pair.get("similarity", False)

        group = pair.get("group", "")
        tree_topic = pair.get("tree_topic", "")
        base_pair_ref = pair.get("base_pair_ref")

        prompts.append({
            "prompt_id": f"tree_{gid:04d}",
            "question": pair["tree_question"],
            "label": "tree",
            "global_pair_id": gid,
            "question_type": pair.get("question_type", "unknown"),
            "domain": "trees",
            "elicitation_style": elicitation_style,
            "similarity": similarity,
            "group": group,
            "tree_topic": tree_topic,
            "base_pair_ref": base_pair_ref,
        })
        prompts.append({
            "prompt_id": f"non_tree_{gid:04d}",
            "question": pair["non_tree_question"],
            "label": "non_tree",
            "global_pair_id": gid,
            "question_type": pair.get("question_type", "unknown"),
            "domain": pair.get("non_tree_domain", "unknown"),
            "elicitation_style": elicitation_style,
            "similarity": similarity,
            "group": group,
            "tree_topic": tree_topic,
            "base_pair_ref": base_pair_ref,
        })

    return prompts


def load_supplementary_examples(yaml_path: str) -> List[dict]:
    """
    Load supplementary examples (handwritten validation or adversarial test set).

    Reads a YAML file with 'examples' key containing items with id, question, label, category.
    Returns list of prompt dicts compatible with the main pipeline.
    """
    prompts = []
    for item in _parse_supplementary_yaml(yaml_path):
        label = "tree" if item.get("label_int", 0) == 1 else "non_tree"
        prompts.append({
            "prompt_id": item["id"],
            "question": item["question"],
            "label": label,
            "global_pair_id": None,
            "question_type": item.get("category", "unknown"),
            "domain": "supplementary",
            "elicitation_style": "default",
            "similarity": False,
        })

    print(f"  Loaded {len(prompts)} supplementary examples from {yaml_path}")
    return prompts


def _parse_supplementary_yaml(yaml_path: str) -> List[dict]:
    """Parse supplementary YAML without pyyaml dependency (for local dev)."""
    with open(yaml_path, "r") as f:
        text = f.read()

    items = []
    current = {}

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current:
                items.append(current)
            current = {"id": stripped.split(":", 1)[1].strip().strip('"')}
        elif stripped.startswith("question:") and current:
            val = stripped.split(":", 1)[1].strip()
            # Handle quoted strings
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            current["question"] = val
        elif stripped.startswith("label:") and current:
            val = stripped.split(":", 1)[1].strip()
            current["label_int"] = int(val)
        elif stripped.startswith("category:") and current:
            current["category"] = stripped.split(":", 1)[1].strip()

    if current:
        items.append(current)

    return items


def load_completed_ids(log_path: str) -> set:
    """Load set of completed prompt_ids from generation log."""
    completed = set()
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        completed.add(entry["prompt_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def save_generation(
    prompt_info: dict,
    full_text: str,
    thinking: str,
    answer: str,
    positions: dict,
    num_generated_tokens: int,
    elapsed: float,
    output_dir: str,
) -> str:
    """Save generation metadata as JSON. Returns the file path."""
    label = prompt_info["label"]
    prompt_id = prompt_info["prompt_id"]

    gen_dir = os.path.join(output_dir, "generations", label)
    os.makedirs(gen_dir, exist_ok=True)

    gen_data = {
        "prompt_id": prompt_id,
        "question": prompt_info["question"],
        "label": label,
        "global_pair_id": prompt_info.get("global_pair_id"),
        "question_type": prompt_info.get("question_type", "unknown"),
        "domain": prompt_info.get("domain", "unknown"),
        "elicitation_style": prompt_info.get("elicitation_style", "default"),
        "similarity": prompt_info.get("similarity", False),
        "group": prompt_info.get("group", ""),
        "tree_topic": prompt_info.get("tree_topic", ""),
        "base_pair_ref": prompt_info.get("base_pair_ref"),
        "full_text": full_text,
        "thinking_trace": thinking,
        "answer": answer,
        "num_generated_tokens": num_generated_tokens,
        "token_positions": {k: int(v) for k, v in positions.items()},
        "elapsed_seconds": round(elapsed, 2),
    }

    path = os.path.join(gen_dir, f"{prompt_id}.json")
    with open(path, "w") as f:
        json.dump(gen_data, f, indent=2)

    return path


def save_activations(
    prompt_info: dict,
    activations: Dict[str, Dict[int, torch.Tensor]],
    output_dir: str,
) -> List[str]:
    """Save activation tensors as .pt files. Returns list of file paths."""
    label = prompt_info["label"]
    prompt_id = prompt_info["prompt_id"]
    saved_paths = []

    for pos_name, layer_acts in activations.items():
        act_dir = os.path.join(output_dir, "activations", label, pos_name)
        os.makedirs(act_dir, exist_ok=True)

        path = os.path.join(act_dir, f"{prompt_id}.pt")
        torch.save(layer_acts, path)
        saved_paths.append(path)

    return saved_paths


def extract_answer(full_text: str) -> str:
    """Extract the answer portion from QwQ output (after </think> tag)."""
    match = re.search(r"</think>\s*", full_text)
    if match:
        return full_text[match.end():].strip()
    return full_text.strip()


def extract_thinking(full_text: str) -> str:
    """Extract the thinking trace from QwQ output.

    Handles two cases:
    - Full text contains both <think> and </think> tags
    - Generated text only (where <think> was in the prompt): everything
      before </think> is the thinking trace
    """
    match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"</think>", full_text)
    if match:
        return full_text[:match.start()].strip()
    return ""


def format_chat_prompt(question: str, tokenizer, elicitation_style: str = "default") -> str:
    """Format a question as a chat prompt string with style-specific instruction."""
    instruction = ELICITATION_INSTRUCTIONS.get(elicitation_style, ELICITATION_INSTRUCTIONS["default"])
    prompt_text = f"{question}\n\n{instruction}"
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_batch(
    prompt_infos: List[dict],
    model,
    tokenizer,
    generation_config: dict,
) -> Tuple[List[torch.Tensor], List[int], float]:
    """
    Generate responses for a batch of prompts simultaneously.

    Args:
        prompt_infos: List of prompt info dicts.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        generation_config: Generation parameters.

    Returns:
        Tuple of (output_ids_list, input_lens, gen_time) where:
        - output_ids_list: list of (1, seq_len) tensors per sequence (unpadded)
        - input_lens: list of prompt token counts
        - gen_time: total generation wall time
    """
    # Format all prompts (with per-prompt elicitation style)
    chat_prompts = [
        format_chat_prompt(p["question"], tokenizer, p.get("elicitation_style", "default"))
        for p in prompt_infos
    ]

    # Tokenize with left-padding for batched generation
    # (causal LMs need left-padding so the final tokens are aligned)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    tokenizer.padding_side = original_padding_side

    # Track per-sequence input lengths (unpadded)
    # With left-padding, real tokens start after the padding
    attention_mask = inputs["attention_mask"]
    input_lens = [int(attention_mask[i].sum().item()) for i in range(len(prompt_infos))]

    # Generate
    t0 = time.time()
    with torch.no_grad():
        output_ids_batch = model.generate(
            **inputs,
            max_new_tokens=generation_config.get("max_new_tokens", 2048),
            temperature=generation_config.get("temperature", 0.6),
            top_p=generation_config.get("top_p", 0.95),
            top_k=generation_config.get("top_k", 20),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0

    # Unpad each sequence: strip left-padding to get clean (1, real_seq_len) tensors
    # The padded input length is inputs["input_ids"].shape[1]
    padded_input_len = inputs["input_ids"].shape[1]
    output_ids_list = []

    for i in range(len(prompt_infos)):
        # Padding tokens are at the start (left-padded)
        pad_count = padded_input_len - input_lens[i]
        # The output for sequence i starts at pad_count
        real_output = output_ids_batch[i, pad_count:].unsqueeze(0)  # (1, real_seq_len)

        # Strip trailing padding: find first EOS in generated portion
        # (pad_token == eos_token == <|im_end|>, so trailing pads are
        # indistinguishable — keep only through the first one)
        gen_portion = real_output[0, input_lens[i]:]
        eos_positions = (gen_portion == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 0:
            first_eos = input_lens[i] + eos_positions[0].item()
            real_output = real_output[:, :first_eos + 1]

        output_ids_list.append(real_output)

    return output_ids_list, input_lens, gen_time


def process_generated_sequence(
    prompt_info: dict,
    output_ids: torch.Tensor,
    input_len: int,
    model,
    tokenizer,
    target_layers: List[int],
    output_dir: str,
) -> dict:
    """
    Process a single generated sequence: decode, find positions, extract activations, save.

    Args:
        prompt_info: Prompt metadata dict.
        output_ids: Unpadded output tensor (1, seq_len).
        input_len: Number of prompt tokens.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        target_layers: Layer indices to extract from.
        output_dir: Base output directory.

    Returns:
        Dict with generation metadata for the log.
    """
    # Decode
    generated_ids = output_ids[0][input_len:]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    thinking = extract_thinking(full_text)
    answer = extract_answer(full_text)
    num_generated_tokens = len(generated_ids)

    # Find token positions
    positions = find_token_positions(output_ids, input_len, tokenizer)

    # Post-generation forward pass for multi-position activation extraction
    t0 = time.time()
    activations = extract_activations_at_positions(
        output_ids,
        model,
        target_layers,
        positions,
        answer_start=positions.get("answer_start"),
        answer_end=positions.get("answer_end"),
    )
    extract_time = time.time() - t0

    # Save generation JSON
    save_generation(
        prompt_info, full_text, thinking, answer,
        positions, num_generated_tokens, extract_time, output_dir,
    )

    # Save activation .pt files
    save_activations(prompt_info, activations, output_dir)

    return {
        "prompt_id": prompt_info["prompt_id"],
        "label": prompt_info["label"],
        "global_pair_id": prompt_info["global_pair_id"],
        "num_generated_tokens": num_generated_tokens,
        "thinking_tokens": len(tokenizer.encode(thinking)) if thinking else 0,
        "answer_tokens": len(tokenizer.encode(answer)) if answer else 0,
        "positions": {k: int(v) for k, v in positions.items()},
        "extract_time": round(extract_time, 2),
    }


def reextract_activations(output_dir: str, target_layers: List[int], model_config: dict, config: dict):
    """
    Re-extract activations from existing generation JSONs.

    Loads each saved generation, re-tokenizes the full sequence (prompt + generated),
    runs a forward pass with hooks on target_layers, and saves new activation .pt files.
    This is useful when you want to extract activations for more layers without re-generating.
    """
    from glob import glob

    # Find all generation JSONs
    gen_files = sorted(
        glob(os.path.join(output_dir, "generations", "tree", "*.json"))
        + glob(os.path.join(output_dir, "generations", "non_tree", "*.json"))
    )
    if not gen_files:
        print("No generation files found. Run without --reextract first.")
        return

    print("=" * 60)
    print("RE-EXTRACTION MODE")
    print("=" * 60)
    print(f"  Found {len(gen_files)} generation files")
    print(f"  Target layers: {len(target_layers)} layers")
    print(f"  Output: {output_dir}")

    # Load model
    print("\n  Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    done = 0
    errors = 0
    start_time = time.time()

    for gen_path in gen_files:
        with open(gen_path, "r") as f:
            gen_data = json.load(f)

        prompt_id = gen_data["prompt_id"]
        label = gen_data["label"]
        question = gen_data["question"]
        style = gen_data.get("elicitation_style", "default")
        thinking = gen_data.get("thinking_trace", "")
        answer = gen_data.get("answer", "")

        try:
            # Reconstruct chat prompt
            chat_prompt = format_chat_prompt(question, tokenizer, style)

            # Reconstruct generated text with think tags
            if thinking:
                generated_text = f"<think>\n{thinking}\n</think>\n{answer}"
            else:
                generated_text = answer

            # Tokenize prompt alone to get input_len
            prompt_ids = tokenizer(chat_prompt, return_tensors="pt")["input_ids"]
            input_len = prompt_ids.shape[1]

            # Tokenize full sequence (prompt + generated)
            full_text = chat_prompt + generated_text
            output_ids = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"].to(model.device)

            # Find token positions
            positions = find_token_positions(output_ids, input_len, tokenizer)

            # Forward pass with hooks
            activations = extract_activations_at_positions(
                output_ids, model, target_layers, positions,
                answer_start=positions.get("answer_start"),
                answer_end=positions.get("answer_end"),
            )

            # Save activations (overwrites old .pt files)
            prompt_info = {"prompt_id": prompt_id, "label": label}
            save_activations(prompt_info, activations, output_dir)

            done += 1
            if done % 50 == 0 or done == len(gen_files):
                elapsed = time.time() - start_time
                rate = done / elapsed * 3600
                remaining = (len(gen_files) - done) / (done / elapsed) if done > 0 else 0
                print(f"  {done}/{len(gen_files)} done ({rate:.0f}/hr, ETA: {remaining/60:.1f}min)")

        except Exception as e:
            print(f"  ERROR on {prompt_id}: {e}")
            errors += 1

        # Clear CUDA cache periodically
        if done % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\n  Re-extraction complete: {done} succeeded, {errors} errors, {total_time/60:.1f}min")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QwQ-32B responses and extract multi-position activations"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to qa_probe_training.yaml config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of prompts to generate simultaneously (default: 10)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: output-dir/logs)")
    parser.add_argument("--supplementary-dir", type=str, default=None,
                        help="Directory containing handwritten_validation.yaml and adversarial_test_set.yaml")
    parser.add_argument("--reextract", action="store_true",
                        help="Re-extract activations from existing generations (skips generation)")
    args = parser.parse_args()

    # Resolve paths
    def resolve_path(p):
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    args.config = resolve_path(args.config)

    # Load configs
    config = load_config(args.config)
    model_config_path = resolve_path(config["model_config"])
    model_config = load_config(model_config_path)

    data_dir = resolve_path(config["data_dir"])

    # Set up logging
    output_dir = args.output_dir or config["storage"]["base_dir"]
    log_dir = args.log_dir or os.path.join(output_dir, "logs")
    setup_logging(log_dir, "generate_and_extract")
    target_layers = config["target_layers"]
    if target_layers == "all":
        target_layers = list(range(model_config["num_layers"]))
    generation_config = config.get("generation", {})
    output_dir = args.output_dir or config["storage"]["base_dir"]

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations", "tree"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generations", "non_tree"), exist_ok=True)

    # Save config snapshot
    config_snapshot_path = os.path.join(output_dir, "config.json")
    with open(config_snapshot_path, "w") as f:
        json.dump(config, f, indent=2)

    # --reextract mode: skip generation, re-run extraction on existing outputs
    if args.reextract:
        reextract_activations(output_dir, target_layers, model_config, config)
        return

    # Load contrastive pairs
    print("=" * 60)
    print("Step 1: Loading contrastive pairs")
    print("=" * 60)
    pairs = load_contrastive_pairs(data_dir)

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"  Limited to {args.max_pairs} pairs ({args.max_pairs * 2} prompts)")

    prompts = build_prompt_list(pairs)
    print(f"  QA prompts: {len(prompts)}")

    # Load supplementary examples if provided
    if args.supplementary_dir:
        supp_dir = Path(resolve_path(args.supplementary_dir))
        for supp_file in ["handwritten_validation.yaml", "adversarial_test_set.yaml"]:
            supp_path = supp_dir / supp_file
            if supp_path.exists():
                supp_prompts = load_supplementary_examples(str(supp_path))
                prompts.extend(supp_prompts)
            else:
                print(f"  WARNING: Supplementary file not found: {supp_path}")

    print(f"  Total prompts: {len(prompts)}")

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

    # Verify layer count
    num_layers = len(model.model.layers)
    assert num_layers == model_config["num_layers"], \
        f"Layer mismatch: {num_layers} vs {model_config['num_layers']}"

    # VRAM budget check
    batch_size = args.batch_size
    max_new_tokens = generation_config.get("max_new_tokens", 2048)
    # KV cache: 2 * 64 layers * 8 kv_heads * 128 head_dim * 2 bytes = 256KB/token
    kv_per_seq_gb = (max_new_tokens + 100) * 256 * 1024 / (1024**3)
    total_kv_gb = batch_size * kv_per_seq_gb
    print(f"\n  Batch size: {batch_size}")
    print(f"  KV cache estimate: {kv_per_seq_gb:.1f} GB/seq × {batch_size} = {total_kv_gb:.1f} GB")

    # Process prompts in batches
    print("\n" + "=" * 60)
    print("Step 3: Generating and extracting (batched)")
    print("=" * 60)
    print(f"  Target layers: {target_layers}")
    print(f"  Token positions: {config['token_positions']}")
    print(f"  Generation: max_new_tokens={max_new_tokens}, "
          f"temp={generation_config.get('temperature', 0.6)}")
    print(f"  Batch size: {batch_size}")

    total_gen_tokens = 0
    total_prompts_done = 0
    start_time = time.time()

    # Split remaining into batches
    batches = [
        remaining[i:i + batch_size]
        for i in range(0, len(remaining), batch_size)
    ]
    print(f"  {len(batches)} batches of up to {batch_size} prompts")

    for batch_idx, batch_prompts in enumerate(batches):
        batch_start = time.time()
        batch_ids = [p["prompt_id"] for p in batch_prompts]
        print(f"\n  Batch {batch_idx+1}/{len(batches)} ({len(batch_prompts)} prompts)")
        print(f"    IDs: {batch_ids[0]} ... {batch_ids[-1]}")

        try:
            # Batched generation
            output_ids_list, input_lens, gen_time = generate_batch(
                batch_prompts, model, tokenizer, generation_config,
            )
            print(f"    Generation: {gen_time:.1f}s for {len(batch_prompts)} prompts "
                  f"({gen_time/len(batch_prompts):.1f}s/prompt)")

            # Sequential extraction + saving for each sequence
            extract_start = time.time()
            for i, prompt_info in enumerate(batch_prompts):
                try:
                    log_entry = process_generated_sequence(
                        prompt_info,
                        output_ids_list[i],
                        input_lens[i],
                        model,
                        tokenizer,
                        target_layers,
                        output_dir,
                    )
                    log_entry["gen_time"] = round(gen_time / len(batch_prompts), 2)
                    log_entry["total_time"] = round(
                        gen_time / len(batch_prompts) + log_entry["extract_time"], 2
                    )

                    # Append to log
                    with open(log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                        f.flush()

                    total_gen_tokens += log_entry["num_generated_tokens"]
                    total_prompts_done += 1

                except Exception as e:
                    print(f"    ERROR on {prompt_info['prompt_id']}: {e}")
                    error_entry = {
                        "prompt_id": prompt_info["prompt_id"],
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    error_log_path = os.path.join(output_dir, "prompts", "errors.jsonl")
                    with open(error_log_path, "a") as f:
                        f.write(json.dumps(error_entry) + "\n")

            extract_time = time.time() - extract_start
            batch_time = time.time() - batch_start

            # Progress report
            elapsed = time.time() - start_time
            prompts_per_hr = total_prompts_done / elapsed * 3600 if elapsed > 0 else 0
            remaining_prompts = len(remaining) - total_prompts_done
            eta_hours = remaining_prompts / prompts_per_hr if prompts_per_hr > 0 else 0

            print(f"    Extraction: {extract_time:.1f}s ({extract_time/len(batch_prompts):.1f}s/prompt)")
            print(f"    Batch total: {batch_time:.1f}s "
                  f"({batch_time/len(batch_prompts):.1f}s/prompt effective)")
            print(f"    Progress: {total_prompts_done}/{len(remaining)} done, "
                  f"{prompts_per_hr:.0f} prompts/hr, ETA: {eta_hours:.1f}h")

        except Exception as e:
            print(f"    BATCH ERROR: {e}")
            # Fall back to sequential processing for this batch
            print(f"    Falling back to sequential processing...")
            for prompt_info in batch_prompts:
                if prompt_info["prompt_id"] in load_completed_ids(log_path):
                    continue
                try:
                    log_entry = process_single_prompt(
                        prompt_info, model, tokenizer, target_layers,
                        generation_config, output_dir,
                    )
                    with open(log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                        f.flush()
                    total_gen_tokens += log_entry["num_generated_tokens"]
                    total_prompts_done += 1
                except Exception as e2:
                    print(f"      ERROR on {prompt_info['prompt_id']}: {e2}")
                    error_entry = {
                        "prompt_id": prompt_info["prompt_id"],
                        "error": str(e2),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    error_log_path = os.path.join(output_dir, "prompts", "errors.jsonl")
                    with open(error_log_path, "a") as f:
                        f.write(json.dumps(error_entry) + "\n")

        # Clear CUDA cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Processed: {total_prompts_done} prompts")
    print(f"  Total generated tokens: {total_gen_tokens}")
    print(f"  Total time: {total_time / 3600:.2f}h")
    print(f"  Effective rate: {total_prompts_done / total_time * 3600:.0f} prompts/hr")
    print(f"  Output: {output_dir}")


def process_single_prompt(
    prompt_info: dict,
    model,
    tokenizer,
    target_layers: List[int],
    generation_config: dict,
    output_dir: str,
) -> dict:
    """
    Fallback: generate and extract for a single prompt (no batching).

    Used when batched generation fails for a batch.
    """
    question = prompt_info["question"]
    style = prompt_info.get("elicitation_style", "default")
    instruction = ELICITATION_INSTRUCTIONS.get(style, ELICITATION_INSTRUCTIONS["default"])
    prompt_text = f"{question}\n\n{instruction}"
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config.get("max_new_tokens", 2048),
            temperature=generation_config.get("temperature", 0.6),
            top_p=generation_config.get("top_p", 0.95),
            top_k=generation_config.get("top_k", 20),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0

    log_entry = process_generated_sequence(
        prompt_info, output_ids, input_len,
        model, tokenizer, target_layers, output_dir,
    )
    log_entry["gen_time"] = round(gen_time, 2)
    log_entry["total_time"] = round(gen_time + log_entry["extract_time"], 2)

    return log_entry


if __name__ == "__main__":
    main()
