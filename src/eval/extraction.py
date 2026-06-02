"""
Activation extraction for the eval pipeline.

Re-extraction context (concept_config["extraction"]["context_handler"]):
  - identity        : forward-pass under the system prompt the rollout was generated with
  - neutral_system  : forward-pass under a neutral "You are a helpful assistant." prompt
                      (sycophancy — removes the system-prompt-text confound from the probe)
  - scenario_system : forward-pass under the bare scenario (deception — strips any
                      evasion addendum so the probe sees the same context as training)

Output per rollout: a dict {position_name: {layer_idx: tensor(hidden,)}} saved as
<rollout_id>.pt, plus an extraction_log.jsonl row. Mirrors the existing
extract_prewritten_activations / multiplication_extract_activations layout so
downstream scoring is uniform.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch

from src.inference.extract_activations import extract_activations_at_positions
from . import reasoning_formats

NEUTRAL_PROMPT = "You are a helpful assistant."


def resolve_extraction_system_prompt(rollout: dict, cc: dict) -> str:
    handler = cc.get("extraction", {}).get("context_handler", "identity")
    if handler == "identity":
        return rollout["input"]["system_prompt"]
    if handler == "neutral_system":
        return cc.get("extraction", {}).get("neutral_prompt", NEUTRAL_PROMPT)
    if handler == "scenario_system":
        return rollout["metadata"].get("scenario", rollout["input"]["system_prompt"])
    raise ValueError(f"Unknown extraction context_handler: {handler}")


def _build_assistant_content(rollout: dict, fmt) -> str:
    out = rollout["output"]
    thinking = (out.get("thinking") or "").strip()
    response = (out.get("response") or "").strip()
    return fmt.reconstruct(thinking, response)


def _compute_positions(full_ids, completion_start: int, tokenizer, fmt) -> dict:
    """Absolute token positions for {last_token, end_of_reasoning,
    first_answer_sentence_end, answer_start/end, reasoning_start/end}.

    The reasoning→answer boundary is located by the model's reasoning-format
    handler (think_tags / harmony / none). Missing positions fall back to
    last_token, matching find_token_positions' contract.
    """
    seq_len = full_ids.shape[1]
    positions = {"last_token": seq_len - 1, "answer_end": seq_len - 1}

    completion_ids = full_ids[0, completion_start:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    boundary = fmt.boundary(completion_text)

    if boundary:
        # Tokens up to and including the boundary marker = end of reasoning.
        before_answer = completion_text[:boundary.end()]
        eor = len(tokenizer.encode(before_answer, add_special_tokens=False)) - 1
        positions["end_of_reasoning"] = min(completion_start + eor, seq_len - 1)
        positions["reasoning_start"] = completion_start
        positions["reasoning_end"] = positions["end_of_reasoning"]

        answer_offset = eor + 1
        text_after = completion_text[boundary.end():]
        stripped = text_after.lstrip()
        ws_len = len(text_after) - len(stripped)
        if ws_len > 0:
            ws_tokens = tokenizer.encode(text_after[:ws_len], add_special_tokens=False)
            answer_offset += len(ws_tokens)
        positions["answer_start"] = min(completion_start + answer_offset, seq_len - 1)

        sent = re.search(r"[.!?]", text_after)
        if sent:
            sent_tokens = tokenizer.encode(text_after[:sent.end()], add_special_tokens=False)
            positions["first_answer_sentence_end"] = min(
                completion_start + answer_offset + len(sent_tokens) - 1, seq_len - 1)
        else:
            positions["first_answer_sentence_end"] = positions["last_token"]
    else:
        # No reasoning trace: the whole completion IS the answer.
        positions["end_of_reasoning"] = positions["last_token"]
        positions["reasoning_start"] = completion_start
        positions["reasoning_end"] = positions["last_token"]
        positions["answer_start"] = completion_start
        # First sentence end within the completion.
        sent = re.search(r"[.!?]", completion_text)
        if sent:
            sent_tokens = tokenizer.encode(completion_text[:sent.end()], add_special_tokens=False)
            positions["first_answer_sentence_end"] = min(
                completion_start + len(sent_tokens) - 1, seq_len - 1)
        else:
            positions["first_answer_sentence_end"] = positions["last_token"]

    return positions


def extract_for_rollout(rollout: dict, model, tokenizer, target_layers: List[int],
                        cc: dict) -> Dict:
    """Run a single forward pass and return {position: {layer: tensor}} + meta.

    Returns (activations_dict, meta_dict). Raises on tokenization/model errors.
    """
    fmt = reasoning_formats.get_format(cc.get("_reasoning_format", "think_tags"))
    system_prompt = resolve_extraction_system_prompt(rollout, cc)
    user_text = rollout["input"]["user_text"]
    assistant_content = _build_assistant_content(rollout, fmt)

    def _messages(include_assistant: bool):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_text})
        if include_assistant:
            msgs.append({"role": "assistant", "content": assistant_content})
        return msgs

    prompt_text = tokenizer.apply_chat_template(
        _messages(False), tokenize=False, add_generation_prompt=True)
    completion_start = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    full_text = tokenizer.apply_chat_template(
        _messages(True), tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    positions = _compute_positions(full_ids, completion_start, tokenizer, fmt)

    extra_mean_pools = {}
    if positions["reasoning_start"] < positions["reasoning_end"]:
        extra_mean_pools["reasoning_mean_pool"] = (positions["reasoning_start"],
                                                   positions["reasoning_end"])

    point_positions = {
        "last_token": positions["last_token"],
        "end_of_reasoning": positions["end_of_reasoning"],
        "first_answer_sentence_end": positions["first_answer_sentence_end"],
    }

    activations = extract_activations_at_positions(
        full_ids, model, target_layers, point_positions,
        answer_start=positions["answer_start"], answer_end=positions["answer_end"],
        extra_mean_pools=extra_mean_pools,
    )

    meta = {
        "rollout_id": rollout["rollout_id"],
        "seq_len": int(full_ids.shape[1]),
        "completion_start": completion_start,
        "positions": {k: int(v) for k, v in positions.items()},
        "extraction_system_prompt_kind": cc.get("extraction", {}).get("context_handler", "identity"),
    }
    return activations, meta
