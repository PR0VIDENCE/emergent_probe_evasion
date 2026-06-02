"""
Batched generation helper for the eval pipeline.

Generalizes the batched-with-fallback pattern from sycophancy_stage0_sanity.py:
each prompt carries its OWN system prompt (needed for deception, where the
system prompt is the per-scenario context), so we build chat prompts per-item
and left-pad. max_new_tokens must be uniform within a batch (the caller groups
by regime / token budget).
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch


def parse_reasoning_response(text: str, has_reasoning: bool = True):
    """Split <think>...</think> from the final response.

    For non-reasoning models (has_reasoning=False) the whole text is the response.
    """
    if has_reasoning and "</think>" in text:
        thinking, _, response = text.partition("</think>")
        return thinking.replace("<think>", "").strip(), response.strip()
    if has_reasoning:
        # Generation truncated before </think> — everything is reasoning.
        return text.strip(), ""
    return "", text.strip()


def _chat_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, prompt_infos: List[dict], gen_config: dict
                   ) -> Tuple[List[Tuple[str, int, bool]], float]:
    """Generate a batch. Each prompt_info has 'system_prompt' + 'user_text'.

    Returns ([(full_response, n_gen_tokens, truncated), ...], batch_seconds).
    """
    chat_prompts = [_chat_prompt(tokenizer, p["system_prompt"], p["user_text"]) for p in prompt_infos]

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tokenizer.padding_side = original_side

    attention_mask = inputs["attention_mask"]
    input_lens = [int(attention_mask[i].sum().item()) for i in range(len(prompt_infos))]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 2048),
            temperature=gen_config.get("temperature", 0.6),
            top_p=gen_config.get("top_p", 0.95),
            top_k=gen_config.get("top_k", 20),
            do_sample=gen_config.get("temperature", 0.6) > 0,
            pad_token_id=pad_id,
        )
    gen_time = time.time() - t0

    padded_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompt_infos)):
        pad_count = padded_len - input_lens[i]
        real = out[i, pad_count:]
        gen_ids = real[input_lens[i]:]
        truncated = True
        if pad_id is not None and gen_ids.numel() > 0:
            eos = (gen_ids == pad_id).nonzero(as_tuple=True)[0]
            if eos.numel() > 0:
                gen_ids = gen_ids[:eos[0].item() + 1]
                truncated = False
        full_response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((full_response, int(gen_ids.shape[0]), truncated))
    return results, gen_time


def generate_with_fallback(model, tokenizer, prompt_infos: List[dict], gen_config: dict):
    """Try a batch; on OOM/runtime error fall back to size-1 generation."""
    try:
        return generate_batch(model, tokenizer, prompt_infos, gen_config)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # noqa: PERF203
        print(f"  WARN: batch of {len(prompt_infos)} failed "
              f"({type(e).__name__}: {str(e)[:120]}); falling back to size-1")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results, total = [], 0.0
        for p in prompt_infos:
            r, t = generate_batch(model, tokenizer, [p], gen_config)
            results.extend(r)
            total += t
        return results, total
