"""
Reasoning-format handlers — the ONE place that knows how a model delimits its
reasoning trace from its final answer.

Selected by the model config's `reasoning_format` (back-compat: derived from the
older boolean `has_reasoning_trace` when absent). Each handler provides:

  - keep_special_tokens : decode generations with special tokens kept? (harmony
                          channel markers are special tokens and must be kept;
                          QwQ's <think> tags are ordinary tokens, so stripped)
  - split(text)         : (thinking, response) from a generated completion
  - reconstruct(t, r)   : assistant-message string for the re-extraction forward
                          pass (must round-trip so boundary() can re-find it)
  - boundary(completion): a re.Match on the decoded assistant completion marking
                          the reasoning→answer transition, or None (whole
                          completion is the answer)

Adding a new reasoning model = add a handler here + name it in the model config.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


class ReasoningFormat:
    name = "base"
    keep_special_tokens = False

    def split(self, text: str) -> Tuple[str, str]:
        raise NotImplementedError

    def reconstruct(self, thinking: str, response: str) -> str:
        raise NotImplementedError

    def boundary(self, completion_text: str):
        raise NotImplementedError


class NoneFormat(ReasoningFormat):
    """Non-reasoning instruct models (Llama-3, Qwen2.5-Instruct, Mistral, Gemma)."""
    name = "none"
    keep_special_tokens = False

    def split(self, text):
        return "", text.strip()

    def reconstruct(self, thinking, response):
        return response

    def boundary(self, completion_text):
        return None


class ThinkTagsFormat(ReasoningFormat):
    """`<think>...</think>` reasoning models (QwQ, DeepSeek-R1 distills, Qwen3-thinking)."""
    name = "think_tags"
    keep_special_tokens = False

    def split(self, text):
        if "</think>" in text:
            thinking, _, response = text.partition("</think>")
            return thinking.replace("<think>", "").strip(), response.strip()
        # Generation truncated before </think> — everything is reasoning.
        return text.strip(), ""

    def reconstruct(self, thinking, response):
        return f"<think>{thinking}</think>{response}" if thinking else response

    def boundary(self, completion_text):
        return re.search(r"</think>", completion_text)


# ---------------------------------------------------------------------------
# Harmony (GPT-OSS) — channel-based: analysis = reasoning, final = answer.
#
# NOTE: written to the documented harmony spec. The channel markers are special
# tokens, so generations must be decoded with skip_special_tokens=False. The
# reconstruct() round-trip through apply_chat_template should be VALIDATED
# against the real gpt-oss tokenizer on the GPU box before trusting the
# reasoning-position columns; the canonical answer_mean_pool position only needs
# the `final` channel to be located correctly, which boundary() handles.
# ---------------------------------------------------------------------------

_HARMONY_STOPS = r"<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|<\|channel\|>"


def _harmony_channel(text: str, channel: str) -> Optional[str]:
    m = re.search(
        r"<\|channel\|>\s*" + channel + r"\s*<\|message\|>(.*?)(?=" + _HARMONY_STOPS + r"|$)",
        text, re.DOTALL,
    )
    return m.group(1) if m else None


class HarmonyFormat(ReasoningFormat):
    name = "harmony"
    keep_special_tokens = True

    def split(self, text):
        analysis = _harmony_channel(text, "analysis")
        final = _harmony_channel(text, "final")
        if final is None:
            # No final channel (truncated mid-analysis) -> all reasoning.
            return (analysis if analysis is not None else text).strip(), ""
        return (analysis or "").strip(), final.strip()

    def reconstruct(self, thinking, response):
        # Rebuild the channel structure so boundary() re-locates the final
        # channel during re-extraction.
        return (f"<|channel|>analysis<|message|>{thinking}<|end|>"
                f"<|start|>assistant<|channel|>final<|message|>{response}")

    def boundary(self, completion_text):
        return re.search(r"<\|channel\|>\s*final\s*<\|message\|>", completion_text)


_REGISTRY = {
    "none": NoneFormat(),
    "think_tags": ThinkTagsFormat(),
    "harmony": HarmonyFormat(),
}


def get_format(name: str) -> ReasoningFormat:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown reasoning_format {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
