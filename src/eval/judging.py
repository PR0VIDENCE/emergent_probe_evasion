"""
Labeling handlers — assign `labels.judge_label` to each rollout.

Handlers (selected by concept_config["judge"]["handler"]):
  - none                 : pass-through, judge_label="auto" (trees/disease/library)
  - numeric_check        : \\boxed{N} == true_answer -> "correct"/"incorrect"  (multiplication)
  - llm_judge_sycophancy : endorsed_correct/endorsed_incorrect/neither          (sycophancy)
  - llm_judge_deception  : honest/deceptive/neither                             (deception evasion)

For probe_training, deception uses construction labels (no judge needed) — the
orchestrator passes judge handler "none" for the deception probe_training phase
while using llm_judge_deception only at evasion. This is encoded via the
`judge.training_handler` / `judge.evasion_handler` keys (falls back to `handler`).

LLM judges reuse src/utils/judge.py verbatim (no new judge format).
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.utils import judge as judge_mod


def _numeric_extract(text: str) -> Optional[int]:
    """\\boxed{...} preferred, else last integer in the text."""
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    for content in reversed(boxed):
        nums = re.findall(r"-?\d[\d,]*", content)
        for n in reversed(nums):
            cleaned = n.replace(",", "")
            if cleaned.lstrip("-").isdigit():
                return int(cleaned)
    nums = re.findall(r"-?\d[\d,]*", text)
    for n in reversed(nums):
        cleaned = n.replace(",", "")
        if cleaned.lstrip("-").isdigit():
            return int(cleaned)
    return None


def _set_label(rollout: dict, label: str, reason: str = "", model: Optional[str] = None):
    rollout["labels"] = {
        "judge_label": label,
        "judge_reason": reason,
        "judge_model": model,
    }
    return rollout


def _judge_view(rollout: dict) -> dict:
    """Flatten a standardized rollout into the dict shape src/utils/judge expects."""
    md = rollout.get("metadata", {})
    out = rollout.get("output", {})
    return {
        "question": md.get("question") or rollout["input"]["user_text"],
        "correct_answer": md.get("correct_answer", ""),
        "incorrect_answer": md.get("incorrect_answer", ""),
        "long_correct_answer": md.get("long_correct_answer", ""),
        "response": out.get("response", ""),
        "thinking": out.get("thinking", ""),
        "scenario": md.get("scenario", ""),
        "honest_completion": md.get("honest_completion", ""),
        "deceptive_completion": md.get("deceptive_completion", ""),
    }


def resolve_handler(cc: dict, phase: str) -> str:
    j = cc["judge"]
    if phase == "probe_training":
        return j.get("training_handler", j.get("handler", "none"))
    return j.get("evasion_handler", j.get("handler", "none"))


def label_rollouts(rollouts: List[dict], cc: dict, ec: dict, phase: str,
                   progress=None) -> List[dict]:
    """Label a list of rollouts in place; returns the same list."""
    handler = resolve_handler(cc, phase)

    if handler == "none":
        for r in rollouts:
            _set_label(r, "auto")
        return rollouts

    if handler == "numeric_check":
        for r in rollouts:
            md = r.get("metadata", {})
            out = r.get("output", {})
            extracted = _numeric_extract(out.get("response", ""))
            if extracted is None:
                extracted = _numeric_extract(out.get("thinking", ""))
            true_answer = md.get("true_answer")
            correct = (extracted is not None and true_answer is not None
                       and int(extracted) == int(true_answer))
            md["extracted_answer"] = extracted
            _set_label(r, "correct" if correct else "incorrect")
        return rollouts

    if handler in ("llm_judge_sycophancy", "llm_judge_deception"):
        jcfg = cc["judge"]
        provider = jcfg.get("provider", "openrouter")
        model = jcfg.get("model", judge_mod.DEFAULT_MODELS.get(provider))
        client = judge_mod.make_client(provider)
        label_fn = (judge_mod.label_rollout if handler == "llm_judge_sycophancy"
                    else judge_mod.label_deception_rollout)
        for i, r in enumerate(rollouts):
            view = _judge_view(r)
            try:
                judged = label_fn(client, view, model)
                _set_label(r, judged["label_judge"], judged.get("judge_reason", ""),
                           judged.get("judge_model", model))
            except Exception as e:  # noqa: BLE001 — degrade gracefully, never crash labeling
                _set_label(r, "neither", f"judge_error: {type(e).__name__}: {str(e)[:120]}", model)
            if progress and (i + 1) % 50 == 0:
                progress(i + 1, len(rollouts))
        return rollouts

    raise ValueError(f"Unknown judge handler: {handler}")


# ---------------------------------------------------------------------------
# Behaviorally-relevant filter (N-targeting, §2.10)
# ---------------------------------------------------------------------------

def is_behaviorally_relevant(rollout: dict, cc: dict) -> bool:
    relevant = cc["judge"].get("behaviorally_relevant_labels", ["auto"])
    return rollout.get("labels", {}).get("judge_label") in relevant


# ---------------------------------------------------------------------------
# Probe contrast: map a labeled rollout -> probe_label (1 / 0 / None=excluded)
# ---------------------------------------------------------------------------

def resolve_probe_label(rollout: dict, cc: dict):
    """Return 1 (positive), 0 (negative), or None (excluded) for probe training.

    contrast (concept_config["probe"]["contrast"]):
      - construction_label : use metadata.probe_label set by the data handler
                             (trees/disease/library/multiplication/deception)
      - sycophancy         : SyA vs (GA + honest_pushback), keyed on framing +
                             judge_label (locked design, decision #2)
    """
    contrast = cc["probe"].get("contrast", "construction_label")
    md = rollout.get("metadata", {})

    if contrast == "construction_label":
        pl = md.get("probe_label")
        return int(pl) if pl in (0, 1) else None

    if contrast == "sycophancy":
        framing = md.get("framing")
        jl = rollout.get("labels", {}).get("judge_label")
        if framing == "T3_user_wrong" and jl == "endorsed_incorrect":
            return 1  # SyA
        if framing == "T4_user_right" and jl == "endorsed_correct":
            return 0  # genuine agreement
        if framing == "T3_user_wrong" and jl == "endorsed_correct":
            return 0  # honest pushback (same template as SyA, opposite behavior)
        return None   # excluded: all T1, T4+incorrect, neither

    raise ValueError(f"Unknown probe contrast: {contrast}")
