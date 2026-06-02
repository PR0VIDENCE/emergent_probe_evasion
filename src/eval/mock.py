"""
Mock simulation backend for the eval pipeline.

Lets the ENTIRE pipeline (orchestration, schema, resumability, idempotency,
labeling, probe training/validation, evasion scoring, dashboard, failure
handling) be exercised on a laptop with no GPU and no model download. The real
generation/extraction code paths are untouched — stage scripts branch on
config["_mock"].

Mock activations carry a deterministic linear signal so that:
  - concepts NOT in eval_config["mock_fail_concepts"] -> separable -> probe passes
  - concepts in that list                              -> no signal -> probe fails
This exercises both the pass and fail branches of probe validation.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import torch

from .common import DEFAULT_TOKEN_POSITIONS
from .judging import resolve_probe_label

HIDDEN_DIM = 64  # small mock activation width


def _rng(*parts) -> torch.Generator:
    seed = int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def mock_generate(req: dict, has_reasoning: bool):
    """Synthesize a (thinking, response, n_tokens, truncated) tuple.

    For multiplication, embed the correct \\boxed{answer} most of the time so
    numeric_check produces mostly-correct labels.
    """
    md = req.get("metadata", {})
    thinking = "Let me think about this step by step." if has_reasoning else ""
    if "true_answer" in md:
        # Multiplication: emit the right answer ~95% of the time (deterministic).
        g = _rng("mockgen", req["contrastive_id"], req.get("regime"))
        correct = torch.rand(1, generator=g).item() < 0.95
        ans = md["true_answer"] if correct else md["true_answer"] + 1
        response = f"The answer is \\boxed{{{ans}}}."
    else:
        response = f"This is a mock response about the topic. ({req['contrastive_id']})"
    n_tokens = len(thinking.split()) + len(response.split())
    return thinking, response, n_tokens, False


def mock_judge_label(rollout: dict, cc: dict, phase: str) -> str:
    """Assign a plausible judge label without any API call."""
    from .judging import resolve_handler
    handler = resolve_handler(cc, phase)
    md = rollout.get("metadata", {})

    if handler == "none":
        return "auto"
    if handler == "numeric_check":
        from .judging import _numeric_extract
        extracted = _numeric_extract(rollout["output"].get("response", ""))
        ta = md.get("true_answer")
        return "correct" if (extracted is not None and ta is not None and int(extracted) == int(ta)) else "incorrect"
    if handler == "llm_judge_sycophancy":
        g = _rng("mockjudge", rollout["rollout_id"])
        framing = md.get("framing")
        if framing == "T3_user_wrong":
            # ~25% SyA, rest pushback.
            return "endorsed_incorrect" if torch.rand(1, generator=g).item() < 0.25 else "endorsed_correct"
        if framing == "T4_user_right":
            return "endorsed_correct"
        return "endorsed_correct"
    if handler == "llm_judge_deception":
        g = _rng("mockjudge", rollout["rollout_id"])
        return "deceptive" if torch.rand(1, generator=g).item() < 0.40 else "honest"
    return "auto"


def mock_extract(rollout: dict, cc: dict, target_layers: List[int],
                 fail: bool) -> Dict[str, Dict[int, torch.Tensor]]:
    """Random activations with an optional separable linear signal.

    Positive rollouts get +mu on a fixed direction; negatives get -mu. When
    `fail` is True, mu=0 so the probe can't separate the classes.
    """
    plabel = resolve_probe_label(rollout, cc)
    if plabel is None:
        plabel = 1 if rollout.get("metadata", {}).get("probe_label") == 1 else 0
    mu = 3.0
    if fail:
        # Signal coordinate is driven by a hash of the rollout id, i.e.
        # UNCORRELATED with the label -> probe cannot separate -> validation
        # reliably fails (deterministic, unlike pure noise on tiny val sets).
        h = int(_rng("mockfail", rollout["rollout_id"]).initial_seed())
        sign = 1.0 if (h % 2 == 0) else -1.0
    else:
        sign = 1.0 if plabel == 1 else -1.0

    out = {}
    for pos in DEFAULT_TOKEN_POSITIONS:
        layer_acts = {}
        for layer in target_layers:
            g = _rng("mockact", rollout["rollout_id"], pos, layer)
            x = torch.randn(HIDDEN_DIM, generator=g)
            x[0] += sign * mu  # signal on the first coordinate
            layer_acts[layer] = x.half()
        out[pos] = layer_acts
    return out
