"""
Shared, resumable run helpers for generation + extraction.

Both the probe-training stages and the evasion stages funnel through these so
the rollout schema, batching, prewritten handling, mock backend, and resume
logic live in exactly one place.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

from . import common, generation, mock


def _done_ids(path) -> set:
    return {r["rollout_id"] for r in common.read_jsonl(path)}


def run_generation(config: dict, concept: str, ctx, phase: str,
                   requests: List[dict], out_path) -> int:
    """Generate (or copy prewritten) rollouts → out_path (append, resumable).

    Returns the number of newly written rollouts.
    """
    cc = config["_concept_configs"][concept]
    model_id = config["_model_name"]
    has_reasoning = common.has_reasoning_trace(config)
    gen_defaults = cc.get("generation", {})

    # Attach the canonical rollout_id to each request for resume + dedupe.
    for req in requests:
        req["_rid"] = common.rollout_id(concept, phase, req["contrastive_id"], req.get("regime"))

    done = _done_ids(out_path)
    pending = [r for r in requests if r["_rid"] not in done]
    if not pending:
        print(f"  [{concept}/{phase}] generation: nothing pending ({len(done)} done)")
        return 0

    print(f"  [{concept}/{phase}] generating {len(pending)} rollouts "
          f"({len(done)} already done)")

    # 1) Prewritten rollouts (deception probe-training): no model call.
    prewritten = [r for r in pending if r.get("prewritten")]
    for req in prewritten:
        rec = common.make_rollout_record(
            concept=concept, phase=phase, regime=req.get("regime"),
            contrastive_id=req["contrastive_id"], model_id=model_id,
            system_prompt=req["system_prompt"], user_text=req["user_text"],
            thinking="", response=req.get("response", ""),
            n_gen_tokens=0, truncated=False, metadata=req.get("metadata", {}),
        )
        rec["metadata"]["prewritten"] = True
        common.append_jsonl(out_path, rec)

    generated = [r for r in pending if not r.get("prewritten")]
    if not generated:
        return len(pending)

    # 2) Mock generation path.
    if config.get("_mock"):
        for req in generated:
            thinking, response, n_tok, trunc = mock.mock_generate(req, has_reasoning)
            rec = common.make_rollout_record(
                concept=concept, phase=phase, regime=req.get("regime"),
                contrastive_id=req["contrastive_id"], model_id=model_id,
                system_prompt=req["system_prompt"], user_text=req["user_text"],
                thinking=thinking, response=response, n_gen_tokens=n_tok,
                truncated=trunc, metadata=req.get("metadata", {}),
            )
            common.append_jsonl(out_path, rec)
        return len(pending)

    # 3) Real batched generation, grouped by token budget.
    model, tokenizer = ctx.get()
    batch_size = config.get("batch_size", gen_defaults.get("batch_size", 10))
    by_budget = defaultdict(list)
    for req in generated:
        by_budget[req["max_new_tokens"]].append(req)

    t0 = time.time()
    n_written = len(prewritten)
    for budget, reqs in sorted(by_budget.items()):
        gen_config = {
            "max_new_tokens": budget,
            "temperature": gen_defaults.get("temperature", 0.6),
            "top_p": gen_defaults.get("top_p", 0.95),
            "top_k": gen_defaults.get("top_k", 20),
        }
        for i in range(0, len(reqs), batch_size):
            batch = reqs[i:i + batch_size]
            results, batch_time = generation.generate_with_fallback(
                model, tokenizer, batch, gen_config)
            for req, (full_response, n_tok, trunc) in zip(batch, results):
                thinking, response = generation.parse_reasoning_response(full_response, has_reasoning)
                rec = common.make_rollout_record(
                    concept=concept, phase=phase, regime=req.get("regime"),
                    contrastive_id=req["contrastive_id"], model_id=model_id,
                    system_prompt=req["system_prompt"], user_text=req["user_text"],
                    thinking=thinking, response=response, n_gen_tokens=n_tok,
                    truncated=trunc, metadata=req.get("metadata", {}),
                )
                common.append_jsonl(out_path, rec)
                n_written += 1
            elapsed = time.time() - t0
            print(f"    [{concept}/{phase}] {n_written}/{len(pending)} "
                  f"(budget={budget}, {elapsed/60:.1f}min)")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return n_written


def run_extraction(config: dict, concept: str, ctx, phase: str,
                   labeled_path, acts_dir, log_path) -> int:
    """Forward-pass extraction for each labeled rollout → acts_dir/<rid>.pt.

    Resumable on the existence of the .pt file. Returns number extracted.
    """
    from . import extraction as extraction_mod
    cc = config["_concept_configs"][concept]
    cc["_has_reasoning_trace"] = common.has_reasoning_trace(config)
    layers = common.target_layers(config)
    acts_dir = Path(acts_dir)
    acts_dir.mkdir(parents=True, exist_ok=True)

    rollouts = common.read_jsonl(labeled_path)
    pending = [r for r in rollouts if not (acts_dir / f"{r['rollout_id']}.pt").exists()]
    if not pending:
        print(f"  [{concept}/{phase}] extraction: nothing pending ({len(rollouts)} total)")
        return 0
    print(f"  [{concept}/{phase}] extracting {len(pending)} rollouts")

    import torch

    # Mock path: random separable activations.
    if config.get("_mock"):
        fail = concept in (config.get("mock_fail_concepts", []) or [])
        for r in pending:
            acts = mock.mock_extract(r, cc, layers, fail)
            torch.save(acts, acts_dir / f"{r['rollout_id']}.pt")
            common.append_jsonl(log_path, {"rollout_id": r["rollout_id"], "mock": True})
        return len(pending)

    model, tokenizer = ctx.get()
    n_done = 0
    t0 = time.time()
    for r in pending:
        try:
            acts, meta = extraction_mod.extract_for_rollout(r, model, tokenizer, layers, cc)
            torch.save(acts, acts_dir / f"{r['rollout_id']}.pt")
            common.append_jsonl(log_path, meta)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            common.log_error(config, concept, f"{phase}_extraction",
                             f"{type(e).__name__}: {str(e)[:160]}",
                             {"rollout_id": r["rollout_id"]})
        if n_done % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n_done % 25 == 0:
            print(f"    [{concept}/{phase}] extracted {n_done}/{len(pending)} "
                  f"({(time.time()-t0)/60:.1f}min)")
    return n_done
