"""
Shared utilities for the standardized probe-evasion eval pipeline.

This module is the single source of truth for:
  - eval-run / concept config loading + path resolution
  - the per-concept directory layout under data/eval_runs/<run_id>/
  - per-concept and global status.json read/update (resumability)
  - the standardized rollout schema (read/write JSONL)
  - run-id derivation

The pipeline is "L2" standardized: one set of stage scripts, with
concept-specific behavior living in small handlers (see data_handlers.py,
judging.py, extraction.py) selected by the concept config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The ordered list of pipeline stages (matches §2.7 of the meta-prompt).
STAGES = [
    "generation",
    "labeling",
    "extraction",
    "probe_training",
    "probe_validation",
    "evasion_generation",
    "evasion_labeling",
    "evasion_extraction",
    "scoring",
]

# The 5 token positions we extract for QwQ-style reasoning models. Concepts
# without a <think> boundary (prewritten deception) fall back gracefully —
# find_token_positions / compute_positions copy last_token into the missing
# slots, so the position set is uniform across concepts.
DEFAULT_TOKEN_POSITIONS = [
    "last_token",
    "end_of_reasoning",
    "first_answer_sentence_end",
    "answer_mean_pool",
    "reasoning_mean_pool",
]


# ---------------------------------------------------------------------------
# Config + path helpers
# ---------------------------------------------------------------------------

def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p) -> Path:
    """Resolve a path against PROJECT_ROOT (not CWD)."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_eval_config(config_path) -> dict:
    """Load an eval-run config and attach resolved sub-configs.

    Returns the raw config dict augmented with:
      - _model_config: the loaded model config dict
      - _model_name: convenience
      - _run_id: resolved run id
      - _concept_configs: {concept_name: concept_config_dict}
      - _universal_regimes: loaded universal regime templates
    """
    config = load_yaml(resolve_path(config_path))

    model_config = load_yaml(resolve_path(config["model_config"]))
    config["_model_config"] = model_config
    config["_model_name"] = model_config.get("model_name", "model")

    config["_run_id"] = derive_run_id(config)

    # Load per-concept configs, applying any per-run overrides (deep-merged).
    # `concept_overrides` lets a run tweak a concept's settings (e.g. lower the
    # validation_threshold per §9) without editing the shared concept config.
    concept_dir = config.get("concept_config_dir", "configs/eval/concepts")
    overrides = config.get("concept_overrides", {}) or {}
    concept_configs = {}
    for concept in config["concepts"]:
        cc_path = resolve_path(Path(concept_dir) / f"{concept}.yaml")
        cc = load_yaml(cc_path)
        if concept in overrides:
            _deep_merge(cc, overrides[concept])
        concept_configs[concept] = cc
    config["_concept_configs"] = concept_configs

    # Universal regimes.
    ur_path = config.get("universal_regimes_file", "configs/eval/universal_regimes.yaml")
    config["_universal_regimes"] = load_yaml(resolve_path(ur_path))["universal_regimes"]

    return config


def _deep_merge(base: dict, override: dict):
    """Recursively merge `override` into `base` (in place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def derive_run_id(config: dict) -> str:
    """run_id explicitly set in config wins; else {model_name}_{date}_{mode}."""
    if config.get("run_id"):
        return config["run_id"]
    model_name = config.get("_model_name", "model").replace("/", "_").replace("-", "")
    date = config.get("date", "nodate")
    mode = "fast" if config.get("fast") else "full"
    return f"{model_name}_{date}_{mode}"


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

def run_dir(config: dict) -> Path:
    base = config.get("output_base_dir", "data/eval_runs")
    base_path = Path(base)
    if base_path.is_absolute():
        return base_path / config["_run_id"]
    return resolve_path(base_path / config["_run_id"])


def concept_dir(config: dict, concept: str) -> Path:
    return run_dir(config) / "concepts" / concept


def stage_paths(config: dict, concept: str) -> Dict[str, Path]:
    """All the canonical paths for a concept's directory tree."""
    cd = concept_dir(config, concept)
    return {
        "concept_dir": cd,
        "status": cd / "status.json",
        "errors": cd / "errors.jsonl",
        # probe training
        "pt_dir": cd / "probe_training",
        "pt_rollouts": cd / "probe_training" / "rollouts.jsonl",
        "pt_labeled": cd / "probe_training" / "rollouts_labeled.jsonl",
        "pt_acts": cd / "probe_training" / "activations",
        "pt_extract_log": cd / "probe_training" / "extraction_log.jsonl",
        # probes
        "probes_dir": cd / "probes",
        "validation": cd / "probes" / "validation.json",
        "probe_results": cd / "probes" / "results.json",
        "combiner": cd / "probes" / "combiner.json",
        # evasion
        "ev_dir": cd / "evasion",
        "ev_rollouts": cd / "evasion" / "rollouts.jsonl",
        "ev_labeled": cd / "evasion" / "rollouts_labeled.jsonl",
        "ev_acts": cd / "evasion" / "activations",
        "ev_extract_log": cd / "evasion" / "extraction_log.jsonl",
        "ev_scores": cd / "evasion" / "scores.jsonl",
        "ev_report": cd / "evasion" / "regime_report.md",
    }


def ensure_concept_dirs(config: dict, concept: str):
    paths = stage_paths(config, concept)
    for key in ("pt_dir", "pt_acts", "probes_dir", "ev_dir", "ev_acts"):
        paths[key].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Status management (resumability)
# ---------------------------------------------------------------------------

def read_status(config: dict, concept: str) -> dict:
    path = stage_paths(config, concept)["status"]
    if path.exists():
        return load_yaml(path) if path.suffix in (".yaml", ".yml") else json.loads(path.read_text())
    return {"concept": concept, "stages": {}, "validation": None}


def write_status(config: dict, concept: str, status: dict):
    path = stage_paths(config, concept)["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2))


def mark_stage(config: dict, concept: str, stage: str, info: Optional[dict] = None):
    status = read_status(config, concept)
    status.setdefault("stages", {})[stage] = {"status": "completed", **(info or {})}
    write_status(config, concept, status)


def stage_done(config: dict, concept: str, stage: str) -> bool:
    status = read_status(config, concept)
    return status.get("stages", {}).get(stage, {}).get("status") == "completed"


def log_error(config: dict, concept: str, stage: str, error: str, context: Optional[dict] = None):
    path = stage_paths(config, concept)["errors"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"stage": stage, "error": error, "context": context or {}}) + "\n")


# ---------------------------------------------------------------------------
# Global run status
# ---------------------------------------------------------------------------

def write_global_status(config: dict):
    """Aggregate per-concept status into the top-level status.json."""
    rd = run_dir(config)
    rd.mkdir(parents=True, exist_ok=True)
    concepts = {}
    for concept in config["concepts"]:
        st = read_status(config, concept)
        concepts[concept] = {
            "stages_done": [s for s, v in st.get("stages", {}).items()
                            if v.get("status") == "completed"],
            "validation": st.get("validation"),
        }
    (rd / "status.json").write_text(json.dumps({
        "run_id": config["_run_id"],
        "model": config["_model_name"],
        "fast": bool(config.get("fast")),
        "concepts": concepts,
    }, indent=2))


def snapshot_config(config: dict):
    """Write a snapshot of the eval-run config to the run dir."""
    rd = run_dir(config)
    rd.mkdir(parents=True, exist_ok=True)
    # Strip the loaded (underscore-prefixed) caches for a clean snapshot.
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    (rd / "config.yaml").write_text(yaml.safe_dump(clean, sort_keys=False))


# ---------------------------------------------------------------------------
# Standardized rollout schema
# ---------------------------------------------------------------------------

def rollout_id(concept: str, phase: str, contrastive_id: str, regime: Optional[str]) -> str:
    """Filesystem-safe id used as the activation .pt filename + join key.

    Format: <concept>_<phase>_<contrastive_id>_<regime>
    regime is 'none' for probe_training rollouts.
    """
    rid = f"{concept}_{phase}_{contrastive_id}_{regime or 'none'}"
    return rid.replace("/", "_").replace(" ", "_")


def make_rollout_record(
    *, concept: str, phase: str, regime: Optional[str], contrastive_id: str,
    model_id: str, system_prompt: str, user_text: str,
    thinking: str = "", response: str = "", n_gen_tokens: int = 0,
    truncated: bool = False, metadata: Optional[dict] = None,
) -> dict:
    """Construct a rollout record in the standardized schema (§2.6)."""
    return {
        "concept": concept,
        "phase": phase,
        "regime": regime,
        "rollout_id": rollout_id(concept, phase, contrastive_id, regime),
        "contrastive_id": contrastive_id,
        "model_id": model_id,
        "input": {"system_prompt": system_prompt, "user_text": user_text},
        "output": {
            "thinking": thinking, "response": response,
            "n_gen_tokens": n_gen_tokens, "truncated": truncated,
        },
        "labels": {"judge_label": None, "judge_reason": "", "judge_model": None},
        "probe_scores": {},
        "metadata": metadata or {},
    }


def target_layers(config: dict) -> List[int]:
    """Layers to probe. Honors the eval config's `target_layers`, but guards
    against out-of-range indices (so a global [4..60] list does not crash a
    model with fewer layers) and derives a sensible ~15-layer default from the
    model's layer count when unspecified."""
    nl = config["_model_config"].get("num_layers", 64)
    tl = config.get("target_layers")
    if tl == "all":
        return list(range(nl))
    step = max(1, round(nl / 15))   # ~15 evenly spaced layers
    default = list(range(step, nl, step))
    if tl:
        kept = [int(l) for l in tl if int(l) < nl]
        return kept or default
    return default


def has_reasoning_trace(config: dict) -> bool:
    return bool(config["_model_config"].get("has_reasoning_trace", True))


def reasoning_format(config: dict) -> str:
    """Resolve the model's reasoning format name.

    Prefers the explicit `reasoning_format` model-config field; falls back to the
    older boolean `has_reasoning_trace` (true -> think_tags, false -> none) so
    existing model configs keep working unchanged.
    """
    mc = config["_model_config"]
    fmt = mc.get("reasoning_format")
    if fmt:
        return fmt
    return "think_tags" if mc.get("has_reasoning_trace", True) else "none"


class ModelContext:
    """Lazily loads the model+tokenizer once and shares it across all stages /
    concepts. In mock mode no model is loaded."""

    def __init__(self, config: dict, mock: bool = False):
        self.config = config
        self.mock = mock
        self._model = None
        self._tokenizer = None

    def get(self):
        if self.mock:
            return None, None
        if self._model is None:
            from src.inference.extract_activations import load_model_and_tokenizer
            self._model, self._tokenizer = load_model_and_tokenizer(self.config["_model_config"])
        return self._model, self._tokenizer


def load_for_cli(config_path, fast: bool = False, mock: bool = False,
                 concepts: Optional[str] = None) -> dict:
    """Load an eval config + apply CLI overrides (fast / mock / concept subset)."""
    config = load_eval_config(config_path)
    if fast:
        config["fast"] = True
    if mock:
        config["_mock"] = True
    if concepts:
        wanted = [c.strip() for c in concepts.split(",") if c.strip()]
        config["concepts"] = [c for c in config["concepts"] if c in wanted]
        config["_concept_configs"] = {c: config["_concept_configs"][c]
                                      for c in config["concepts"]}
    # run_id depends on fast/mode — re-derive after overrides.
    config["_run_id"] = derive_run_id(config)
    return config


def read_jsonl(path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, records: List[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
