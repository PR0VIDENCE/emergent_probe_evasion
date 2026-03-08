"""
Concept configuration loader and helpers.

Each concept (trees, library, currency, disease, ...) has a YAML config
that defines labels, field names, group structure, keywords, and topic lists.
Scripts load the concept config from the experiment YAML and use these helpers
to get concept-specific strings instead of hardcoded values.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Defaults for backward compatibility with existing trees data
_TREES_DEFAULTS = {
    "labels": {"positive": "tree", "negative": "non_tree"},
    "field_names": {
        "positive_question": "tree_question",
        "negative_question": "non_tree_question",
    },
}


def load_concept_config(path: str) -> dict:
    """Load and validate a concept config YAML."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    required = ["concept_name", "labels", "field_names"]
    for key in required:
        if key not in config:
            raise ValueError(f"Concept config missing required key: {key}")

    labels = config["labels"]
    if "positive" not in labels or "negative" not in labels:
        raise ValueError("Concept config labels must have 'positive' and 'negative'")

    fields = config["field_names"]
    if "positive_question" not in fields or "negative_question" not in fields:
        raise ValueError("Concept config field_names must have 'positive_question' and 'negative_question'")

    return config


def get_label_names(config: Optional[dict] = None) -> Tuple[str, str]:
    """Return (positive_label, negative_label) from concept config or defaults."""
    if config is None:
        return _TREES_DEFAULTS["labels"]["positive"], _TREES_DEFAULTS["labels"]["negative"]
    labels = config["labels"]
    return labels["positive"], labels["negative"]


def get_field_names(config: Optional[dict] = None) -> Tuple[str, str]:
    """Return (positive_question_field, negative_question_field) from concept config or defaults."""
    if config is None:
        return _TREES_DEFAULTS["field_names"]["positive_question"], _TREES_DEFAULTS["field_names"]["negative_question"]
    fields = config["field_names"]
    return fields["positive_question"], fields["negative_question"]


def get_concept_name(config: Optional[dict] = None) -> str:
    """Return the concept name string."""
    if config is None:
        return "trees"
    return config["concept_name"]


def get_no_keyword_instruction(config: Optional[dict] = None) -> str:
    """Return the no_keyword elicitation instruction for this concept."""
    if config is None:
        return "Answer without using the word 'tree' or 'trees'. Answer in 1-3 sentences."
    return config.get("no_keyword_instruction",
                      f"Answer without directly naming the core concept. Answer in 1-3 sentences.")


def get_negative_keywords(config: Optional[dict] = None) -> List[str]:
    """Return regex patterns for keywords that should not appear in negative questions."""
    if config is None:
        return [
            r"\btree\b", r"\btrees\b", r"\bforest\b", r"\bforests\b",
            r"\bwood\b", r"\bwoods\b", r"\bbark\b", r"\bleaf\b", r"\bleaves\b",
            r"\broot\b", r"\broots\b", r"\bbranch\b", r"\bbranches\b",
            r"\btrunk\b", r"\btrunks\b", r"\bcanopy\b", r"\bsapling\b",
            r"\blumber\b", r"\btimber\b",
        ]
    return config.get("negative_keywords", [])


def get_group_structure(config: Optional[dict] = None) -> dict:
    """Return group structure with counts: {group: count}."""
    if config is None:
        return {"A": 25, "B": 375, "C": 150, "D": 125, "E": 125, "A_prefix": 175}
    return config.get("group_structure", {"A": 25, "B": 375, "C": 150, "D": 125, "E": 125, "A_prefix": 175})


def get_pair_id_ranges(config: Optional[dict] = None) -> Dict[str, Tuple[int, int]]:
    """Compute pair_id ranges per group from group structure."""
    structure = get_group_structure(config)
    ranges = {}
    # A is always 0..A_count-1
    a_count = structure.get("A", 25)
    ranges["A"] = (0, a_count - 1)

    # B starts after A
    b_count = structure.get("B", 375)
    b_start = a_count
    ranges["B"] = (b_start, b_start + b_count - 1)

    # C starts after B
    c_count = structure.get("C", 150)
    c_start = b_start + b_count
    ranges["C"] = (c_start, c_start + c_count - 1)

    # D starts after C
    d_count = structure.get("D", 125)
    d_start = c_start + c_count
    ranges["D"] = (d_start, d_start + d_count - 1)

    # E starts after D
    e_count = structure.get("E", 125)
    e_start = d_start + d_count
    ranges["E"] = (e_start, e_start + e_count - 1)

    # Prefix starts at next round number (800 for 975 structure)
    pfx_count = structure.get("A_prefix", 175)
    pfx_start = e_start + e_count
    # Round up to next 100
    pfx_start = ((pfx_start + 99) // 100) * 100
    ranges["A_prefix"] = (pfx_start, pfx_start + pfx_count - 1)

    return ranges


def load_concept_from_experiment(experiment_config: dict, project_root: str) -> Optional[dict]:
    """
    Load concept config from an experiment config if concept_config field is present.

    Returns None if no concept_config field (backward compat with trees).
    """
    concept_config_path = experiment_config.get("concept_config")
    if concept_config_path is None:
        return None
    path = Path(concept_config_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    return load_concept_config(str(path))
