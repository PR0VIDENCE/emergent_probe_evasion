"""Trial file discovery, loading, and indexing."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def discover_regimes(data_dir: str) -> List[str]:
    """Scan for JSON files in trials/ subdirectory and return regime names."""
    trials_dir = os.path.join(data_dir, "trials")
    if not os.path.isdir(trials_dir):
        # Maybe files are directly in the data_dir
        trials_dir = data_dir

    regimes = []
    if os.path.isdir(trials_dir):
        for f in sorted(os.listdir(trials_dir)):
            if f.endswith(".json"):
                regimes.append(f.replace(".json", ""))
    return regimes


def load_trials(data_dir: str, regime_name: str) -> Optional[List[dict]]:
    """Load one regime's trial data from JSON."""
    # Try trials/ subdirectory first, then data_dir directly
    for base in [os.path.join(data_dir, "trials"), data_dir]:
        path = os.path.join(base, f"{regime_name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return None


def load_all_trials(data_dir: str, regime_names: List[str]) -> Dict[str, List[dict]]:
    """Load trials for multiple regimes."""
    result = {}
    for name in regime_names:
        trials = load_trials(data_dir, name)
        if trials is not None:
            result[name] = trials
    return result


def build_trial_index(trials: List[dict]) -> Dict[Tuple[str, int], dict]:
    """Index trials by (question_id, rollout) -> trial dict."""
    index = {}
    for trial in trials:
        key = (trial["question_id"], trial["rollout"])
        index[key] = trial
    return index


def get_trial_probe_results(trial: dict) -> dict:
    """Extract probe_results from a trial, handling feedback regimes."""
    if "turns" in trial and trial["turns"]:
        return trial["turns"][-1].get("probe_results", {})
    return trial.get("probe_results", {})


def get_unique_questions(trials: List[dict]) -> List[str]:
    """Get unique question IDs from trials, preserving order."""
    seen = set()
    result = []
    for trial in trials:
        qid = trial["question_id"]
        if qid not in seen:
            seen.add(qid)
            result.append(qid)
    return result


def get_unique_rollouts(trials: List[dict]) -> List[int]:
    """Get unique rollout numbers from trials."""
    return sorted(set(trial["rollout"] for trial in trials))


def load_summary(data_dir: str) -> Optional[dict]:
    """Load summary.json if it exists."""
    path = os.path.join(data_dir, "summary.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
