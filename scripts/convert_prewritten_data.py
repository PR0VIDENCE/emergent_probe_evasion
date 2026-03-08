#!/usr/bin/env python3
"""
Convert Apollo deception datasets into committed scenario files for the
prewritten activation extraction pipeline.

Reads from deception-detection/ (gitignored, local only) and writes to
data/concepts/deception_prewritten/ and data/concepts/deception_ai_liar/
(committed, available on RunPod).

Usage (local only):
    python scripts/convert_prewritten_data.py
"""

import json
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Category keywords (from convert_deception_data.py)
CATEGORY_KEYWORDS = {
    "student": ["student", "homework", "assignment", "lecture", "professor", "teacher", "college", "class", "school", "grade"],
    "salesperson": ["salesperson", "sales", "customer", "product", "sell", "commission"],
    "employee": ["employee", "manager", "promotion", "project", "deadline", "report", "shipment", "company", "coworker"],
    "civilian": ["civilian", "officer", "police", "parking", "ticket", "speeding"],
    "medical": ["medical", "doctor", "patient", "hospital", "health", "nurse", "therapist", "prescription"],
    "tenant": ["tenant", "landlord", "apartment", "rent", "lease"],
    "child": ["child", "parent", "mom", "dad", "chore"],
}


def categorize_scenario(scenario: str) -> str:
    lower = scenario.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def convert_roleplaying():
    """Convert roleplaying dataset to scenarios.yaml."""
    source = os.path.join(
        PROJECT_ROOT, "deception-detection", "data", "roleplaying", "dataset.yaml"
    )
    output_dir = os.path.join(PROJECT_ROOT, "data", "concepts", "deception_prewritten")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading: {source}")
    with open(source, "r") as f:
        raw = yaml.safe_load(f)

    scenarios = []
    for i, s in enumerate(raw):
        scenarios.append({
            "pair_id": i,
            "scenario": s["scenario"],
            "question": s["question"],
            "answer_prefix": s.get("answer_prefix", ""),
            "deceptive_completion": s["deceptive_completion"],
            "honest_completion": s["honest_completion"],
            "category": categorize_scenario(s["scenario"]),
        })

    output_path = os.path.join(output_dir, "scenarios.yaml")
    with open(output_path, "w") as f:
        yaml.dump({"scenarios": scenarios}, f, default_flow_style=False, sort_keys=False, width=200)

    print(f"  Wrote {len(scenarios)} roleplaying scenarios to {output_path}")

    # Print category distribution
    cats = {}
    for s in scenarios:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")


def convert_ai_liar():
    """Convert AI Liar dataset to scenarios.yaml."""
    source = os.path.join(
        PROJECT_ROOT, "deception-detection", "data",
        "how_to_catch_an_ai_liar", "goal_directed_lying.json"
    )
    output_dir = os.path.join(PROJECT_ROOT, "data", "concepts", "deception_ai_liar")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nReading: {source}")
    with open(source, "r") as f:
        raw = json.load(f)

    scenarios = []
    pair_id = 0
    for entry in raw:
        if not entry.get("include_in_with_answers"):
            continue
        scenarios.append({
            "pair_id": pair_id,
            "deceive_instruction": entry["deceive_instruction"],
            "normal_instruction": entry["normal_instruction"],
            "question": entry["question"],
            "question_prefix": entry.get("question_prefix", "Human:"),
            "answer_prefix": entry.get("answer_prefix", "AI:"),
            "answer": entry["answer"],
            "category": "ai_liar",
        })
        pair_id += 1

    output_path = os.path.join(output_dir, "scenarios.yaml")
    with open(output_path, "w") as f:
        yaml.dump({"scenarios": scenarios}, f, default_flow_style=False, sort_keys=False, width=200)

    print(f"  Wrote {len(scenarios)} AI Liar scenarios to {output_path}")


def main():
    convert_roleplaying()
    convert_ai_liar()
    print("\nDone! These files are committed and available on RunPod.")


if __name__ == "__main__":
    main()
