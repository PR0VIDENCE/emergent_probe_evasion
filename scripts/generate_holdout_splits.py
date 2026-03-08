"""
Generate per-category holdout split files for leave-one-category-out evaluation.

For each category in the roleplaying deception dataset:
  - All pairs of that category → test
  - Remaining pairs → 89/11 train/val split (seed=42)
  - Output: data/concepts/deception_prewritten/split_holdout_{category}.yaml

Categories: student, salesperson, employee, civilian, medical, tenant, other
(child is skipped — only 2 pairs)

Usage:
    python scripts/generate_holdout_splits.py
"""

import os
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# Skip categories with too few pairs
MIN_PAIRS = 5

SEED = 42
TRAIN_RATIO = 0.89  # of non-holdout pairs
# VAL_RATIO = 0.11 (remainder)

SOURCE_PATH = os.path.join(
    PROJECT_ROOT, "deception-detection", "data", "roleplaying", "dataset.yaml"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "concepts", "deception_prewritten")


def categorize_scenario(scenario: str) -> str:
    """Auto-categorize a scenario by keyword matching."""
    lower = scenario.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def main():
    print(f"Reading source dataset: {SOURCE_PATH}")
    with open(SOURCE_PATH, "r") as f:
        scenarios = yaml.safe_load(f)

    print(f"Found {len(scenarios)} scenarios")

    # Categorize all scenarios
    pair_categories = {}
    cat_counts = {}
    for i, s in enumerate(scenarios):
        cat = categorize_scenario(s["scenario"])
        pair_categories[i] = cat
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        skip_str = " (SKIP — too few)" if count < MIN_PAIRS else ""
        print(f"  {cat}: {count}{skip_str}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate holdout split for each category
    categories_to_process = [cat for cat, count in cat_counts.items() if count >= MIN_PAIRS]
    print(f"\nGenerating holdout splits for {len(categories_to_process)} categories...")

    for holdout_cat in sorted(categories_to_process):
        holdout_ids = [pid for pid, cat in pair_categories.items() if cat == holdout_cat]
        non_holdout_ids = [pid for pid, cat in pair_categories.items() if cat != holdout_cat]

        # Split non-holdout into train/val
        rng = random.Random(SEED)
        shuffled = list(non_holdout_ids)
        rng.shuffle(shuffled)

        n_train = int(len(shuffled) * TRAIN_RATIO)
        train_ids = sorted(shuffled[:n_train])
        val_ids = sorted(shuffled[n_train:])

        # Build assignments
        assignments = {}
        for pid in train_ids:
            assignments[pid] = "train"
        for pid in val_ids:
            assignments[pid] = "val"
        for pid in holdout_ids:
            assignments[pid] = "test"

        split_data = {
            "description": f"Leave-one-category-out: holdout={holdout_cat} ({len(holdout_ids)} pairs)",
            "seed": SEED,
            "holdout_category": holdout_cat,
            "strategy": {
                "holdout_pairs": len(holdout_ids),
                "train_pairs": len(train_ids),
                "val_pairs": len(val_ids),
            },
            "assignments": assignments,
        }

        output_path = os.path.join(OUTPUT_DIR, f"split_holdout_{holdout_cat}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(split_data, f, default_flow_style=False, sort_keys=False)

        print(f"  {holdout_cat}: test={len(holdout_ids)}, train={len(train_ids)}, val={len(val_ids)} → {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
