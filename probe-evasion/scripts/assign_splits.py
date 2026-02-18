"""
Assign train/val/test splits for the v2 dataset.

Split strategy:
- 20 handwritten pairs (pair_id 0-19) + their prefix variants -> train
- 5 handwritten pairs (pair_id 20-24) + their prefix variants -> test
- 775 generated pairs (pair_id 25-799) -> 80/10/10 pair-aware shuffle (seed=42)

Outputs split_assignment.yaml with pair-to-split mapping.

Usage:
    python scripts/assign_splits.py
    python scripts/assign_splits.py --data-dir data/concepts/trees_qa_v2
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Handwritten pair IDs
HW_TRAIN_IDS = list(range(0, 20))   # pair_id 0-19
HW_TEST_IDS = list(range(20, 25))   # pair_id 20-24

# Generated pair ID range
GENERATED_START = 25
GENERATED_END = 799

# Prefix variant start
PFX_START = 800


def load_all_pairs(data_dir: Path) -> list:
    """Load all pairs from batch YAML files."""
    batch_files = sorted(data_dir.glob("contrastive_batch_*.yaml"))
    all_pairs = []
    for bf in batch_files:
        with open(bf, "r") as f:
            data = yaml.safe_load(f)
        all_pairs.extend(data.get("pairs", []))
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Assign train/val/test splits")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/concepts/trees_qa_v2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    pairs = load_all_pairs(data_dir)
    print(f"Loaded {len(pairs)} total pairs")

    # Build pair_id -> group and base_pair_ref mapping
    pair_info = {}
    for p in pairs:
        pid = p["pair_id"]
        pair_info[pid] = {
            "group": p.get("group", "unknown"),
            "base_pair_ref": p.get("base_pair_ref"),
        }

    # Classify pair IDs
    all_ids = sorted(pair_info.keys())
    hw_ids = [pid for pid in all_ids if pid in HW_TRAIN_IDS or pid in HW_TEST_IDS]
    pfx_ids = [pid for pid in all_ids if pid >= PFX_START]
    gen_ids = [pid for pid in all_ids if GENERATED_START <= pid <= GENERATED_END]

    print(f"  Handwritten: {len(hw_ids)}")
    print(f"  Generated: {len(gen_ids)}")
    print(f"  Prefix variants: {len(pfx_ids)}")

    # Build split assignment
    split_assignment = {}

    # Handwritten pairs
    for pid in HW_TRAIN_IDS:
        if pid in pair_info:
            split_assignment[pid] = "train"
    for pid in HW_TEST_IDS:
        if pid in pair_info:
            split_assignment[pid] = "test"

    # Prefix variants follow their base pair
    for pid in pfx_ids:
        info = pair_info[pid]
        base_ref = info.get("base_pair_ref", "")
        if base_ref:
            # Extract original pair_id from "hw_N" format
            try:
                base_id = int(base_ref.replace("hw_", ""))
                if base_id in HW_TRAIN_IDS:
                    split_assignment[pid] = "train"
                elif base_id in HW_TEST_IDS:
                    split_assignment[pid] = "test"
                else:
                    split_assignment[pid] = "train"  # fallback
            except ValueError:
                split_assignment[pid] = "train"  # fallback
        else:
            split_assignment[pid] = "train"  # fallback

    # Generated pairs: 80/10/10 pair-aware shuffle
    rng = random.Random(args.seed)
    shuffled_gen = list(gen_ids)
    rng.shuffle(shuffled_gen)

    n = len(shuffled_gen)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    for pid in shuffled_gen[:n_train]:
        split_assignment[pid] = "train"
    for pid in shuffled_gen[n_train:n_train + n_val]:
        split_assignment[pid] = "val"
    for pid in shuffled_gen[n_train + n_val:]:
        split_assignment[pid] = "test"

    # Count splits
    from collections import Counter
    split_counts = Counter(split_assignment.values())
    print(f"\nSplit assignment:")
    for split_name in ["train", "val", "test"]:
        count = split_counts.get(split_name, 0)
        print(f"  {split_name}: {count} pairs ({count * 2} prompts)")

    # Count by group x split
    print(f"\nPer-group breakdown:")
    group_split_counts = {}
    for pid, split_name in split_assignment.items():
        group = pair_info.get(pid, {}).get("group", "unknown")
        key = (group, split_name)
        group_split_counts[key] = group_split_counts.get(key, 0) + 1

    groups = sorted(set(pair_info[pid]["group"] for pid in split_assignment if pid in pair_info))
    header = f"  {'Group':<12} {'train':>8} {'val':>8} {'test':>8} {'total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for group in groups:
        train_c = group_split_counts.get((group, "train"), 0)
        val_c = group_split_counts.get((group, "val"), 0)
        test_c = group_split_counts.get((group, "test"), 0)
        total = train_c + val_c + test_c
        print(f"  {group:<12} {train_c:>8} {val_c:>8} {test_c:>8} {total:>8}")

    # Save
    output = {
        "description": "Pre-assigned train/val/test splits for v2 dataset",
        "seed": args.seed,
        "strategy": {
            "handwritten_train": HW_TRAIN_IDS,
            "handwritten_test": HW_TEST_IDS,
            "generated_split_ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "prefix_variants": "follow base pair assignment",
        },
        "assignments": {int(k): v for k, v in sorted(split_assignment.items())},
    }

    output_path = data_dir / "split_assignment.yaml"
    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
