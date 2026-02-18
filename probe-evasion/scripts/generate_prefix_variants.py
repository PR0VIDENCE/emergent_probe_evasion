"""
Generate prefix variants for handwritten anchor pairs (Group A).

Takes the 25 handwritten pairs from contrastive_batch_A_01.yaml and creates
7 additional files with non-default elicitation styles. Each variant copies
all 25 pairs with updated elicitation_style, pair_id (800+), group (A_prefix),
and base_pair_ref pointing back to the original.

Usage:
    python scripts/generate_prefix_variants.py
    python scripts/generate_prefix_variants.py --data-dir data/concepts/trees_qa_v2
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Non-default prefix styles (default is already on the originals)
PREFIX_VARIANTS = [
    "concise", "verbose", "eli5",
    "numbered_list", "no_keyword", "academic", "casual",
]

PFX_START_ID = 800


def main():
    parser = argparse.ArgumentParser(description="Generate prefix variants for Group A pairs")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/concepts/trees_qa_v2)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Input YAML file (default: contrastive_batch_A_01.yaml in data-dir)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2"
    input_file = Path(args.input_file) if args.input_file else data_dir / "contrastive_batch_A_01.yaml"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Create contrastive_batch_A_01.yaml with 25 handwritten pairs first.")
        sys.exit(1)

    with open(input_file, "r") as f:
        original_data = yaml.safe_load(f)

    original_pairs = original_data["pairs"]
    print(f"Loaded {len(original_pairs)} handwritten pairs from {input_file}")

    if len(original_pairs) != 25:
        print(f"WARNING: Expected 25 pairs, got {len(original_pairs)}")

    current_id = PFX_START_ID

    for variant_idx, style in enumerate(PREFIX_VARIANTS):
        variant_pairs = []

        for pair in original_pairs:
            new_pair = copy.deepcopy(pair)
            original_pair_id = pair["pair_id"]

            new_pair["pair_id"] = current_id
            new_pair["elicitation_style"] = style
            new_pair["group"] = "A_prefix"
            new_pair["base_pair_ref"] = f"hw_{original_pair_id}"
            # Preserve tree_topic if present
            if "tree_topic" not in new_pair:
                new_pair["tree_topic"] = "handwritten"

            variant_pairs.append(new_pair)
            current_id += 1

        variant_data = {"pairs": variant_pairs}
        batch_num = variant_idx + 1
        output_file = data_dir / f"contrastive_batch_PFX_{batch_num:02d}.yaml"

        with open(output_file, "w") as f:
            yaml.dump(variant_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"  {style}: {len(variant_pairs)} pairs -> {output_file} (IDs {variant_pairs[0]['pair_id']}-{variant_pairs[-1]['pair_id']})")

    total_variants = current_id - PFX_START_ID
    print(f"\nGenerated {total_variants} prefix variants (IDs {PFX_START_ID}-{current_id - 1})")
    print("Done!")


if __name__ == "__main__":
    main()
