"""
Validate the v2 contrastive question pair dataset.

Checks:
- All YAML files parse correctly with required fields
- No duplicate pair_id values across all batch files
- No duplicate non-tree domains (for groups B, E)
- Group distribution matches targets (25/375/150/125/125/175)
- Elicitation style distribution roughly balanced
- Keyword screening: flags non-tree questions containing tree-related words
- Question length validation (10-200 words)

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data-dir data/concepts/trees_qa_v2
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tree-related keywords that should NOT appear in non-tree questions
TREE_KEYWORDS = [
    r"\btree\b", r"\btrees\b", r"\bforest\b", r"\bforests\b",
    r"\bwood\b", r"\bwoods\b", r"\bbark\b", r"\bleaf\b", r"\bleaves\b",
    r"\broot\b", r"\broots\b", r"\bbranch\b", r"\bbranches\b",
    r"\btrunk\b", r"\btrunks\b", r"\bcanopy\b", r"\bsapling\b",
    r"\blumber\b", r"\btimber\b",
]

# Expected group distribution
EXPECTED_COUNTS = {
    "A": 25,
    "B": 375,
    "C": 150,
    "D": 125,
    "E": 125,
    "A_prefix": 175,
}

REQUIRED_FIELDS = [
    "pair_id", "tree_question", "non_tree_question",
    "question_type", "non_tree_domain", "elicitation_style",
]

VALID_QUESTION_TYPES = {"factual", "explanatory", "comparative", "process"}

VALID_ELICITATION_STYLES = {
    "default", "concise", "verbose", "eli5",
    "numbered_list", "no_keyword", "academic", "casual",
    "explicit_mention",  # backward compat alias
}


def load_all_pairs(data_dir: Path) -> list:
    """Load all pairs from all batch YAML files."""
    batch_files = sorted(data_dir.glob("contrastive_batch_*.yaml"))
    if not batch_files:
        print(f"ERROR: No contrastive_batch_*.yaml files found in {data_dir}")
        return []

    all_pairs = []
    for batch_file in batch_files:
        try:
            with open(batch_file, "r") as f:
                data = yaml.safe_load(f)
            pairs = data.get("pairs", [])
            for pair in pairs:
                pair["_source_file"] = batch_file.name
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"ERROR: Failed to parse {batch_file}: {e}")

    print(f"Loaded {len(all_pairs)} pairs from {len(batch_files)} batch files")
    return all_pairs


def check_required_fields(pairs: list) -> int:
    """Check all pairs have required fields."""
    errors = 0
    for pair in pairs:
        for field in REQUIRED_FIELDS:
            if field not in pair:
                print(f"  MISSING FIELD: pair_id={pair.get('pair_id', '?')} missing '{field}' (file: {pair.get('_source_file')})")
                errors += 1
    return errors


def check_duplicate_ids(pairs: list) -> int:
    """Check for duplicate pair_id values."""
    ids = [p["pair_id"] for p in pairs if "pair_id" in p]
    counter = Counter(ids)
    dupes = {k: v for k, v in counter.items() if v > 1}
    if dupes:
        print(f"  DUPLICATE pair_ids: {dupes}")
    return len(dupes)


def check_duplicate_domains(pairs: list) -> int:
    """Check for duplicate non-tree domains in groups B and E."""
    domains_be = []
    for p in pairs:
        group = p.get("group", "")
        if group in ("B", "E"):
            domains_be.append(p.get("non_tree_domain", "").lower().strip())

    counter = Counter(domains_be)
    dupes = {k: v for k, v in counter.items() if v > 1}
    if dupes:
        print(f"  DUPLICATE domains (Groups B+E): {len(dupes)} duplicates")
        for d, c in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
            print(f"    '{d}' appears {c} times")
    return len(dupes)


def check_group_distribution(pairs: list) -> int:
    """Check group distribution matches targets."""
    groups = Counter(p.get("group", "unknown") for p in pairs)
    errors = 0
    print(f"  Group distribution:")
    for group, expected in EXPECTED_COUNTS.items():
        actual = groups.get(group, 0)
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            errors += 1
        print(f"    {group}: {actual}/{expected} {status}")

    unknown = groups.get("unknown", 0)
    if unknown > 0:
        print(f"    unknown: {unknown} (ERROR - missing group field)")
        errors += 1

    return errors


def check_elicitation_balance(pairs: list) -> int:
    """Check elicitation style distribution is roughly balanced."""
    # Only check generated pairs (not prefix variants which are intentionally one-style-each)
    generated = [p for p in pairs if p.get("group") not in ("A", "A_prefix")]
    styles = Counter(p.get("elicitation_style", "default") for p in generated)

    print(f"  Elicitation style distribution (generated pairs only):")
    total = len(generated)
    expected = total / len(VALID_ELICITATION_STYLES) if total > 0 else 0
    warnings = 0
    for style in sorted(VALID_ELICITATION_STYLES):
        count = styles.get(style, 0)
        pct = count / total * 100 if total > 0 else 0
        deviation = abs(count - expected) / expected * 100 if expected > 0 else 0
        flag = " WARNING (>50% deviation)" if deviation > 50 else ""
        if flag:
            warnings += 1
        print(f"    {style}: {count} ({pct:.1f}%){flag}")

    return warnings


def check_question_types(pairs: list) -> int:
    """Check question_type values are valid."""
    errors = 0
    for pair in pairs:
        qt = pair.get("question_type")
        if qt and qt not in VALID_QUESTION_TYPES:
            print(f"  INVALID question_type: '{qt}' in pair {pair.get('pair_id')}")
            errors += 1
    return errors


def check_keyword_screening(pairs: list) -> int:
    """Flag non-tree questions that contain tree-related keywords."""
    # Groups where keyword presence is expected/acceptable
    exempt_groups = {"D", "E"}  # D uses tree vocabulary; E is about trees

    flagged = 0
    for pair in pairs:
        group = pair.get("group", "")
        if group in exempt_groups:
            continue

        non_tree_q = pair.get("non_tree_question", "")
        for kw_pattern in TREE_KEYWORDS:
            match = re.search(kw_pattern, non_tree_q, re.IGNORECASE)
            if match:
                print(f"  KEYWORD WARNING: pair {pair.get('pair_id')} (group {group}): "
                      f"non-tree Q contains '{match.group()}': \"{non_tree_q[:80]}...\"")
                flagged += 1
                break  # One flag per question is enough

    return flagged


def check_question_lengths(pairs: list) -> int:
    """Check question lengths are within acceptable range (10-200 words)."""
    warnings = 0
    for pair in pairs:
        for field in ("tree_question", "non_tree_question"):
            q = pair.get(field, "")
            word_count = len(q.split())
            if word_count < 10:
                print(f"  TOO SHORT ({word_count} words): pair {pair.get('pair_id')} {field}: \"{q[:60]}\"")
                warnings += 1
            elif word_count > 200:
                print(f"  TOO LONG ({word_count} words): pair {pair.get('pair_id')} {field}: \"{q[:60]}...\"")
                warnings += 1
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate v2 contrastive question pair dataset")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/concepts/trees_qa_v2)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "concepts" / "trees_qa_v2"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)
    print(f"Directory: {data_dir}\n")

    pairs = load_all_pairs(data_dir)
    if not pairs:
        sys.exit(1)

    total_errors = 0
    total_warnings = 0

    print("\n1. Required fields check:")
    e = check_required_fields(pairs)
    total_errors += e
    if e == 0:
        print("  OK")

    print("\n2. Duplicate pair_id check:")
    e = check_duplicate_ids(pairs)
    total_errors += e
    if e == 0:
        print("  OK - all pair_ids unique")

    print("\n3. Duplicate domain check (Groups B+E):")
    e = check_duplicate_domains(pairs)
    total_warnings += e
    if e == 0:
        print("  OK - all domains unique")

    print("\n4. Group distribution check:")
    e = check_group_distribution(pairs)
    total_errors += e

    print("\n5. Elicitation style balance:")
    w = check_elicitation_balance(pairs)
    total_warnings += w

    print("\n6. Question type validation:")
    e = check_question_types(pairs)
    total_errors += e
    if e == 0:
        print("  OK - all question types valid")

    print("\n7. Keyword screening (non-tree questions):")
    w = check_keyword_screening(pairs)
    total_warnings += w
    if w == 0:
        print("  OK - no tree-related keywords found in non-tree questions")

    print("\n8. Question length validation:")
    w = check_question_lengths(pairs)
    total_warnings += w
    if w == 0:
        print("  OK - all questions within 10-200 word range")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")

    if total_errors > 0:
        print("\n  VALIDATION FAILED - fix errors before proceeding")
        sys.exit(1)
    elif total_warnings > 0:
        print("\n  VALIDATION PASSED with warnings - review flagged items")
    else:
        print("\n  VALIDATION PASSED")


if __name__ == "__main__":
    main()
