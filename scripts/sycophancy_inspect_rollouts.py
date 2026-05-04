"""
Spot-check rollouts produced by sycophancy_stage0_sanity.py (or any other
sycophancy stage). Prints sample rollouts grouped by (framing, label) cell
with truncated reasoning + response so you can sanity-check what the model
actually said vs what the regex labeler decided.

Usage:
  uv run python scripts/sycophancy_inspect_rollouts.py
  uv run python scripts/sycophancy_inspect_rollouts.py --rollouts data/concepts/sycophancy_qa_v2/stage0_sanity/rollouts.jsonl
  uv run python scripts/sycophancy_inspect_rollouts.py --per-cell 3 --max-chars 400
  uv run python scripts/sycophancy_inspect_rollouts.py --filter-label ambiguous
  uv run python scripts/sycophancy_inspect_rollouts.py --show-thinking
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_ROLLOUTS = PROJECT_ROOT / "data/concepts/sycophancy_qa_v2/stage0_sanity/rollouts.jsonl"

FRAMING_ORDER = ["T1_neutral", "T3_user_wrong", "T4_user_right"]
LABEL_ORDER = ["endorsed_correct", "endorsed_incorrect", "both", "ambiguous"]


def truncate(s, n):
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + " [...]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", default=str(DEFAULT_ROLLOUTS),
                   help="Path to rollouts.jsonl")
    p.add_argument("--per-cell", type=int, default=2,
                   help="Number of examples to show per (framing, label) cell")
    p.add_argument("--max-chars", type=int, default=300,
                   help="Truncate response to this many chars")
    p.add_argument("--filter-label", default=None,
                   choices=LABEL_ORDER,
                   help="Only show rollouts with this label")
    p.add_argument("--filter-framing", default=None,
                   choices=FRAMING_ORDER,
                   help="Only show rollouts with this framing")
    p.add_argument("--filter-source", default=None,
                   help="Only show rollouts from this source dataset (trivia_qa/truthful_qa)")
    p.add_argument("--show-thinking", action="store_true",
                   help="Also print truncated reasoning trace")
    p.add_argument("--show-counts-only", action="store_true",
                   help="Print only the (framing, label) count grid")
    args = p.parse_args()

    path = Path(args.rollouts)
    if not path.exists():
        print(f"ERROR: rollouts file not found: {path}", file=sys.stderr)
        print("       Run scripts/sycophancy_stage0_sanity.py first.", file=sys.stderr)
        return 1

    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))

    # Filter
    filtered = rollouts
    if args.filter_label:
        filtered = [r for r in filtered if r["label"] == args.filter_label]
    if args.filter_framing:
        filtered = [r for r in filtered if r["framing"] == args.filter_framing]
    if args.filter_source:
        filtered = [r for r in filtered if r["source"] == args.filter_source]

    print(f"Loaded {len(rollouts)} rollouts from {path}")
    if len(filtered) != len(rollouts):
        print(f"After filtering: {len(filtered)} rollouts")

    # Build (framing, label) grid
    grid = defaultdict(int)
    for r in rollouts:
        grid[(r["framing"], r["label"])] += 1

    print("\n" + "=" * 72)
    print("COUNT GRID (framing x label)")
    print("=" * 72)
    header = f"{'framing':<18s}" + "".join(f"{l:>20s}" for l in LABEL_ORDER) + f"{'total':>10s}"
    print(header)
    print("-" * len(header))
    for framing in FRAMING_ORDER:
        row_total = sum(grid[(framing, l)] for l in LABEL_ORDER)
        row = f"{framing:<18s}"
        for label in LABEL_ORDER:
            count = grid[(framing, label)]
            pct = 100 * count / row_total if row_total else 0
            row += f"{count:>5d} ({pct:>4.1f}%) "
        row += f"{row_total:>10d}"
        print(row)

    if args.show_counts_only:
        return 0

    # Group filtered rollouts by (framing, label)
    by_cell = defaultdict(list)
    for r in filtered:
        by_cell[(r["framing"], r["label"])].append(r)

    print("\n" + "=" * 72)
    print(f"SAMPLE ROLLOUTS — up to {args.per_cell} per (framing, label) cell")
    print("=" * 72)

    for framing in FRAMING_ORDER:
        if args.filter_framing and framing != args.filter_framing:
            continue
        for label in LABEL_ORDER:
            if args.filter_label and label != args.filter_label:
                continue
            cell = by_cell[(framing, label)]
            if not cell:
                continue
            print(f"\n--- {framing} | {label} (n={len(cell)}) ---")
            for r in cell[: args.per_cell]:
                print(f"\n  [{r['source']}]  Q: {truncate(r['question'], 120)}")
                print(f"  correct={r['correct_answer']!r}  incorrect={r['incorrect_answer']!r}")
                print(f"  user said: {truncate(r['user_text'], 200)}")
                if args.show_thinking:
                    print(f"  <think>: {truncate(r.get('thinking', ''), args.max_chars)}")
                response = r.get("response") or r.get("thinking") or ""
                print(f"  response: {truncate(response, args.max_chars)}")
                print(f"  n_tokens={r.get('n_gen_tokens', '?')}  gen_seconds={r.get('gen_seconds', '?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
