"""
Apply LLM-as-judge labels to a sycophancy rollouts.jsonl.

Thin CLI wrapper around src/utils/judge.py. Resumable: re-running the same
command picks up where the previous run left off (skips already-labeled rollouts
in the output file).

Default backend: OpenRouter + Gemini 2.5 Flash (cheap and sufficient for binary
endorsement classification). Set OPENROUTER_API_KEY in your env.

Usage:
  # Stage 0 re-labeling (default paths, default model)
  python scripts/sycophancy_label.py --rollouts data/concepts/sycophancy_qa_v2/stage0_sanity/rollouts.jsonl

  # Smoke test on 5 rollouts to verify the judge before paying for the full run
  python scripts/sycophancy_label.py --rollouts <path> --max-rollouts 5

  # Dry run: print first two judge prompts, no API call
  python scripts/sycophancy_label.py --rollouts <path> --dry-run

  # Override provider / model
  python scripts/sycophancy_label.py --rollouts <path> --provider openai --model gpt-4o-mini
  python scripts/sycophancy_label.py --rollouts <path> --provider anthropic --model claude-haiku-4-5-20251001
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.judge import (
    DEFAULT_MODELS,
    build_judge_prompt,
    label_rollout,
    make_client,
)


REGEX_LABELS_FOR_TABLE = ["endorsed_correct", "endorsed_incorrect", "both", "ambiguous"]
JUDGE_LABELS_FOR_TABLE = ["endorsed_correct", "endorsed_incorrect", "neither"]


def load_rollouts(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                keys.add((r.get("source"), r.get("question"), r.get("framing")))
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", required=True,
                   help="Input rollouts.jsonl from a sycophancy stage")
    p.add_argument("--output", default=None,
                   help="Output path. Defaults to <rollouts_dir>/rollouts_labeled.jsonl")
    p.add_argument("--provider", default="openrouter",
                   choices=["openrouter", "openai", "anthropic"],
                   help="Judge backend (default: openrouter)")
    p.add_argument("--model", default=None,
                   help="Model id for the judge. Defaults per-provider: "
                        f"{DEFAULT_MODELS}")
    p.add_argument("--base-url", default=None,
                   help="Optional custom OpenAI-compatible base URL "
                        "(uses OPENAI_API_KEY when set)")
    p.add_argument("--max-rollouts", type=int, default=None,
                   help="Stop after N rollouts (smoke test)")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="max_tokens for the judge call (200 is plenty)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print first 2 judge prompts and exit (no API call)")
    args = p.parse_args()

    in_path = Path(args.rollouts)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    if not in_path.exists():
        print(f"ERROR: rollouts file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.parent / "rollouts_labeled.jsonl"
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    model = args.model or DEFAULT_MODELS[args.provider]

    rollouts = load_rollouts(in_path)
    print(f"Loaded {len(rollouts)} rollouts from {in_path}")

    if args.dry_run:
        print(f"\nDRY RUN — first 2 judge prompts (no API call):")
        print(f"Provider: {args.provider} | Model: {model}\n")
        for r in rollouts[:2]:
            print("=" * 72)
            print(f"[{r.get('source')} / {r.get('framing')}]   regex_label={r.get('label')}")
            print("-" * 72)
            print(build_judge_prompt(r))
            print()
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(out_path)
    if completed:
        print(f"  resume: {len(completed)} already labeled, will skip")

    pending = [r for r in rollouts
               if (r.get("source"), r.get("question"), r.get("framing")) not in completed]
    if args.max_rollouts:
        pending = pending[:args.max_rollouts]
        print(f"  smoke test: limiting to {args.max_rollouts}")

    client = make_client(args.provider, base_url=args.base_url)

    print(f"\nLabeling {len(pending)} rollouts")
    print(f"  provider: {args.provider}")
    print(f"  model:    {model}")
    print(f"  output:   {out_path}\n")

    by_judge = defaultdict(int)
    by_pair = defaultdict(int)
    n_errors = 0
    start = time.time()

    with open(out_path, "a") as f_out:
        for i, rollout in enumerate(pending):
            try:
                labeled = label_rollout(client, rollout, model, args.max_tokens)
                f_out.write(json.dumps(labeled) + "\n")
                f_out.flush()

                judge_label = labeled["label_judge"]
                regex_label = rollout.get("label", "?")
                by_judge[judge_label] += 1
                by_pair[(regex_label, judge_label)] += 1

                if (i + 1) % 10 == 0 or i == len(pending) - 1:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                    print(f"  {i+1}/{len(pending)}  ({rate:.1f}/min, ETA {eta:.1f}min) "
                          f"last: regex={regex_label} -> judge={judge_label}")
            except Exception as e:
                n_errors += 1
                print(f"  ERROR rollout {i} ({rollout.get('framing')}): "
                      f"{type(e).__name__}: {str(e)[:140]}")

    print("\n" + "=" * 60)
    print("JUDGE LABEL DISTRIBUTION")
    print("=" * 60)
    total = sum(by_judge.values())
    for name in JUDGE_LABELS_FOR_TABLE:
        n = by_judge.get(name, 0)
        pct = 100 * n / total if total else 0
        print(f"  {name:22s}  {n:4d}  ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("REGEX vs JUDGE — confusion (rows: regex, cols: judge)")
    print("=" * 60)
    title = "regex \\ judge"
    header = f"{title:<22s}" + "".join(f"{j:>22s}" for j in JUDGE_LABELS_FOR_TABLE) + f"{'total':>10s}"
    print(header)
    print("-" * len(header))
    for r_lbl in REGEX_LABELS_FOR_TABLE:
        row = f"{r_lbl:<22s}"
        row_total = 0
        for j_lbl in JUDGE_LABELS_FOR_TABLE:
            n = by_pair.get((r_lbl, j_lbl), 0)
            row_total += n
            row += f"{n:>22d}"
        if row_total > 0:
            row += f"{row_total:>10d}"
            print(row)

    if n_errors:
        print(f"\nWARN: {n_errors} rollout(s) errored. Re-run to retry just those.")

    print(f"\nResults: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
