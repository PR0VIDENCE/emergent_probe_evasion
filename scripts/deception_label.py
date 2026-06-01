"""
Label deception-evasion rollouts with the LLM judge.

Parallel to sycophancy_label.py but uses the deception judge (compares
free-form response to honest/deceptive reference completions).

Each input rollout must contain:
  scenario, question, response, honest_completion, deceptive_completion

Output adds: label_judge ∈ {honest, deceptive, neither}, judge_letter,
            judge_reason, judge_model.

Resumable: skips rollouts that already have a label_judge field in the
output file.

Usage:
  python scripts/deception_label.py --rollouts <path/to/rollouts.jsonl>
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.judge import (
    DEFAULT_MODELS,
    label_deception_rollout,
    make_client,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", required=True,
                   help="Input rollouts.jsonl with scenario/question/response/refs")
    p.add_argument("--output", default=None,
                   help="Output path. Default: <rollouts_dir>/rollouts_labeled.jsonl")
    p.add_argument("--provider", default="openrouter",
                   choices=["openrouter", "openai", "anthropic"])
    p.add_argument("--model", default=None,
                   help="Judge model id. Default per-provider in DEFAULT_MODELS.")
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-rollouts", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.is_absolute():
        rollouts_path = PROJECT_ROOT / rollouts_path
    out_path = Path(args.output) if args.output else rollouts_path.with_name("rollouts_labeled.jsonl")

    # Load all input rollouts
    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    print(f"Loaded {len(rollouts)} rollouts from {rollouts_path}")

    # Resume: skip already-labeled rollout_ids
    already = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                rid = r.get("rollout_id") or f"{r['question_id']}_{r.get('framing','')}_{r['system_prompt_id']}"
                already[rid] = r
        print(f"  resumed: {len(already)} already labeled in {out_path.name}")

    model = args.model or DEFAULT_MODELS[args.provider]
    print(f"Using judge: provider={args.provider} model={model}")

    pending = [r for r in rollouts if
               (r.get("rollout_id") or f"{r['question_id']}_{r.get('framing','')}_{r['system_prompt_id']}")
               not in already]
    if args.max_rollouts:
        pending = pending[:args.max_rollouts]
    print(f"Pending to label: {len(pending)}")

    if args.dry_run:
        print("\nDRY RUN — first 2 judge prompts:")
        from src.utils.judge import build_deception_judge_prompt
        for r in pending[:2]:
            print("\n" + "=" * 60)
            print(build_deception_judge_prompt(r))
        return

    client = make_client(args.provider, base_url=args.base_url)

    start = time.time()
    with open(out_path, "a") as f:
        # First flush previously-labeled entries that we resumed
        if not already and out_path.exists():
            pass
        for i, r in enumerate(pending):
            labeled = label_deception_rollout(client, r, model, args.max_tokens)
            f.write(json.dumps(labeled) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(pending):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                eta = (len(pending) - (i + 1)) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] last: {labeled['label_judge']}  "
                      f"({rate:.0f}/min, ETA {eta:.1f}min)")

    print(f"\nWrote labels to {out_path}")


if __name__ == "__main__":
    main()
