"""
Head-to-head comparison of two SyA probe runs.

Used by sycophancy_truncation_ab.sh to diff a filtered-truncated run against
a with-truncated run, but generic enough for any two probe directories that
share a results.json schema.

Outputs a markdown report:
  - Sample-size delta (how many rollouts dropped)
  - Per-probe AUROC delta (val + OOD pooled, DiffMean + LR)
  - Calibration delta (TPR@FPR, catch-all FPR) parsed from each run's
    calibration/calibration_report.md

Usage:
  uv run python scripts/sycophancy_compare_probe_runs.py \\
      --run-a data/.../stage1_probes_filtered \\
      --run-b data/.../stage1_probes_with_truncated \\
      --output data/.../stage1_probes_ab_comparison.md
"""

import argparse
import json
import re
from pathlib import Path


def load_results(d):
    """Find and load results*.json under a probes directory."""
    candidates = sorted(d.glob("results*.json"))
    if not candidates:
        raise FileNotFoundError(f"no results*.json under {d}")
    with open(candidates[0]) as f:
        return json.load(f)


def fmt_num(v, digits=3):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "—"


def fmt_delta(x, y, digits=3):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return f"{x - y:+.{digits}f}"
    return "—"


def parse_calibration(d):
    """Pull TPR@FPR and 'catch-all' FPR rows for each probe section in calibration_report.md."""
    cands = list(d.glob("calibration*/calibration_report*.md"))
    if not cands:
        return []
    text = cands[0].read_text()
    parsed = []
    # Each probe gets its own "## Probe: `key`" section
    sections = text.split("\n## Probe: ")
    for sec in sections[1:]:
        first_line = sec.splitlines()[0]
        probe_name = first_line.strip().strip("`").strip()
        # Capture the entire DiffMean TPR section up to the next subheading
        # (the prior regex with `\n\n` stopped at the blank line right after
        # the header and never reached the table).
        dm_block = re.search(
            r"### TPR @ fixed FPR \(diffmean\)(.*?)(?=### TPR @ fixed FPR \(lr\)|### Top \d|\Z)",
            sec, flags=re.S,
        )
        catch = re.search(r"Catching ALL .*? costs FPR = ([\d.]+)%", sec)
        if not dm_block or not catch:
            continue
        rows = re.findall(r"\| (\d+)% \| .*? \| \*\*(\d+)%\*\* \|",
                          dm_block.group(0))
        tpr_at = {int(fpr): int(tpr) for fpr, tpr in rows}
        parsed.append({
            "probe": probe_name,
            "tpr_at_5": tpr_at.get(5),
            "tpr_at_10": tpr_at.get(10),
            "tpr_at_20": tpr_at.get(20),
            "catch_all_fpr": float(catch.group(1)),
        })
    return parsed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-a", required=True, help="Path to first probes directory (e.g. ..._filtered)")
    p.add_argument("--run-b", required=True, help="Path to second probes directory (e.g. ..._with_truncated)")
    p.add_argument("--label-a", default="A", help="Label for run A in the report")
    p.add_argument("--label-b", default="B", help="Label for run B in the report")
    p.add_argument("--output", required=True, help="Path to write the markdown report")
    p.add_argument("--top-k", type=int, default=8,
                   help="Show top-K probes by run A's OOD pooled DiffMean")
    args = p.parse_args()

    A = Path(args.run_a)
    B = Path(args.run_b)
    OUT = Path(args.output)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    a = load_results(A)
    b = load_results(B)

    # ---- Sample sizes (same across all probes within a run) ----
    k = next(iter(a))
    def szs(r):
        return {
            "train":  (r[k]["train"]["n_pos"], r[k]["train"]["n_neg"]),
            "val":    (r[k]["val"]["n_pos"], r[k]["val"]["n_neg"]),
            "ood":    (r[k]["ood_pooled"]["n_pos"], r[k]["ood_pooled"]["n_neg"]),
        }
    sa = szs(a)
    sb = szs(b)

    lines = []
    lines.append(f"# Probe-run comparison: {args.label_a} vs {args.label_b}\n")
    lines.append(f"**Run {args.label_a}** — `{A}`")
    lines.append(f"**Run {args.label_b}** — `{B}`\n")

    lines.append("## Sample sizes (identical across all (layer, position) pairs in a run)\n")
    lines.append(f"| split | {args.label_a} pos | {args.label_a} neg | {args.label_b} pos | {args.label_b} neg | Δpos | Δneg |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ["train", "val", "ood"]:
        ap, an = sa[name]
        bp, bn = sb[name]
        lines.append(f"| {name} | {ap} | {an} | {bp} | {bn} | {ap - bp:+d} | {an - bn:+d} |")
    lines.append("")
    lines.append(f"Negative Δ in run {args.label_a} = run {args.label_a} dropped that many rollouts.\n")

    # ---- Probe-by-probe AUROC ----
    shared = sorted(k for k in a if k in b)
    # Rank by run A's ood_pooled DiffMean
    def a_ood(key):
        v = a[key].get("ood_pooled", {}).get("diffmean_auroc")
        return v if isinstance(v, (int, float)) else -1
    shared.sort(key=a_ood, reverse=True)

    lines.append(f"## Top {args.top_k} probes by run {args.label_a}'s OOD pooled DiffMean\n")
    lines.append(
        f"| probe | {args.label_a} val DM | {args.label_b} val DM | Δ | "
        f"{args.label_a} OOD DM | {args.label_b} OOD DM | Δ | "
        f"{args.label_a} OOD LR | {args.label_b} OOD LR | Δ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in shared[:args.top_k]:
        ar, br = a[key], b[key]
        av = ar["val"]["diffmean_auroc"]; bv = br["val"]["diffmean_auroc"]
        ao = ar["ood_pooled"]["diffmean_auroc"]; bo = br["ood_pooled"]["diffmean_auroc"]
        al = ar["ood_pooled"]["lr_auroc"]; bl = br["ood_pooled"]["lr_auroc"]
        lines.append(
            f"| {key} | {fmt_num(av)} | {fmt_num(bv)} | {fmt_delta(av, bv)} | "
            f"{fmt_num(ao)} | {fmt_num(bo)} | {fmt_delta(ao, bo)} | "
            f"{fmt_num(al)} | {fmt_num(bl)} | {fmt_delta(al, bl)} |"
        )
    lines.append("")
    lines.append(f"**Δ = {args.label_a} − {args.label_b}**. Positive Δ on OOD = run {args.label_a} generalizes better.\n")

    # ---- Calibration delta ----
    ca = parse_calibration(A)
    cb = parse_calibration(B)
    if ca and cb:
        by_b = {x["probe"]: x for x in cb}
        lines.append("## Calibration delta (DiffMean, OOD)\n")
        lines.append(
            f"| probe | {args.label_a} TPR@5% | {args.label_b} TPR@5% | "
            f"{args.label_a} TPR@10% | {args.label_b} TPR@10% | "
            f"{args.label_a} catch-all FPR | {args.label_b} catch-all FPR | Δ catch-all |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for x in ca:
            y = by_b.get(x["probe"])
            def pct(v): return f"{v}%" if isinstance(v, (int, float)) else "—"
            if y is None:
                # Probe present in A but not B
                lines.append(
                    f"| {x['probe']} | {pct(x['tpr_at_5'])} | — | "
                    f"{pct(x['tpr_at_10'])} | — | "
                    f"{x['catch_all_fpr']:.1f}% | — | — |"
                )
                continue
            # Both present — full row
            a_catch = x["catch_all_fpr"]; b_catch = y["catch_all_fpr"]
            delta_catch = f"{a_catch - b_catch:+.1f}%"
            lines.append(
                f"| {x['probe']} | {pct(x['tpr_at_5'])} | {pct(y['tpr_at_5'])} | "
                f"{pct(x['tpr_at_10'])} | {pct(y['tpr_at_10'])} | "
                f"{a_catch:.1f}% | {b_catch:.1f}% | {delta_catch} |"
            )
        lines.append("")
        lines.append(f"Δ catch-all < 0 means run {args.label_a} needs lower FPR to catch all OOD positives — i.e., better calibration.\n")
    else:
        lines.append("(No calibration reports found in one or both runs — skipping calibration delta.)\n")

    OUT.write_text("\n".join(lines))
    print(f"Wrote comparison report → {OUT}")


if __name__ == "__main__":
    main()
