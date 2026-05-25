#!/bin/bash
#
# A/B test: how much of the current probe results is polluted by truncated
# rollouts (model hit max_new_tokens during <think>, never produced an answer)?
#
# Run A (filtered):  --exclude-truncated (default) → stage1_probes_filtered/
# Run B (with):      --no-exclude-truncated         → stage1_probes_with_truncated/
#
# Then calibration on both, plus a head-to-head delta report.
# Total runtime: ~10 min on RunPod (probes are CPU-only).
#
# Usage:
#   bash scripts/sycophancy_truncation_ab.sh
#
# Output:
#   stage1_probes_filtered/results.json + summary.md + calibration/
#   stage1_probes_with_truncated/results.json + summary.md + calibration/
#   stage1_probes_ab_comparison.md  (the head-to-head report)

set -e
set -u
set -o pipefail

CONFIG=configs/experiments/qa_probe_training_sycophancy.yaml
DATA_DIR=data/concepts/sycophancy_qa_v2
ACTS="$DATA_DIR/stage1_activations"
PROBE_A="$DATA_DIR/stage1_probes_filtered"
PROBE_B="$DATA_DIR/stage1_probes_with_truncated"
REPORT="$DATA_DIR/stage1_probes_ab_comparison.md"

start_time=$(date +%s)
step() {
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)]  $1"
    echo "================================================================"
}

step "A: train probes with --exclude-truncated (default ON)"
uv run python scripts/sycophancy_train_probes.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --output-dir "$PROBE_A"

step "B: train probes with --no-exclude-truncated (diagnostic)"
uv run python scripts/sycophancy_train_probes.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --output-dir "$PROBE_B" \
    --no-exclude-truncated

step "calibration A (filtered)"
uv run python scripts/sycophancy_calibration_analysis.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --probes-dir "$PROBE_A"

step "calibration B (with truncated)"
uv run python scripts/sycophancy_calibration_analysis.py \
    --config "$CONFIG" \
    --activations-dir "$ACTS" \
    --probes-dir "$PROBE_B" \
    --no-exclude-truncated

step "writing head-to-head comparison report"
uv run python - <<PYEOF
"""Head-to-head comparison of the two probe runs."""
import json, re
from pathlib import Path

A = Path("$PROBE_A")
B = Path("$PROBE_B")
REPORT = Path("$REPORT")

# --- Load results.json from both runs ---
def load_results(d):
    p = d / "results.json"
    if not p.exists():
        # auto-archived suffix variations like results (1).json
        cands = sorted(d.glob("results*.json"))
        if cands: p = cands[0]
    with open(p) as f:
        return json.load(f)

a = load_results(A)
b = load_results(B)

# --- Sample sizes (use any probe key — train/val/OOD ns are the same) ---
k = next(iter(a))
a_n = (a[k]["train"]["n_pos"], a[k]["train"]["n_neg"],
       a[k]["val"]["n_pos"], a[k]["val"]["n_neg"],
       a[k]["ood_pooled"]["n_pos"], a[k]["ood_pooled"]["n_neg"])
b_n = (b[k]["train"]["n_pos"], b[k]["train"]["n_neg"],
       b[k]["val"]["n_pos"], b[k]["val"]["n_neg"],
       b[k]["ood_pooled"]["n_pos"], b[k]["ood_pooled"]["n_neg"])

lines = ["# Truncation A/B comparison\n"]
lines.append("**Run A** (`stage1_probes_filtered/`) — \`--exclude-truncated\` ON (default)")
lines.append("**Run B** (`stage1_probes_with_truncated/`) — \`--no-exclude-truncated\` (diagnostic)\n")
lines.append("## Sample sizes (same across all probes)\n")
lines.append("| split | A pos | A neg | B pos | B neg | Δpos | Δneg |")
lines.append("|---|---|---|---|---|---|---|")
for name, ia, ib in [("train", 0, 0), ("val", 2, 2), ("ood_pooled", 4, 4)]:
    ap, an = a_n[ia], a_n[ia+1]
    bp, bn = b_n[ib], b_n[ib+1]
    lines.append(f"| {name} | {ap} | {an} | {bp} | {bn} | {bp-ap:+d} | {bn-an:+d} |")
lines.append("")
lines.append("Negative Δpos/Δneg means run A dropped that many truncated rollouts.\n")

# --- Side-by-side AUROC for selected probes ---
of_interest = sorted([k for k in a if k in b],
                     key=lambda k: -a[k].get("ood_pooled", {}).get("diffmean_auroc") or 0)[:8]
lines.append("## Top 8 by A's OOD pooled DiffMean — head-to-head\n")
lines.append("| probe | A val DM | B val DM | Δ | A OOD DM | B OOD DM | Δ | A OOD LR | B OOD LR | Δ |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
def fmt(v): return f"{v:.3f}" if isinstance(v,(int,float)) else "—"
def diff(x,y):
    if isinstance(x,(int,float)) and isinstance(y,(int,float)):
        return f"{x-y:+.3f}"
    return "—"
for k in of_interest:
    ar = a[k]; br = b[k]
    av = ar["val"]["diffmean_auroc"]; bv = br["val"]["diffmean_auroc"]
    ao = ar["ood_pooled"]["diffmean_auroc"]; bo = br["ood_pooled"]["diffmean_auroc"]
    al = ar["ood_pooled"]["lr_auroc"]; bl = br["ood_pooled"]["lr_auroc"]
    lines.append(f"| {k} | {fmt(av)} | {fmt(bv)} | {diff(av,bv)} | {fmt(ao)} | {fmt(bo)} | {diff(ao,bo)} | {fmt(al)} | {fmt(bl)} | {diff(al,bl)} |")
lines.append("")
lines.append("**Δ = A − B**. Positive Δ on OOD means filtering truncated improved generalization.\n")

# --- Parse calibration_report.md from both — pull TPR@FPR + "catch all" FPR ---
def parse_calibration(d):
    report_path = d / "calibration" / "calibration_report.md"
    if not report_path.exists():
        cands = list(d.glob("calibration*/calibration_report*.md"))
        if cands: report_path = cands[0]
    text = report_path.read_text()
    # Pick the first probe section's TPR@FPR and catch-all blocks (DiffMean)
    sections = text.split("\n## Probe: ")
    parsed = []
    for sec in sections[1:]:
        probe_name = sec.splitlines()[0].strip(" \`")
        # Pull the DiffMean TPR table (first occurrence)
        dm_block = re.search(r"### TPR @ fixed FPR \(diffmean\)\s*\n.*?\n\n", sec, re.S)
        catch_block = re.search(r"Catching ALL .* costs FPR = ([\d\.]+)%", sec)
        if not dm_block or not catch_block: continue
        # Rows look like: | 5% | -X.X | **57%** | 4/7 | ...
        rows = re.findall(r"\| (\d+%) \| .*? \| \*\*(\d+)%\*\* \|", dm_block.group(0))
        tpr_at = {fpr: int(tpr) for fpr, tpr in rows}
        parsed.append({"probe": probe_name, "tpr_at_5": tpr_at.get("5%"), "tpr_at_10": tpr_at.get("10%"),
                       "tpr_at_20": tpr_at.get("20%"), "catch_all_fpr": float(catch_block.group(1))})
    return parsed

ca = parse_calibration(A)
cb = parse_calibration(B)
if ca and cb:
    lines.append("## Calibration delta (DiffMean, OOD)\n")
    lines.append("| probe | A TPR@5% | B TPR@5% | A TPR@10% | B TPR@10% | A catch-all FPR | B catch-all FPR | Δ catch-all |")
    lines.append("|---|---|---|---|---|---|---|---|")
    by_probe_b = {x["probe"]: x for x in cb}
    for x in ca:
        y = by_probe_b.get(x["probe"], {})
        delta_catch = f"{x['catch_all_fpr'] - y['catch_all_fpr']:+.1f}%" if y else "—"
        lines.append(f"| {x['probe']} | {x['tpr_at_5']}% | {y.get('tpr_at_5','—')}% | "
                     f"{x['tpr_at_10']}% | {y.get('tpr_at_10','—')}% | "
                     f"{x['catch_all_fpr']:.1f}% | {y.get('catch_all_fpr','—'):.1f}% | {delta_catch} |")
    lines.append("")
    lines.append("Negative Δ catch-all means filtering reduced the FPR needed to catch all OOD positives "
                 "— direct evidence the truncated rollouts were defining the FP threshold.\n")

REPORT.write_text("\n".join(lines))
print(f"  wrote {REPORT}")
PYEOF

duration=$(( $(date +%s) - start_time ))
echo ""
echo "================================================================"
printf "A/B complete in %dm %ds\n" $(( duration / 60 )) $(( duration % 60 ))
echo "================================================================"
echo "Outputs:"
echo "  $PROBE_A/ (filtered)"
echo "  $PROBE_B/ (with truncated)"
echo "  $REPORT  (head-to-head)"
