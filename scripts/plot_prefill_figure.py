"""
Generate figure comparing deception prefill experiment conditions.

Shows mean probe confidence (with bootstrap 95% CIs) for 5 reasoning-trace
conditions: naked deception (QwQ), moralizing (QwQ), moralizing (Claude),
self-deceptive (QwQ), and honest reasoning (Claude).

Usage:
    python scripts/plot_prefill_figure.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "figures"
DATA_FILE = PROJECT_ROOT / "data" / "outputs" / "deception_prefill_experiments" / "all_trials_combined.jsonl"

N_BOOTSTRAP = 10000

# (condition, source) tuples and their display labels
CONDITIONS = [
    ("naked_deception", "qwq"),
    ("moral_reframing", "qwq"),
    ("moral_reframing", "claude"),
    ("self_deceptive", "qwq"),
    ("honest_reasoning_deceptive", "claude"),
]

LABELS = [
    "Baseline\n(QwQ)",
    "Moralizing\n(QwQ)",
    "Moralizing\n(Claude)",
    "Self-Deceptive\n(QwQ)",
    "Honest Reasoning\n(Claude)",
]

COLORS = [
    "#8B3A00",   # burnt orange — naked deception (baseline)
    "#D4A84B",   # gold — qwq moralizing
    "#5B8C5A",   # muted green — claude moralizing
    "#8C4B6E",   # muted plum — qwq self-deceptive
    "#2B4162",   # navy — claude honest reasoning
]


def load_trials():
    """Load all trials from combined JSONL."""
    trials = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def get_confidences(trials, condition, source):
    """Extract mean_confidence values for a specific condition+source."""
    return np.array([
        t["mean_confidence"] for t in trials
        if t["condition"] == condition and t["source"] == source
    ])


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap 95% CI. Returns (mean, lower_err, upper_err)."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    mean = np.mean(values)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return mean, mean - lo, hi - mean


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading trial data...")
    trials = load_trials()
    print(f"  Loaded {len(trials)} trials")

    fig, ax = plt.subplots(figsize=(7, 5.25))

    x = np.arange(len(CONDITIONS))
    means = []
    err_lo = []
    err_hi = []

    for (condition, source), label in zip(CONDITIONS, LABELS):
        values = get_confidences(trials, condition, source)
        m, lo, hi = bootstrap_ci(values)
        means.append(m)
        err_lo.append(lo)
        err_hi.append(hi)
        print(f"  {label.replace(chr(10), ' ')}: n={len(values)}, "
              f"mean={m:.4f} [{m - lo:.4f}, {m + hi:.4f}]")

    bars = ax.bar(x, means, 0.6,
                  yerr=[err_lo, err_hi], capsize=4,
                  color=COLORS, alpha=0.9,
                  edgecolor="white", linewidth=0.5,
                  error_kw={"ecolor": "#333333", "capthick": 1})

    ax.set_ylabel("Mean Probe Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("Probe Detection Across Reasoning Trace Styles")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"fig4_prefill_reasoning.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
    print(f"\n  Saved fig4_prefill_reasoning.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    main()
