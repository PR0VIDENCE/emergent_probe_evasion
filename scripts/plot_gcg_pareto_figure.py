"""
Generate figure showing GCG Pareto frontier between fluency and probe evasion.

GCG can greedily craft reasoning traces that fool probes, but at a harsh
fluency cost — producing a Pareto frontier between coherence and evasion.

Usage:
    python scripts/plot_gcg_pareto_figure.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "figures"
DATA_FILE = PROJECT_ROOT / "data" / "outputs" / "bpj" / "bpj_result.json"


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

    print("Loading GCG results...")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    frontier = data["pareto_frontier"]
    frac_fluent = np.array([p["frac_fluent"] for p in frontier])
    score = np.array([p["score"] for p in frontier])

    print(f"  {len(frontier)} Pareto frontier points")
    print(f"  Fluency range: {frac_fluent.min():.3f} - {frac_fluent.max():.3f}")
    print(f"  Score range:   {score.min():.4f} - {score.max():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.25))

    ax.plot(frac_fluent, score, "-", color="#2B4162", linewidth=2.5, alpha=0.9)
    ax.fill_between(frac_fluent, score, alpha=0.08, color="#2B4162")

    # Mark endpoints
    ax.scatter([frac_fluent[0]], [score[0]], color="#8B3A00", s=80, zorder=5,
               label=f"Max evasion (score={score[0]:.2f})", edgecolors="white", linewidths=0.5)
    ax.scatter([frac_fluent[-1]], [score[-1]], color="#5B8C5A", s=80, zorder=5,
               label=f"Max fluency (score={score[-1]:.2f})", edgecolors="white", linewidths=0.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Fraction of Fluent Tokens")
    ax.set_ylabel("Probe Confidence")
    ax.set_xlim(0.25, 0.85)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    ax.set_title("GCG Evasion: Fluency vs. Probe Confidence Tradeoff")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"fig6_gcg_pareto.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
    print(f"\n  Saved fig6_gcg_pareto.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    main()
