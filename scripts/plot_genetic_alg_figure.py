"""
Generate figure showing genetic algorithm evasion progress over generations.

Plots the best (lowest) probe confidence per generation across 16 generations
of Claude-iterated reasoning traces.

Usage:
    python scripts/plot_genetic_alg_figure.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "figures"
DATA_FILE = PROJECT_ROOT / "data" / "outputs" / "genetic_alg_qwen" / "population.json"


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

    print("Loading population data...")
    with open(DATA_FILE, "r") as f:
        population = json.load(f)
    print(f"  Loaded {len(population)} individuals")

    # Compute best (lowest) score per generation
    min_scores = defaultdict(lambda: 1.0)
    for item in population:
        g = item["generation"]
        min_scores[g] = min(min_scores[g], item["score"])

    generations = sorted(min_scores.keys())
    # Cumulative minimum: best score seen up to and including each generation
    raw_scores = [min_scores[g] for g in generations]
    best_scores = []
    running_min = 1.0
    for s in raw_scores:
        running_min = min(running_min, s)
        best_scores.append(running_min)

    print("  Best score per generation:")
    for g, s in zip(generations, best_scores):
        print(f"    gen {g:2d}: {s:.6f}")

    fig, ax = plt.subplots(figsize=(7, 5.25))

    ax.plot(generations, best_scores, "-o", color="#8B3A00",
            markersize=6, linewidth=2, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Probe Confidence")
    ax.set_xticks(generations)
    ax.set_ylim(0.99, 1.001)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title("Genetic Algorithm Fails to Evade Probes\n(950 Candidates over 15 Generations)",
                 fontsize=14, fontweight="bold", linespacing=1.4)
    ax.annotate("Note: y-axis starts at 0.990",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="#666666",
                style="italic")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"fig5_genetic_alg.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
    print(f"\n  Saved fig5_genetic_alg.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    main()
