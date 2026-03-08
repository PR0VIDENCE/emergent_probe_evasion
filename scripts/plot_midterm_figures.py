"""
Generate midterm report figures from evasion experiment results.

Figs 1-2: Mean probe confidence (answer_mean_pool, top-5 layers) with
           bootstrap 95% CIs across trials. Positive questions only.
Fig 3:     TPR (majority_vote) per layer, positive questions only.

Produces 3 PDF figures:
  fig1_evasion.pdf   — Evasion prompts do not reduce probe detection
  fig2_controls.pdf  — Probes remain accurate across diverse prompting styles
  fig3_verbatim.pdf  — Verbatim copying with evasion intent drops probe signal

Usage:
    python scripts/plot_midterm_figures.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "figures"

POSITION = "answer_mean_pool"
TOP_K = 5
N_BOOTSTRAP = 10000

# Trial-level data files per concept
TRIAL_DIR = {
    "trees": PROJECT_ROOT / "data" / "outputs" / "tree_quick_experiments",
    "library": PROJECT_ROOT / "data" / "outputs" / "library_quick_experiments",
    "disease": PROJECT_ROOT / "data" / "outputs" / "disease_quick_experiments",
}

REGIME_FILES = {
    "trees": {
        "baseline": "baseline (6).json",
        "technical_disclosure": "technical_disclosure (5).json",
        "control_eli5": "control_eli5.json",
        "control_brevity": "control_brevity (1).json",
        "control_scientific_padding": "control_scientific_padding.json",
        "control_concept_substitution": "control_concept_substitution (1).json",
        "verbatim_evasion": "verbatim_evasion.json",
    },
    "library": {
        "baseline": "baseline (7).json",
        "technical_disclosure": "technical_disclosure (6).json",
        "control_eli5": "control_eli5 (1).json",
        "control_brevity": "control_brevity (2).json",
        "control_scientific_padding": "control_scientific_padding (1).json",
        "control_concept_substitution": "control_concept_substitution (2).json",
        "verbatim_evasion": "verbatim_evasion (1).json",
    },
    "disease": {
        "baseline": "baseline.json",
        "technical_disclosure": "technical_disclosure.json",
        "control_eli5": "control_eli5.json",
        "control_brevity": "control_brevity.json",
        "control_scientific_padding": "control_scientific_padding.json",
        "control_concept_substitution": "control_concept_substitution.json",
        "verbatim_evasion": "verbatim_evasion.json",
    },
}

COLORS = {
    "baseline": "#2196F3",
    "technical_disclosure": "#F44336",
    "control_eli5": "#4CAF50",
    "control_brevity": "#FF9800",
    "control_scientific_padding": "#9C27B0",
    "control_concept_substitution": "#795548",
    "verbatim_evasion": "#E91E63",
}

REGIME_LABELS = {
    "baseline": "Baseline",
    "technical_disclosure": "Evasion Prompt",
    "control_eli5": "ELI5",
    "control_brevity": "Brevity",
    "control_scientific_padding": "Scientific",
    "control_concept_substitution": "Wrong Concept",
    "verbatim_evasion": "Verbatim + Evasion",
}

CONCEPT_LABELS = {
    "trees": "Trees",
    "library": "Library",
    "disease": "Disease",
}


def load_trials(concept, regime):
    """Load trial-level JSON for a concept/regime."""
    trial_dir = TRIAL_DIR[concept]
    filename = REGIME_FILES[concept].get(regime)
    if not filename:
        return []
    path = trial_dir / filename
    if not path.exists():
        print(f"  WARNING: Missing {path}")
        return []
    with open(path, "r") as f:
        return json.load(f)


def get_positive_trials(trials):
    """Filter to positive-concept questions only (exclude nc* negative controls)."""
    return [t for t in trials if not t.get("question_id", "").startswith("nc")]


def get_probe_results(trial):
    """Get probe_results from a trial, handling multi-turn."""
    if "turns" in trial:
        return trial["turns"][-1].get("probe_results", {})
    return trial.get("probe_results", {})


def get_top_k_layers_by_confidence(trials, position=POSITION, k=TOP_K):
    """Select top-K layers by mean confidence on baseline positive trials."""
    layer_confs = {}
    for trial in get_positive_trials(trials):
        pos_data = get_probe_results(trial).get(position, {})
        for layer_key, layer_data in pos_data.items():
            layer_idx = int(layer_key) if isinstance(layer_key, str) and layer_key.isdigit() else layer_key
            if layer_idx not in layer_confs:
                layer_confs[layer_idx] = []
            layer_confs[layer_idx].append(layer_data["mean_confidence"])

    layer_means = {l: np.mean(c) for l, c in layer_confs.items()}
    sorted_layers = sorted(layer_means.keys(), key=lambda l: layer_means[l], reverse=True)
    return sorted_layers[:k]


def compute_per_trial_confidence(trials, top_layers, position=POSITION):
    """
    For each positive trial, compute mean confidence averaged across top-K layers.

    Returns array of per-trial confidence values.
    """
    values = []
    for trial in get_positive_trials(trials):
        pos_data = get_probe_results(trial).get(position, {})
        layer_confs = []
        for layer in top_layers:
            key = str(layer) if str(layer) in pos_data else layer
            if key in pos_data:
                layer_confs.append(pos_data[key]["mean_confidence"])
        if layer_confs:
            values.append(np.mean(layer_confs))
    return np.array(values)


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


def compute_tp_rate_by_layer(trials, position=POSITION):
    """Compute TP rate (majority_vote) per layer on positive questions only."""
    layer_votes = {}
    for trial in get_positive_trials(trials):
        pos_data = get_probe_results(trial).get(position, {})
        for layer_key, layer_data in pos_data.items():
            layer_idx = int(layer_key) if isinstance(layer_key, str) and layer_key.isdigit() else layer_key
            if layer_idx not in layer_votes:
                layer_votes[layer_idx] = []
            layer_votes[layer_idx].append(layer_data["majority_vote"])

    tp_rates = {}
    for layer_idx in sorted(layer_votes.keys()):
        votes = layer_votes[layer_idx]
        tp_rates[layer_idx] = np.mean(votes) if votes else 0.0
    return tp_rates


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig1_evasion(all_trials, top_layers_by_concept):
    """Bar chart: Baseline vs Technical Disclosure across concepts."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    concepts = ["trees", "library", "disease"]
    regimes = ["baseline", "technical_disclosure"]
    x = np.arange(len(concepts))
    width = 0.3

    for i, regime in enumerate(regimes):
        means = []
        err_lo = []
        err_hi = []
        for concept in concepts:
            trials = all_trials[concept][regime]
            top_layers = top_layers_by_concept[concept]
            values = compute_per_trial_confidence(trials, top_layers)
            m, lo, hi = bootstrap_ci(values)
            means.append(m)
            err_lo.append(lo)
            err_hi.append(hi)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=[err_lo, err_hi], capsize=3,
               label=REGIME_LABELS[regime], color=COLORS[regime], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Mean Probe Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels([CONCEPT_LABELS[c] for c in concepts])
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(frameon=False)
    ax.set_title("Evasion Prompts Do Not Reduce Probe Detection")

    fig.tight_layout()
    path = OUTPUT_DIR / "fig1_evasion.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


def fig2_controls(all_trials, top_layers_by_concept):
    """Bar chart: Confidence across control prompting styles."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    concepts = ["trees", "library", "disease"]
    regimes = ["baseline", "technical_disclosure", "control_eli5", "control_brevity",
               "control_scientific_padding", "control_concept_substitution"]
    x = np.arange(len(concepts))
    n = len(regimes)
    width = 0.15

    for i, regime in enumerate(regimes):
        means = []
        err_lo = []
        err_hi = []
        for concept in concepts:
            trials = all_trials[concept][regime]
            top_layers = top_layers_by_concept[concept]
            values = compute_per_trial_confidence(trials, top_layers)
            m, lo, hi = bootstrap_ci(values)
            means.append(m)
            err_lo.append(lo)
            err_hi.append(hi)
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width,
               yerr=[err_lo, err_hi], capsize=2,
               label=REGIME_LABELS[regime], color=COLORS[regime], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Mean Probe Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels([CONCEPT_LABELS[c] for c in concepts])
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Probes Remain Accurate Across Diverse Prompting Styles")

    fig.tight_layout()
    path = OUTPUT_DIR / "fig2_controls.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


def fig3_verbatim(all_trials):
    """Line plot: per-layer TPR for baseline vs verbatim_evasion."""
    concepts = ["trees", "library", "disease"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for idx, concept in enumerate(concepts):
        ax = axes[idx]

        for regime, style in [("baseline", "-o"), ("verbatim_evasion", "--s")]:
            trials = all_trials[concept][regime]
            tp_rates = compute_tp_rate_by_layer(trials) if trials else {}
            layers = sorted(tp_rates.keys())
            rates = [tp_rates[l] for l in layers]
            ax.plot(layers, rates, style, color=COLORS[regime],
                    markersize=3, linewidth=1.5, label=REGIME_LABELS[regime])

        ax.set_title(CONCEPT_LABELS[concept])
        ax.set_xlabel("Layer")
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.05, 1.05)

        if idx == 0:
            ax.set_ylabel("Probe Accuracy (TPR)")
            ax.legend(frameon=False, loc="lower left", fontsize=8)

    fig.suptitle("Verbatim Copying with Evasion Intent Drops Probe Signal",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig3_verbatim.pdf"
    fig.savefig(path)
    print(f"  Saved {path}")
    plt.close(fig)


def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    concepts = ["trees", "library", "disease"]
    all_regimes = ["baseline", "technical_disclosure", "control_eli5", "control_brevity",
                   "control_scientific_padding", "control_concept_substitution", "verbatim_evasion"]

    # Load all trial data
    print("Loading trial data...")
    all_trials = {}
    for concept in concepts:
        all_trials[concept] = {}
        for regime in all_regimes:
            all_trials[concept][regime] = load_trials(concept, regime)

    # Select top-K layers per concept from baseline
    print("Selecting top-5 layers per concept...")
    top_layers_by_concept = {}
    for concept in concepts:
        top_layers = get_top_k_layers_by_confidence(all_trials[concept]["baseline"])
        top_layers_by_concept[concept] = top_layers
        values = compute_per_trial_confidence(all_trials[concept]["baseline"], top_layers)
        print(f"  {concept}: layers={top_layers}, "
              f"baseline confidence={np.mean(values):.3f} +/- {np.std(values):.3f}")

    print("\nGenerating figures...")
    fig1_evasion(all_trials, top_layers_by_concept)
    fig2_controls(all_trials, top_layers_by_concept)
    fig3_verbatim(all_trials)
    print("\nDone!")


if __name__ == "__main__":
    main()
