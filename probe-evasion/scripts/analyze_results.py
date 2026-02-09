#!/usr/bin/env python3
"""
Convenience script to analyze experimental results.

Usage:
    python scripts/analyze_results.py results.json --output-dir analysis/
    python scripts/analyze_results.py results.json --compare baseline informed
    python scripts/analyze_results.py results.json --plot-only --output-dir figures/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import (
    load_results,
    results_to_dataframe,
    analyze_experiment,
    compare_regimes,
    print_summary,
    create_summary_dashboard,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze probe evasion experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with CSV outputs
  python scripts/analyze_results.py data/outputs/baseline_results.json -o analysis/

  # Compare two regimes
  python scripts/analyze_results.py data/outputs/all_results.json --compare baseline informed

  # Generate plots only
  python scripts/analyze_results.py data/outputs/all_results.json --plots -o figures/

  # Quick summary only
  python scripts/analyze_results.py data/outputs/baseline_results.json --summary-only
        """
    )

    parser.add_argument("results_path", help="Path to JSON results file")
    parser.add_argument("-o", "--output-dir", help="Directory to save outputs")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two regimes")
    parser.add_argument("--plots", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary and exit")
    parser.add_argument("--layers", nargs="+", type=int,
                        help="Specific layers to analyze")

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.results_path}")
    results = load_results(args.results_path)
    df = results_to_dataframe(results)
    print(f"Loaded {len(df)} samples across {df['regime'].nunique()} regime(s)")

    # Summary
    print_summary(df, title="Experiment Results")

    if args.summary_only:
        return

    # Comparison
    if args.compare:
        regime_a, regime_b = args.compare
        if regime_a in df["regime"].values and regime_b in df["regime"].values:
            comparison = compare_regimes(df, regime_a, regime_b)
            print(f"\n{'='*60}")
            print(f" Comparison: {regime_a} vs {regime_b}")
            print(f"{'='*60}\n")
            print(comparison.to_string(index=False))

            # Interpretation
            print("\nInterpretation:")
            for _, row in comparison.iterrows():
                layer = row["layer"]
                diff = row["difference"]
                d = row["cohens_d"]
                p = row.get("p_value")

                direction = "higher" if diff > 0 else "lower"
                effect = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"

                msg = f"  Layer {layer}: {regime_a} {direction} by {abs(diff):.3f} ({effect} effect, d={d:.2f})"
                if p is not None:
                    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    msg += f", p={p:.4f}{sig}"
                print(msg)
        else:
            print(f"Warning: Regimes not found. Available: {df['regime'].unique().tolist()}")

    # Full analysis with CSVs
    if args.output_dir and not args.plots:
        analyze_experiment(args.results_path, args.output_dir)

    # Plots
    if args.plots and args.output_dir:
        create_summary_dashboard(df, args.output_dir, layers=args.layers)
    elif args.plots:
        print("Warning: --plots requires --output-dir")


if __name__ == "__main__":
    main()
