#!/usr/bin/env python3
"""
Extract DSC Statistics for Manuscript

This script demonstrates how to extract and format DSC statistics from
experiment results for use in manuscript writing.

Usage:
    # Extract stats from a specific experiment
    python scripts/report_manuscript_stats.py results/quadrants_mono_handcrafted/

    # Extract stats from all experiments in results/
    python scripts/report_manuscript_stats.py --all

    # Generate LaTeX table
    python scripts/report_manuscript_stats.py results/quadrants_mono_trained/ --latex

Examples of extracting specific statistics:

    1. Total simplices across all classes:
       >>> import pandas as pd
       >>> df = pd.read_csv("results/quadrants_mono_handcrafted/dsc_statistics_all.csv")
       >>> print(f"Total simplices: {df['total_simplices'].sum()}")

    2. Max dimension reached:
       >>> print(f"Max dimension: {df['max_dimension'].max()}")

    3. Compare one-hot vs two-hot complexity:
       >>> onehot = df[df['label'].str.contains('one-hot')]
       >>> twohot = df[df['label'].str.contains('\\+')]
       >>> print(f"One-hot mean: {onehot['total_simplices'].mean():.1f}")
       >>> print(f"Two-hot mean: {twohot['total_simplices'].mean():.1f}")

    4. Find most complex DSC:
       >>> most_complex = df.loc[df['total_simplices'].idxmax()]
       >>> print(f"Most complex: {most_complex['label']} ({most_complex['total_simplices']} simplices)")
"""

import argparse
import csv
import sys
from neural_dsc.utils.progress import RunTimer
from pathlib import Path

# Add src to path for package imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_statistics(csv_path: Path) -> list:
    """Load statistics from a CSV file."""
    if not csv_path.exists():
        return []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_summary(stats: list, title: str = "DSC Statistics"):
    """Print a formatted summary of statistics."""
    if not stats:
        print(f"No statistics found for {title}")
        return

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    # Calculate aggregate statistics
    total_simplices = sum(int(s.get('total_simplices', 0)) for s in stats)
    max_dim = max(int(s.get('max_dimension', 0)) for s in stats)
    total_vertices = sum(int(s.get('total_vertices', 0)) for s in stats)

    print(f"\nAggregate Statistics:")
    print(f"  Number of DSCs: {len(stats)}")
    print(f"  Total simplices (all DSCs): {total_simplices}")
    print(f"  Max dimension reached: {max_dim}")
    print(f"  Total unique neurons: {total_vertices}")

    # Per-DSC breakdown
    print(f"\nPer-DSC Breakdown:")
    print(f"  {'Label':<30} {'Simplices':>10} {'Max Dim':>10} {'Neurons':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    for s in stats:
        label = s.get('label', 'Unknown')[:30]
        simplices = s.get('total_simplices', 0)
        dim = s.get('max_dimension', 0)
        neurons = s.get('total_vertices', 0)
        print(f"  {label:<30} {simplices:>10} {dim:>10} {neurons:>10}")

    # Simple statistics
    simplices_list = [int(s.get('total_simplices', 0)) for s in stats]
    if simplices_list:
        print(f"\nSimplex Count Statistics:")
        print(f"  Mean: {sum(simplices_list)/len(simplices_list):.1f}")
        print(f"  Min: {min(simplices_list)}")
        print(f"  Max: {max(simplices_list)}")


def generate_latex_table(stats: list, title: str = "DSC Statistics") -> str:
    """Generate a LaTeX table from statistics."""
    if not stats:
        return "% No statistics available"

    lines = [
        f"% {title}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "DSC & Total Simplices & Max Dimension & Unique Neurons \\\\",
        "\\midrule",
    ]

    for s in stats:
        label = s.get('label', 'Unknown').replace('_', '\\_')
        simplices = s.get('total_simplices', 0)
        dim = s.get('max_dimension', 0)
        neurons = s.get('total_vertices', 0)
        lines.append(f"{label} & {simplices} & {dim} & {neurons} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def analyze_experiment(output_dir: Path, latex: bool = False):
    """Analyze statistics from an experiment output directory."""

    # Find available statistics files
    stats_files = list(output_dir.glob("dsc_statistics*.csv"))

    if not stats_files:
        print(f"No statistics files found in {output_dir}")
        return

    print(f"\nAnalyzing: {output_dir.name}")
    print(f"Found {len(stats_files)} statistics file(s)")

    # Load and display each statistics file
    for stats_file in sorted(stats_files):
        stats = load_statistics(stats_file)
        title = stats_file.stem.replace('dsc_statistics_', '').replace('dsc_statistics', 'all').title()

        if latex:
            print(generate_latex_table(stats, f"{output_dir.name} - {title}"))
        else:
            print_summary(stats, f"{output_dir.name} - {title}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DSC statistics for manuscript writing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to experiment output directory"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all experiments in results/"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX tables instead of text output"
    )
    args = parser.parse_args()

    results_dir = REPO_ROOT / "results"

    if args.all:
        # Analyze all experiments
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return

        experiment_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
        if not experiment_dirs:
            print("No experiment directories found")
            return

        for exp_dir in experiment_dirs:
            analyze_experiment(exp_dir, args.latex)

    elif args.path:
        # Analyze specific experiment
        if not args.path.exists():
            print(f"Directory not found: {args.path}")
            return

        analyze_experiment(args.path, args.latex)

    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Examples:")
        print("="*60)
        print("""
# Load CSV in Python:
import pandas as pd
df = pd.read_csv("results/quadrants_mono_handcrafted/dsc_statistics_all.csv")

# Get total simplices per DSC:
print(df[['label', 'total_simplices']])

# Compare classes:
print(df.groupby(df['label'].str.contains('one-hot'))['total_simplices'].mean())

# Find highest dimension simplices:
dim_cols = [c for c in df.columns if c.startswith('dim') and c.endswith('_count')]
for col in dim_cols:
    if df[col].sum() > 0:
        print(f"{col}: {df[col].sum()} simplices")
""")


if __name__ == "__main__":
    with RunTimer("manuscript stats extraction"):
        main()
