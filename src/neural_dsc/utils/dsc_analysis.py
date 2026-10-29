"""
DSC Analysis Utilities

Functions for analyzing Directed Simplicial Complexes and extracting
statistics for manuscript writing and interpretation.
"""

import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def compute_simplex_statistics(dsc) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a DSC.

    Args:
        dsc: DirectedSimplicialComplex instance

    Returns:
        Dictionary containing:
        - 'counts_by_dimension': {dim: count} for each dimension
        - 'total_simplices': total number of simplices
        - 'max_dimension': highest dimension with non-zero simplices
        - 'filtration_stats_by_dimension': {dim: {'mean', 'median', 'min', 'max', 'q25', 'q75'}}
        - 'total_vertices': number of unique neurons in the complex

    Example:
        >>> stats = compute_simplex_statistics(dsc)
        >>> print(f"Total simplices: {stats['total_simplices']}")
        >>> print(f"By dimension: {stats['counts_by_dimension']}")
    """
    counts = {}
    filtration_by_dim = defaultdict(list)
    all_vertices = set()

    # Collect statistics from the complex
    for dim, simplex_list in dsc.complex.items():
        counts[dim] = len(simplex_list)
        for simplex, filtration in simplex_list:
            filtration_by_dim[dim].append(filtration)
            all_vertices.update(simplex)

    # Compute filtration statistics per dimension
    filtration_stats = {}
    for dim, filt_values in filtration_by_dim.items():
        if filt_values:
            filt_array = np.array(filt_values)
            filtration_stats[dim] = {
                'mean': float(np.mean(filt_array)),
                'median': float(np.median(filt_array)),
                'min': float(np.min(filt_array)),
                'max': float(np.max(filt_array)),
                'q25': float(np.percentile(filt_array, 25)),
                'q75': float(np.percentile(filt_array, 75)),
                'count': len(filt_values)
            }

    # Find max dimension with simplices
    max_dim = max(counts.keys()) if counts else 0

    return {
        'counts_by_dimension': counts,
        'total_simplices': sum(counts.values()),
        'max_dimension': max_dim,
        'filtration_stats_by_dimension': filtration_stats,
        'total_vertices': len(all_vertices)
    }


def analyze_neuron_participation(dsc, layer_sizes: List[int]) -> Dict[str, Any]:
    """
    Analyze which neurons participate in simplices and how frequently.

    Args:
        dsc: DirectedSimplicialComplex instance
        layer_sizes: List of layer sizes [input_size, hidden1_size, ..., output_size]

    Returns:
        Dictionary containing:
        - 'neuron_counts': {neuron_id: count} - number of simplices each neuron appears in
        - 'neuron_ranking': List of (neuron_id, count) sorted by count (descending)
        - 'layer_participation': {layer_idx: count} - simplices per layer
        - 'top_neurons': Top 10 most participating neurons

    Example:
        >>> layer_sizes = [2, 4, 4, 4]  # Quadrant classifier
        >>> participation = analyze_neuron_participation(dsc, layer_sizes)
        >>> print(f"Top neuron: {participation['top_neurons'][0]}")
    """
    neuron_counts = defaultdict(int)

    # Count appearances in all simplices
    for dim, simplex_list in dsc.complex.items():
        for simplex, filtration in simplex_list:
            for neuron in simplex:
                neuron_counts[neuron] += 1

    # Rank neurons by participation
    neuron_ranking = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)

    # Compute layer-wise participation
    layer_participation = defaultdict(int)
    layer_boundaries = []
    cumsum = 0
    for size in layer_sizes:
        layer_boundaries.append(cumsum)
        cumsum += size
    layer_boundaries.append(cumsum)

    for neuron, count in neuron_counts.items():
        # Find which layer this neuron belongs to
        for layer_idx in range(len(layer_sizes)):
            if layer_boundaries[layer_idx] <= neuron < layer_boundaries[layer_idx + 1]:
                layer_participation[layer_idx] += count
                break

    return {
        'neuron_counts': dict(neuron_counts),
        'neuron_ranking': neuron_ranking,
        'layer_participation': dict(layer_participation),
        'top_neurons': neuron_ranking[:10] if neuron_ranking else []
    }


def analyze_directedness(dsc) -> Dict[str, Any]:
    """
    Analyze directedness patterns in the DSC.

    For each vertex set, determines how many directed orderings exist
    versus how many are theoretically possible.

    Args:
        dsc: DirectedSimplicialComplex instance

    Returns:
        Dictionary containing:
        - 'vertex_set_stats': {vertex_set: {'found': n, 'possible': n, 'fraction': f}}
        - 'asymmetry_examples': List of examples where A→B exists but not B→A
        - 'fully_directed': Count of vertex sets with all orderings
        - 'partially_directed': Count of vertex sets with some orderings

    Example:
        >>> directedness = analyze_directedness(dsc)
        >>> print(f"Asymmetry examples: {directedness['asymmetry_examples'][:5]}")
    """
    # Group simplices by their unordered vertex set
    vertex_set_to_simplices = defaultdict(list)

    for dim, simplex_list in dsc.complex.items():
        if dim == 0:
            continue  # Skip 0-simplices (vertices)
        for simplex, filtration in simplex_list:
            vertex_set = tuple(sorted(simplex))
            vertex_set_to_simplices[vertex_set].append((simplex, filtration))

    # Analyze each vertex set
    vertex_set_stats = {}
    asymmetry_examples = []
    fully_directed = 0
    partially_directed = 0

    for vertex_set, directed_simplices in vertex_set_to_simplices.items():
        n = len(vertex_set)
        possible = math.factorial(n)
        found = len(directed_simplices)
        fraction = found / possible if possible > 0 else 0

        vertex_set_stats[vertex_set] = {
            'found': found,
            'possible': possible,
            'fraction': fraction,
            'simplices': directed_simplices
        }

        if found == possible:
            fully_directed += 1
        elif found > 0:
            partially_directed += 1

        # Check for asymmetry (for 2-simplices / edges)
        if n == 2:
            a, b = vertex_set
            has_ab = any(s[0] == a and s[1] == b for s, _ in directed_simplices)
            has_ba = any(s[0] == b and s[1] == a for s, _ in directed_simplices)

            if has_ab and not has_ba:
                asymmetry_examples.append({
                    'vertices': (a, b),
                    'direction': f"{a}→{b}",
                    'missing': f"{b}→{a}"
                })
            elif has_ba and not has_ab:
                asymmetry_examples.append({
                    'vertices': (a, b),
                    'direction': f"{b}→{a}",
                    'missing': f"{a}→{b}"
                })

    return {
        'vertex_set_stats': vertex_set_stats,
        'asymmetry_examples': asymmetry_examples,
        'fully_directed_count': fully_directed,
        'partially_directed_count': partially_directed,
        'total_vertex_sets': len(vertex_set_stats)
    }


def get_filtration_distribution(dsc) -> Dict[str, Any]:
    """
    Analyze the distribution of filtration values in the DSC.

    Args:
        dsc: DirectedSimplicialComplex instance

    Returns:
        Dictionary containing:
        - 'all_filtrations': List of all filtration values
        - 'by_dimension': {dim: [filtrations]}
        - 'quartiles': {'q25', 'median', 'q75'} across all dimensions
        - 'histogram': Histogram bins and counts

    Example:
        >>> dist = get_filtration_distribution(dsc)
        >>> print(f"Median filtration: {dist['quartiles']['median']}")
    """
    all_filtrations = []
    by_dimension = defaultdict(list)

    for dim, simplex_list in dsc.complex.items():
        for simplex, filtration in simplex_list:
            all_filtrations.append(filtration)
            by_dimension[dim].append(filtration)

    result = {
        'by_dimension': dict(by_dimension)
    }

    if all_filtrations:
        filt_array = np.array(all_filtrations)
        result['all_filtrations'] = all_filtrations
        result['quartiles'] = {
            'q25': float(np.percentile(filt_array, 25)),
            'median': float(np.median(filt_array)),
            'q75': float(np.percentile(filt_array, 75)),
            'min': float(np.min(filt_array)),
            'max': float(np.max(filt_array))
        }

        # Compute histogram
        counts, bins = np.histogram(filt_array, bins=20)
        result['histogram'] = {
            'counts': counts.tolist(),
            'bins': bins.tolist()
        }
    else:
        result['all_filtrations'] = []
        result['quartiles'] = {'q25': 0, 'median': 0, 'q75': 0, 'min': 0, 'max': 0}
        result['histogram'] = {'counts': [], 'bins': []}

    return result


def format_simplex_summary(dsc, name: str = "DSC", model=None, top_k: int = 10) -> str:
    """
    Generate a human-readable summary of the DSC.

    Args:
        dsc: DirectedSimplicialComplex instance
        name: Name to use in the summary
        model: Optional MLP model for neuron labeling
        top_k: Number of top simplices to show per dimension

    Returns:
        Formatted string summary

    Example:
        >>> summary = format_simplex_summary(dsc, "Class 0 DSC")
        >>> print(summary)
    """
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"SUMMARY: {name}")
    lines.append(f"{'=' * 80}")

    stats = compute_simplex_statistics(dsc)

    # Overall statistics
    lines.append(f"\nTotal simplices: {stats['total_simplices']}")
    lines.append(f"Unique neurons: {stats['total_vertices']}")
    lines.append(f"Max dimension: {stats['max_dimension']}")

    # By dimension
    lines.append(f"\nSimplices by dimension:")
    for dim in sorted(stats['counts_by_dimension'].keys()):
        count = stats['counts_by_dimension'][dim]
        lines.append(f"  Dimension {dim}: {count} simplices")

        if dim in stats['filtration_stats_by_dimension']:
            filt_stats = stats['filtration_stats_by_dimension'][dim]
            lines.append(f"    Filtration - min: {filt_stats['min']:.4f}, "
                        f"median: {filt_stats['median']:.4f}, "
                        f"max: {filt_stats['max']:.4f}")

    # Top simplices per dimension
    lines.append(f"\nTop {top_k} simplices by dimension (lowest filtration):")
    for dim in sorted(dsc.complex.keys()):
        if dim == 0:
            continue  # Skip vertices
        simplex_list = dsc.complex[dim]
        if not simplex_list:
            continue

        # Sort by filtration value
        sorted_simplices = sorted(simplex_list, key=lambda x: x[1])[:top_k]

        lines.append(f"\n  Dimension {dim}:")
        for simplex, filtration in sorted_simplices:
            simplex_str = f"({', '.join(map(str, simplex))})"
            lines.append(f"    {simplex_str}: filtration = {filtration:.4f}")

    lines.append(f"\n{'=' * 80}\n")

    return '\n'.join(lines)


def print_dsc_summary(dsc, name: str = "DSC", model=None):
    """
    Print a brief console summary of the DSC.

    Args:
        dsc: DirectedSimplicialComplex instance
        name: Name to display
        model: Optional MLP model for neuron labeling

    Example:
        >>> print_dsc_summary(dsc, "Class 0 One-Hot DSC")
    """
    stats = compute_simplex_statistics(dsc)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Total simplices: {stats['total_simplices']}")
    print(f"By dimension: {dict(stats['counts_by_dimension'])}")
    print(f"Max dimension: {stats['max_dimension']}")
    print(f"Unique neurons: {stats['total_vertices']}")

    # Show filtration summary
    if stats['filtration_stats_by_dimension']:
        print(f"\nFiltration ranges by dimension:")
        for dim in sorted(stats['filtration_stats_by_dimension'].keys()):
            fstats = stats['filtration_stats_by_dimension'][dim]
            print(f"  Dim {dim}: [{fstats['min']:.4f}, {fstats['max']:.4f}], "
                  f"median={fstats['median']:.4f}")
    print(f"{'='*60}\n")


def save_dsc_statistics_csv(stats_list: List[Dict[str, Any]], filepath: str,
                             labels: Optional[List[str]] = None):
    """
    Save DSC statistics to CSV file.

    Args:
        stats_list: List of statistics dictionaries from compute_simplex_statistics()
        filepath: Path to save CSV file
        labels: Optional labels for each DSC (e.g., ["Class 0", "Class 1", ...])

    CSV Output Format:
        Each row represents one DSC. Columns are:
        - label: Name/identifier for the DSC
        - total_simplices: Total count of all simplices
        - max_dimension: Highest dimension with non-zero simplices
        - total_vertices: Number of unique neurons in the complex

        For each dimension d present in any DSC:
        - dim{d}_count: Number of d-simplices
        - dim{d}_filt_min: Minimum filtration value
        - dim{d}_filt_median: Median filtration value
        - dim{d}_filt_max: Maximum filtration value
        - dim{d}_filt_mean: Mean filtration value

    Example:
        >>> stats = [compute_simplex_statistics(dsc) for dsc in dscs]
        >>> save_dsc_statistics_csv(stats, "output/dsc_stats.csv",
        ...                         labels=["Class 0", "Class 1"])
    """
    if not stats_list:
        return

    # Determine all dimensions present
    all_dims = set()
    for stats in stats_list:
        all_dims.update(stats['counts_by_dimension'].keys())
    all_dims = sorted(all_dims)

    # Write CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['label', 'total_simplices', 'max_dimension', 'total_vertices']
        for dim in all_dims:
            header.extend([
                f'dim{dim}_count',
                f'dim{dim}_filt_min',
                f'dim{dim}_filt_median',
                f'dim{dim}_filt_max',
                f'dim{dim}_filt_mean'
            ])
        writer.writerow(header)

        # Data rows
        for idx, stats in enumerate(stats_list):
            label = labels[idx] if labels and idx < len(labels) else f"DSC_{idx}"

            row = [
                label,
                stats['total_simplices'],
                stats['max_dimension'],
                stats['total_vertices']
            ]

            for dim in all_dims:
                if dim in stats['counts_by_dimension']:
                    row.append(stats['counts_by_dimension'][dim])

                    if dim in stats['filtration_stats_by_dimension']:
                        fstats = stats['filtration_stats_by_dimension'][dim]
                        row.extend([
                            fstats['min'],
                            fstats['median'],
                            fstats['max'],
                            fstats['mean']
                        ])
                    else:
                        row.extend(['', '', '', ''])
                else:
                    row.extend([0, '', '', '', ''])

            writer.writerow(row)

    print(f"  Saved DSC statistics to {filepath}")


def create_analysis_summary(dsc_dict: Dict[str, Any], output_dir: str,
                            model=None, layer_sizes: Optional[List[int]] = None):
    """
    Create comprehensive analysis summary for a set of DSCs.

    Args:
        dsc_dict: Dictionary mapping names to DSCs, e.g., {"Class 0": dsc0, ...}
        output_dir: Directory to save outputs
        model: Optional MLP model
        layer_sizes: Optional list of layer sizes for participation analysis

    Creates:
        - {output_dir}/dsc_statistics.csv
        - {output_dir}/dsc_analysis_summary.txt

    Example:
        >>> dsc_dict = {"Class 0": dsc0, "Class 1": dsc1}
        >>> create_analysis_summary(dsc_dict, "output/", model, [2, 4, 4, 4])
    """
    from pathlib import Path
    output_path = Path(output_dir)

    # Compute statistics for all DSCs
    stats_list = []
    labels = []
    summaries = []

    for name, dsc in dsc_dict.items():
        labels.append(name)
        stats = compute_simplex_statistics(dsc)
        stats_list.append(stats)

        # Generate detailed summary
        summary = format_simplex_summary(dsc, name, model, top_k=5)
        summaries.append(summary)

        # Print brief summary to console
        print_dsc_summary(dsc, name, model)

    # Save CSV
    csv_path = output_path / "dsc_statistics.csv"
    save_dsc_statistics_csv(stats_list, str(csv_path), labels)

    # Save text summary
    txt_path = output_path / "dsc_analysis_summary.txt"
    with open(txt_path, 'w') as f:
        f.write('\n'.join(summaries))
    print(f"  Saved detailed analysis to {txt_path}")

    # If layer sizes provided, analyze neuron participation
    if layer_sizes:
        print("\nNeuron participation analysis:")
        for name, dsc in dsc_dict.items():
            participation = analyze_neuron_participation(dsc, layer_sizes)
            print(f"\n{name}:")
            print(f"  Top 5 neurons: {participation['top_neurons'][:5]}")
            print(f"  Layer participation: {participation['layer_participation']}")
