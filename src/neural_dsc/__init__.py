"""
neural_dsc: filtered directed simplicial complexes (DSCs) of neural networks.

Builds and analyses filtered directed simplicial complexes from the activations
of a neural network under input-space ablation, as introduced in "The Tangled
Web They Weave: Exploring Neural Representations with Directed Simplicial
Complexes" (NeurReps 2026).
"""

__version__ = "1.0.0"

# Core classes
from .mlp import MLP
from .directed_simplicial_complex import DirectedSimplicialComplex

# I/O utilities
from .io.persistence import (
    save_probe_data,
    load_probe_data,
    save_pvalues,
    load_pvalues,
    save_jaccard_data,
    load_jaccard_data,
    save_sweep_results,
    load_sweep_results,
    save_metadata,
    load_metadata,
    check_config_fingerprint,
)

# Utilities
from .utils.jaccard_index import (
    JACCARD_FILTRATION_GRID,
    JACCARD_FILTRATION_INTERVALS,
    JACCARD_FILTRATION_POINTS,
    JACCARD_METHOD_ID,
    compute_jaccard_distance,
    compute_jaccard_index,
    filtration_jaccard_distance,
)
from .utils.pearl_causality import pearl_causal_test
from .utils.select import select_topk_by_cosine_similarity, get_twohot_indices
from .utils.dsc_analysis import (
    compute_simplex_statistics,
    analyze_neuron_participation,
    analyze_directedness,
    get_filtration_distribution,
    format_simplex_summary,
    print_dsc_summary,
    save_dsc_statistics_csv,
    create_analysis_summary,
)

# Data generation
from .data.gen import generate_quadrant_data

# Model generation
from .models.gen import create_quadrants_mono, create_quadrants_poly

# Visualization
from .viz.complex import plot_simplicial_complex
from .viz.data import plot_onehot_points, plot_twohot_points
from .viz.network import plot_neural_network

__all__ = [
    # Version
    "__version__",
    # Core classes
    "MLP",
    "DirectedSimplicialComplex",
    # I/O
    "save_probe_data",
    "load_probe_data",
    "save_pvalues",
    "load_pvalues",
    "save_jaccard_data",
    "load_jaccard_data",
    "save_sweep_results",
    "load_sweep_results",
    "save_metadata",
    "check_config_fingerprint",
    "load_metadata",
    # Utilities
    "JACCARD_FILTRATION_GRID",
    "JACCARD_FILTRATION_INTERVALS",
    "JACCARD_FILTRATION_POINTS",
    "JACCARD_METHOD_ID",
    "compute_jaccard_distance",
    "compute_jaccard_index",
    "filtration_jaccard_distance",
    "pearl_causal_test",
    "select_topk_by_cosine_similarity",
    "get_twohot_indices",
    # DSC Analysis
    "compute_simplex_statistics",
    "analyze_neuron_participation",
    "analyze_directedness",
    "get_filtration_distribution",
    "format_simplex_summary",
    "print_dsc_summary",
    "save_dsc_statistics_csv",
    "create_analysis_summary",
    # Data generation
    "generate_quadrant_data",
    # Model generation
    "create_quadrants_mono",
    "create_quadrants_poly",
    # Visualization
    "plot_simplicial_complex",
    "plot_onehot_points",
    "plot_twohot_points",
    "plot_neural_network",
]
