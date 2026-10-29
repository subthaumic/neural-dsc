"""Plot the final one-hot DSCs of the handcrafted quadrant classifiers.

Four panels per architecture, one per class, drawn in the shared DSC language
from `neural_dsc.viz.complex` so these read the same way as every other complex in the
paper. What this script adds over a bare call is the four-up arrangement, a
single key across the panels, and the pure-versus-random comparison that
Appendix A leans on.

Both orderings of a predictor pair are drawn as their own patch: `(0, 3, 6)` and
`(3, 0, 6)` are different simplices resting on different arcs of the reciprocal
0--3 edge, and collapsing them hid that the complex contains both.

Usage:
    python scripts/plot_handcrafted_dsc_logic.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.io.persistence import load_probe_data
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.viz.activations import fill_kwargs, panel_values, shared_node_norm
from neural_dsc.viz.complex import add_dsc_key, draw_dsc, neuron_layout, panel_figsize

MODEL_FACTORIES = {"mono": create_quadrants_mono, "poly": create_quadrants_poly}

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figs" / "handcrafted_logic"


def load_complex(path):
    complex_ = DirectedSimplicialComplex()
    complex_.load(str(path))
    return complex_


def plot_architecture(name, layer_sizes, architecture_title):
    result_dir = RESULTS_DIR / f"quadrants_{name}_handcrafted"
    complexes = [
        load_complex(result_dir / f"dsc_class_{class_index}.txt")
        for class_index in range(4)
    ]
    layout = neuron_layout(layer_sizes)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colormaps["viridis"]
    activations = panel_values(
        MODEL_FACTORIES[name](),
        (load_probe_data(str(result_dir / "probe_data_onehot.pt")) or [])[:len(complexes)])
    node_norm = shared_node_norm(activations)

    fig, axes = plt.subplots(
        2, 2, figsize=panel_figsize(layout, 2, 2, panel_height=3.7),
        constrained_layout=True)
    # The gutter has to read as clearly wider than any gap inside a panel, or
    # the four nets merge into one field of circles and it stops being obvious
    # which nodes belong to which result.
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.14,
                                hspace=0.10)
    for class_index, (axis, complex_) in enumerate(zip(axes.flat, complexes)):
        draw_dsc(axis, complex_, layer_sizes, norm, cmap,
                 title=f"Q{class_index + 1}", layout=layout,
                 node_norm=node_norm,
                 **fill_kwargs(activations, class_index))

    # Before add_key: that freezes the layout engine, and a suptitle added
    # afterwards is placed at the default height, on top of the first row.
    fig.suptitle(
        f"{architecture_title}: pure-class directed simplicial complexes",
        fontsize=13,
    )
    add_dsc_key(fig, axes, norm, cmap, shrink=0.34)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURES_DIR / f"{name}_onehot_dsc_logic"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_poly_random_comparison():
    """Contrast the pure Q1 complex with the one built on a random mixture."""
    result_dir = RESULTS_DIR / "quadrants_poly_handcrafted"
    complexes = [
        load_complex(result_dir / "dsc_class_0.txt"),
        load_complex(result_dir / "dsc_random.txt"),
    ]
    layer_sizes = [2, 3, 4, 4]
    layout = neuron_layout(layer_sizes)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colormaps["viridis"]

    fig, axes = plt.subplots(
        1, 2, figsize=panel_figsize(layout, 2, 1, panel_height=4.6),
        constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.07)
    titles = ["Pure Q1", "Uniform-random context"]
    for axis, complex_, title in zip(axes, complexes, titles):
        draw_dsc(axis, complex_, layer_sizes, norm, cmap, title=title,
                 layout=layout)

    fig.suptitle(
        "Polysemantic classifier: pure class versus random mixture", fontsize=13,
    )
    add_dsc_key(fig, axes, norm, cmap, shrink=0.30, gap=0.020)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURES_DIR / "poly_q1_vs_random_dsc"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_architecture("mono", [2, 4, 4, 4], "Monosemantic classifier")
    plot_architecture("poly", [2, 3, 4, 4], "Polysemantic classifier")
    plot_poly_random_comparison()


if __name__ == "__main__":
    main()
