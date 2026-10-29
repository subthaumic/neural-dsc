"""
Figures for the untrained-baseline quadrant runs.

Companion to `plot_quadrants.py`, kept separate because the untrained producers
have a different output layout: one subdirectory per seed under
`results/quadrants_{arch}_untrained/`, rather than a single directory per
context. They also build only the one-hot complexes, on the *trained* run's
probe sets, so the trained/untrained comparison is on identical points.

The figure code lives here. It used to sit in the producers, which this script
imported and called back into -- the split was only half done, and it meant a
compute script owned three plotting functions and wrote PDFs beside its data.
The producers are still imported, but only for what they know about the run:
where a seed's results are, how to rebuild its model, and how to aggregate the
summary rows.

USAGE:
  python scripts/plot_quadrants_untrained.py           # mono and poly
  python scripts/plot_quadrants_untrained.py mono

Seeds with no result directory are reported and skipped.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.io.persistence import load_probe_data
from neural_dsc.viz.data import get_onehot_markers
from neural_dsc.viz.activations import panel_values, shared_node_norm
from neural_dsc.viz.complex import plot_simplicial_complex
from neural_dsc.viz.network import plot_neural_network
from neural_dsc.utils.progress import RunTimer

REPO_ROOT = Path(__file__).parent.parent
FIGS_DIR = REPO_ROOT / "figs" / "quadrants"
ARCHITECTURES = ["mono", "poly"]
# Untrained networks are named by their layer widths, as in the article.
LAYER_WIDTHS = {"mono": "[2,4,4,4]", "poly": "[2,3,4,4]"}

# Shared with the producers by value rather than by import: these decide what a
# figure says, not what it is computed from, so they belong on this side of the
# split. CLASS_LABELS matches the repo-wide Q1..Q4 naming.
CLASS_LABELS = ("Q1", "Q2", "Q3", "Q4")
COLORS = ["r", "g", "b", "purple"]


def figure_dir_for(architecture, seed=None):
    directory = FIGS_DIR / f"{architecture}_untrained"
    if seed is not None:
        directory = directory / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def max_filtration(complexes):
    values = [
        float(filtration)
        for sc in complexes
        for simplices in sc.complex.values()
        for _, filtration in simplices
    ]
    return max(values) if values else 1.0


def plot_onehot_complexes(model, complexes, figure_dir, architecture, seed,
                          onehot_data=None):
    vmax = max_filtration(complexes)
    values = panel_values(model, onehot_data)
    node_norm = shared_node_norm(values)
    for class_idx, sc in enumerate(complexes):
        plot_simplicial_complex(
            sc, model, vmin=0, vmax=vmax,
            title=(f"Untrained {LAYER_WIDTHS[architecture]} seed {seed}: "
                   f"{CLASS_LABELS[class_idx]} one-hot"),
            values=values[class_idx] if values else None, node_norm=node_norm,
            save_path=str(figure_dir / f"dsc_class_{class_idx}.pdf"),
            show_plot=False,
        )
        plt.close()


def plot_selected_onehot_points(X, y, onehot_data, figure_dir, architecture, seed):
    markers = get_onehot_markers(len(onehot_data))
    plt.figure(figsize=(8, 8))
    for class_idx in range(len(onehot_data)):
        class_indices = y == class_idx
        plt.scatter(X[class_indices, 0].numpy(), X[class_indices, 1].numpy(),
                    c="lightgray", alpha=1, label=None)
    for class_idx, points in enumerate(onehot_data):
        plt.scatter(points[:, 0].numpy(), points[:, 1].numpy(),
                    c=COLORS[class_idx], marker=markers[class_idx], alpha=1,
                    label=f"{CLASS_LABELS[class_idx]} one-hot")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_dir
                / f"quadrants_{architecture}_untrained_seed{seed}_onehot.pdf")
    plt.close()


def plot_untrained_comparison(summary_rows, path):
    rows = [row for row in summary_rows if int(row["n_untrained_seeds"]) > 0]
    if not rows:
        print("No untrained summary rows available; skipping comparison figure")
        return

    labels = [f"{row['architecture']} {row['class_label']}" for row in rows]
    trained = [float(row["trained_nonvertex_simplices"]) for row in rows]
    untrained = [float(row["untrained_nonvertex_simplices_mean"]) for row in rows]
    errors = [float(row["untrained_nonvertex_simplices_std"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, trained, width, label="Trained")
    ax.bar(x + width / 2, untrained, width, yerr=errors, label="Untrained",
           capsize=3)
    ax.set_ylabel("Non-vertex simplices")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.set_title("Trained vs. untrained DSC size")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  comparison figure written to {path}")


def load_producer(architecture):
    """Import a compute_quadrants_*_untrained script as a module."""
    path = REPO_ROOT / "scripts" / f"compute_quadrants_{architecture}_untrained.py"
    name = f"producer_{architecture}_untrained"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plot_seed(producer, seed, X, y):
    """Draw one seed's figures from its result directory."""
    output_dir = producer.output_dir_for(seed)
    if not output_dir.exists():
        print(f"  seed {seed}: no results in {output_dir} -- skipping")
        return False

    complexes = [
        producer.load_complex(output_dir / f"dsc_class_{i}.txt") for i in range(4)
    ]
    onehot_data = load_probe_data(str(output_dir / "probe_data_onehot.pt"))
    if onehot_data is None:
        print(f"  seed {seed}: no probe data in {output_dir} -- skipping")
        return False

    model = producer.load_or_create_untrained_model(seed)
    architecture = producer.ARCHITECTURE
    figure_dir = figure_dir_for(architecture, seed)

    plot_neural_network(
        model,
        save_path=str(figure_dir
                      / f"quadrants_{architecture}_untrained_network.pdf"),
        show_plot=False,
        title=f"Untrained {LAYER_WIDTHS[architecture]} Quadrant Classifier (seed {seed})",
    )
    plt.close()

    plot_selected_onehot_points(X, y, onehot_data, figure_dir, architecture, seed)
    plot_onehot_complexes(model, complexes, figure_dir, architecture, seed,
                          onehot_data)
    print(f"  seed {seed}: figures written to {figure_dir}")
    return True


def main(run):
    requested = sys.argv[1:]
    architectures = [a for a in ARCHITECTURES if not requested or a in requested]
    unknown = set(requested) - set(ARCHITECTURES)
    if unknown:
        raise SystemExit(f"Unknown architecture(s): {sorted(unknown)}. "
                         f"Known: {ARCHITECTURES}")

    X, y = generate_quadrant_data()

    for architecture in architectures:
        run.section(f"{architecture} untrained")
        producer = load_producer(architecture)
        drawn = sum(plot_seed(producer, seed, X, y) for seed in producer.SEEDS)
        if not drawn:
            continue

        # The across-seed comparison needs the aggregate the producer wrote, so
        # it is recomputed from the same result directories rather than parsed
        # back out of summary.csv.
        summary_rows = producer.aggregate_summary_rows(producer.collect_summary_rows())
        plot_untrained_comparison(
            summary_rows,
            figure_dir_for(architecture) / "untrained_comparison.pdf",
        )


if __name__ == "__main__":
    with RunTimer("untrained quadrant figures") as run:
        main(run)
