"""
Figures for the quadrant DSC experiments.

Reads what `compute_quadrants_*.py` wrote and draws every figure from it. The
construction is hours on a server and the figures are seconds on a laptop, so
they are separate scripts: a caption, a colormap or an axis label can be changed
without rebuilding a single complex.

Everything is reconstructed from disk -- the saved model weights, the saved probe
sets, the saved DSC text files and the cached Jaccard JSON. The script therefore
also serves as a check that those artifacts are sufficient to reproduce the
figures without a rerun.

One script rather than six because the figure code was identical across the
producers apart from the architecture name in two titles and a filename prefix.

USAGE:
  python scripts/plot_quadrants.py              # every context found on disk
  python scripts/plot_quadrants.py mono_handcrafted poly_trained

Contexts with no result directory are reported and skipped.
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

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.viz.activations import panel_values, shared_node_norm
from neural_dsc.viz.complex import plot_simplicial_complex
from neural_dsc.viz.network import plot_neural_network
from neural_dsc.utils.select import get_twohot_indices
from neural_dsc.utils.jaccard_index import compute_jaccard_distance
from neural_dsc.viz.data import (
    get_twohot_colors,
    get_onehot_markers,
    get_twohot_markers,
    plot_onehot_points,
    plot_twohot_points,
)
from neural_dsc.io.persistence import load_probe_data, load_jaccard_data, save_jaccard_data
from neural_dsc.utils.progress import RunTimer

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
FIGS_DIR = REPO_ROOT / "figs" / "quadrants"

# Marker styles, kept in one place so every context draws the same way.
RANDOM_MARKER = "x"
BASE_COLORS = ["r", "g", "b", "purple"]


@dataclass
class Context:
    """One producer's output directory and the labels its figures carry."""
    name: str
    factory: object
    arch_title: str

    @property
    def prefix(self):
        return f"quadrants_{self.name}"

    @property
    def output_dir(self):
        """Where the producer's data lives. Read from, and written to only for
        derived tables -- the Jaccard JSON and CSV are results, not pictures."""
        return RESULTS_DIR / self.prefix

    @property
    def figure_dir(self):
        """Where this context's figures go. Every other plot_ script writes
        under figs/, and a figure in results/ is one more thing to explain when
        someone asks which files a producer owns."""
        return FIGS_DIR / self.name

    @property
    def model_path(self):
        return MODELS_DIR / f"{self.prefix}.pth"

    @property
    def network_title(self):
        return f"{self.arch_title} Quadrant Classifier"

    @property
    def jaccard_subject(self):
        """The title's subject mid-sentence: "Monosemantic" stays capitalised
        as in the handcrafted figures, "Trained [2,4,4,4]" does not."""
        if self.name.endswith("handcrafted"):
            return self.arch_title
        return self.arch_title[0].lower() + self.arch_title[1:]

    @property
    def has_twohot(self):
        # The untrained runs deliberately build only the one-hot complexes, on
        # the trained run's probe sets, so the comparison is on identical points.
        return not self.name.endswith("untrained")


# Naming: "monosemantic" and "polysemantic" name only the handcrafted networks.
# The architectures, and the trained and untrained networks, are named by their
# layer widths, [2,4,4,4] and [2,3,4,4], as in the article.
CONTEXTS = [
    Context("mono_handcrafted", create_quadrants_mono, "Monosemantic"),
    Context("mono_trained", create_quadrants_mono, "Trained [2,4,4,4]"),
    Context("mono_untrained", create_quadrants_mono, "Untrained [2,4,4,4]"),
    Context("poly_handcrafted", create_quadrants_poly, "Polysemantic"),
    Context("poly_trained", create_quadrants_poly, "Trained [2,3,4,4]"),
    Context("poly_untrained", create_quadrants_poly, "Untrained [2,3,4,4]"),
]


def load_complex(path):
    """Load one saved DSC, or None if the producer never wrote it."""
    if not path.exists():
        return None
    sc = DirectedSimplicialComplex()
    sc.load(str(path))
    return sc


def max_filtration(complexes):
    """Largest filtration value across a list of complexes, for a shared colour scale."""
    best = 0
    for sc in complexes:
        if sc is None:
            continue
        for simplices in sc.complex.values():
            for _, filtration in simplices:
                best = max(best, filtration)
    return best



# The quadrant experiment's two-hot contexts are the four class pairs that share
# a decision boundary. The diagonally opposed pairs (Q1,Q3) and (Q2,Q4) are
# constructed differently -- they pool two pure regions instead of straddling a
# boundary -- so they are not part of this experiment and appear in no figure.
ADJACENT_TWOHOT = ((0, 1), (1, 2), (2, 3), (0, 3))


def adjacent_twohot_indices(n_classes):
    """Two-hot class pairs sharing a decision boundary, in label order."""
    if n_classes != 4:
        return get_twohot_indices(n_classes)
    return sorted(tuple(sorted(pair)) for pair in ADJACENT_TWOHOT)


def load_jaccard_matrix(path, shape):
    """Load a cached Jaccard matrix, discarding one whose shape has gone stale.

    The set of contexts a matrix covers can change between runs; a cached matrix
    from an earlier set would otherwise be indexed against the wrong labels.
    """
    matrix, by_dim = load_jaccard_data(str(path))
    if matrix is not None and tuple(matrix.shape) != tuple(shape):
        print(f"  cached {Path(path).name} covers a different context set -- recomputing")
        return None, None
    return matrix, by_dim


def load_context(ctx, n_classes=4):
    """
    Rebuild everything the figures need from `ctx`'s result directory.

    Returns None when the directory does not exist yet, so a partial set of
    producer runs still plots what it has.
    """
    if not ctx.output_dir.exists():
        return None

    net = ctx.factory()
    if ctx.model_path.exists():
        net.load_state_dict(torch.load(ctx.model_path))
    net.eval()

    twohot_indices = adjacent_twohot_indices(n_classes)
    filelabels = [f"{i}{j}" for i, j in twohot_indices]

    onehot_data = load_probe_data(str(ctx.output_dir / "probe_data_onehot.pt")) or []
    twohot_data = load_probe_data(str(ctx.output_dir / "probe_data_twohot.pt")) or []
    # The producers save the adjacent pairs only, in the order of
    # `get_twohot_indices` with the diagonal pairs left out.
    if twohot_data:
        saved_pairs = [p for p in get_twohot_indices(n_classes) if p in twohot_indices]
        twohot_data = [
            twohot_data[saved_pairs.index(pair)]
            for pair in twohot_indices
            if pair in saved_pairs
        ]
    random_data = load_probe_data(str(ctx.output_dir / "probe_data_random.pt")) or []

    onehot_complexes = [
        load_complex(ctx.output_dir / f"dsc_class_{i}.txt") for i in range(n_classes)
    ]
    two_hot_complexes = [
        load_complex(ctx.output_dir / f"dsc_twohot_{label}.txt") for label in filelabels
    ] if ctx.has_twohot else []
    sc_random = load_complex(ctx.output_dir / "dsc_random.txt")

    if not any(sc is not None for sc in onehot_complexes):
        return None

    return {
        "net": net,
        "onehot_data": onehot_data,
        "twohot_data": twohot_data,
        "random_points": random_data[0] if random_data else None,
        "onehot_complexes": onehot_complexes,
        "two_hot_complexes": [sc for sc in two_hot_complexes],
        "sc_random": sc_random,
        "twohot_indices": twohot_indices,
        "filelabels_twohot": filelabels,
        "labels_twohot": [f"Q{i + 1}+Q{j + 1}" for i, j in twohot_indices],
        "vmax": max_filtration(onehot_complexes),
        "two_hot_vmax": max_filtration(two_hot_complexes + [sc_random]),
    }


def plot_context(ctx, loaded, X, y, n_classes=4):
    """Draw every figure for one context."""
    output_dir = ctx.output_dir
    figure_dir = ctx.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    net = loaded["net"]
    onehot_data = loaded["onehot_data"]
    twohot_data = loaded["twohot_data"]
    random_points = loaded["random_points"]
    onehot_complexes = loaded["onehot_complexes"]
    two_hot_complexes = loaded["two_hot_complexes"]
    sc_random = loaded["sc_random"]
    twohot_indices = loaded["twohot_indices"]
    filelabels_twohot = loaded["filelabels_twohot"]
    labels_twohot = loaded["labels_twohot"]
    vmax = loaded["vmax"]
    two_hot_vmax = loaded["two_hot_vmax"]
    colors = list(BASE_COLORS)
    random_marker = RANDOM_MARKER

    # Node fills, on one scale across every context this figure set covers,
    # so a neuron shaded alike in the one-hot and two-hot panels really is
    # driven alike. Built from the same saved probes the complexes were.
    onehot_values = panel_values(net, onehot_data)
    twohot_values = panel_values(net, twohot_data)
    random_values = panel_values(
        net, [random_points] if random_points is not None else [])
    node_norm = shared_node_norm(
        (onehot_values or []) + (twohot_values or []) + (random_values or []))

    all_dims = set()
    for sc in onehot_complexes + two_hot_complexes:
        if sc is not None:
            all_dims.update(sc.complex.keys())
    if sc_random is not None:
        all_dims.update(sc_random.complex.keys())
    all_dims = sorted([d for d in all_dims if d != 0])

    # ---- probe sets over the input domain ----
    markers = get_onehot_markers(n_classes)
    if onehot_data:
        plot_onehot_points(X, y, onehot_data, n_classes, figure_dir, colors, markers)
    if twohot_data:
        # One colour and marker per *pair*, so these are not the one-hot lists.
        plot_twohot_points(
            X, y, twohot_data, n_classes, twohot_indices, figure_dir,
            get_twohot_colors(colors, twohot_indices), get_twohot_markers(n_classes),
        )

    # ---- network diagram ----
    # Plotting is cheap relative to producing the complexes. Always redraw so a
    # run cannot retain stale geometry or caption placement from an older
    # renderer alongside newly generated panels.
    network_pdf = figure_dir / f"{ctx.prefix}_network.pdf"
    plot_neural_network(
        net,
        save_path=str(network_pdf),
        show_plot=False,
        title=ctx.network_title,
    )

    # -------------------------------------------------------------------------

    # ---- one-hot activations and complexes ----
    # 2.3 plot network activations and DSCs for each class
    for class_idx in range(len(onehot_data)):
        activations_pdf = figure_dir / f"activations_class_{class_idx}.pdf"
        plot_neural_network(
            net,
            input=onehot_data[class_idx],
            title=f"Neuron Activations for Q{class_idx + 1} one-hot",
            save_path=str(activations_pdf),
            show_plot=False,
        )
        dsc_pdf = figure_dir / f"dsc_class_{class_idx}.pdf"
        plot_simplicial_complex(
            onehot_complexes[class_idx],
            net,
            vmin=0,
            vmax=vmax,
            title=f"Q{class_idx + 1} one-hot",
            values=onehot_values[class_idx] if onehot_values else None,
            node_norm=node_norm,
            save_path=str(dsc_pdf),
            show_plot=False,
        )

        # Figure 2 supplies its panel and stage headings in TikZ.  Keep a
        # dedicated pair of title- and frame-free assets so that the composed
        # figure has exactly its two outer rounded panels rather than a stack
        # of nested matplotlib frames.
        if class_idx == 0 and ctx.name in {
            "mono_handcrafted", "poly_handcrafted"
        }:
            figure2_activations_pdf = (
                figure_dir / f"{ctx.prefix}_figure2_activations.pdf"
            )
            plot_neural_network(
                net,
                input=onehot_data[class_idx],
                title=None,
                frame=False,
                save_path=str(figure2_activations_pdf),
                show_plot=False,
            )
            figure2_dsc_pdf = figure_dir / f"{ctx.prefix}_figure2_dsc.pdf"
            plot_simplicial_complex(
                onehot_complexes[class_idx],
                net,
                vmin=0,
                vmax=vmax,
                title=None,
                frame=False,
                values=onehot_values[class_idx] if onehot_values else None,
                node_norm=node_norm,
                save_path=str(figure2_dsc_pdf),
                show_plot=False,
            )

    if not ctx.has_twohot:
        return

    # ---- two-hot complexes ----
    # 3.3 Plot DSCs for two-hot encoded points
    for idx, sc in enumerate(two_hot_complexes):
        dsc_pdf = figure_dir / f"dsc_twohot_{filelabels_twohot[idx]}.pdf"
        plot_simplicial_complex(
            sc,
            net,
            vmin=0,
            vmax=max(vmax, two_hot_vmax),
            title=labels_twohot[idx],
            values=twohot_values[idx] if twohot_values else None,
            node_norm=node_norm,
            save_path=str(dsc_pdf),
            show_plot=False,
        )

    # ---- random control ----
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        class_indices = y == i
        plt.scatter(
            X[class_indices, 0].numpy(),
            X[class_indices, 1].numpy(),
            color="lightgray",
            alpha=1,
            label=None,
        )
    plt.scatter(
        random_points[:, 0].numpy(),
        random_points[:, 1].numpy(),
        color="black",
        marker=random_marker,
        alpha=1,
        label="Random",
    )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.tight_layout()
    random_indiv_pdf = figure_dir / f"{ctx.prefix}_random.pdf"
    plt.savefig(str(random_indiv_pdf))
    plt.close()


    # 4.3 Plot DSC for random points
    dsc_pdf_random = figure_dir / "dsc_random.pdf"
    plot_simplicial_complex(
        sc_random,
        net,
        vmin=0,
        vmax=max(vmax, two_hot_vmax),
        title="Uniform-random context",
        values=random_values[0] if random_values else None,
        node_norm=node_norm,
        save_path=str(dsc_pdf_random),
        show_plot=False,
    )

    # ---- Jaccard matrices ----
    # The producer computes these and caches them as JSON; this reads the cache.
    # The fallback recomputes from the loaded complexes rather than failing,
    # since everything it needs is already in hand -- so a result directory
    # missing its JSON still plots, and gets the cache written for next time.
    # 5.1 Jaccard distances: two-hot vs one-hot
    jaccard_twohot_file = output_dir / "jaccard_twohot_vs_onehot.json"
    jaccard_matrix, _ = load_jaccard_matrix(
        jaccard_twohot_file, (len(two_hot_complexes), len(onehot_complexes))
    )

    if jaccard_matrix is None:
        print("Computing Jaccard distances for two-hot vs one-hot...")
        n_onehot = len(onehot_complexes)
        n_twohot = len(two_hot_complexes)
        jaccard_matrix = np.zeros((n_twohot, n_onehot))
        for i in range(n_twohot):
            for j in range(n_onehot):
                jaccard_matrix[i, j] = compute_jaccard_distance(
                    two_hot_complexes[i], onehot_complexes[j]
                )
        save_jaccard_data(jaccard_matrix, {}, str(jaccard_twohot_file))
    else:
        print(f"Loaded existing Jaccard data from {jaccard_twohot_file}")
        n_onehot = len(onehot_complexes)
        n_twohot = len(two_hot_complexes)

    plt.figure(figsize=(8, 6))
    labels_twohot = [f"Q{i + 1}+Q{j + 1}" for i, j in twohot_indices]
    labels_onehot = [f"Q{k + 1}" for k in range(n_onehot)]
    im = plt.imshow(jaccard_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Jaccard distance")
    plt.title("Jaccard distance", fontsize=14)
    plt.xticks(np.arange(n_onehot), labels_onehot, fontsize=12)
    plt.yticks(np.arange(n_twohot), labels_twohot, fontsize=12)
    for i in range(n_twohot):
        for j in range(n_onehot):
            plt.text(
                j,
                i,
                f"{jaccard_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if jaccard_matrix[i, j] < 0.5 else "black",
                fontsize=12,
            )
    jaccard_pdf = figure_dir / "jaccard_matrix_twohot_vs_onehot.pdf"
    plt.savefig(str(jaccard_pdf))
    plt.close()

    # 5.2 Jaccard distances: random vs one-hot
    jaccard_random_file = output_dir / "jaccard_random_vs_onehot.json"
    jaccard_matrix_random_onehot, _ = load_jaccard_matrix(
        jaccard_random_file, (1, len(onehot_complexes))
    )

    if jaccard_matrix_random_onehot is None:
        print("Computing Jaccard distances for random vs one-hot/two-hot...")
        jaccard_matrix_random_onehot = np.zeros((1, n_onehot))
        for j in range(n_onehot):
            jaccard_matrix_random_onehot[0, j] = compute_jaccard_distance(
                sc_random, onehot_complexes[j]
            )
        save_jaccard_data(jaccard_matrix_random_onehot, {}, str(jaccard_random_file))
    else:
        print(f"Loaded existing Jaccard data from {jaccard_random_file}")

    plt.figure(figsize=(8, 3))
    im = plt.imshow(jaccard_matrix_random_onehot, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Jaccard distance")
    plt.title("Jaccard distance", fontsize=14)
    plt.xticks(np.arange(n_onehot), labels_onehot, fontsize=12)
    plt.yticks([0], ["Random"], fontsize=12)
    for j in range(n_onehot):
        plt.text(
            j,
            0,
            f"{jaccard_matrix_random_onehot[0, j]:.2f}",
            ha="center",
            va="center",
            color="white" if jaccard_matrix_random_onehot[0, j] < 0.5 else "black",
            fontsize=12,
        )
    jaccard_pdf = figure_dir / "jaccard_matrix_random_vs_onehot.pdf"
    plt.savefig(str(jaccard_pdf))
    plt.close()

    for dim in all_dims:
        jaccard_random_dim_file = output_dir / f"jaccard_random_vs_onehot_dim{dim}.json"
        jaccard_matrix_random_onehot_dim, _ = load_jaccard_matrix(
            jaccard_random_dim_file, (1, len(onehot_complexes))
        )

        if jaccard_matrix_random_onehot_dim is None:
            jaccard_matrix_random_onehot_dim = np.zeros((1, n_onehot))
            for j in range(n_onehot):
                jaccard_dict = compute_jaccard_distance(
                    sc_random, onehot_complexes[j], by_dimension=True
                )
                val = jaccard_dict.get(dim, 0) # type: ignore
                jaccard_matrix_random_onehot_dim[0, j] = val
            save_jaccard_data(jaccard_matrix_random_onehot_dim, {}, str(jaccard_random_dim_file))
        plt.figure(figsize=(8, 3))
        im = plt.imshow(
            jaccard_matrix_random_onehot_dim, cmap="viridis", vmin=0, vmax=1
        )
        plt.colorbar(im, label="Jaccard distance")
        plt.title(f"Jaccard distance (dimension {dim})", fontsize=14)
        plt.xticks(np.arange(n_onehot), labels_onehot, fontsize=12)
        plt.yticks([0], ["Random"], fontsize=12)
        for j in range(n_onehot):
            val = jaccard_matrix_random_onehot_dim[0, j]
            if not np.isnan(val):
                plt.text(
                    j,
                    0,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.5 else "black",
                    fontsize=12,
                )
        jaccard_pdf = figure_dir / f"jaccard_matrix_random_vs_onehot_dim{dim}.pdf"
        plt.savefig(str(jaccard_pdf))
        plt.close()

    # 5.3. Full Jaccard distance matrix (all DSCs)
    all_complexes_block = onehot_complexes + two_hot_complexes + [sc_random]
    all_labels_block = labels_onehot + labels_twohot + ["Random"]
    n_block = len(all_complexes_block)

    jaccard_full_file = output_dir / "jaccard_matrix_full.json"
    jaccard_matrix_block, _ = load_jaccard_matrix(jaccard_full_file, (n_block, n_block))

    if jaccard_matrix_block is None:
        print("Computing full symmetric Jaccard distance matrix...")
        jaccard_matrix_block = np.zeros((n_block, n_block))
        for i in range(n_block):
            for j in range(i, n_block):
                if (
                    all_complexes_block[i] is not None
                    and all_complexes_block[j] is not None
                ):
                    val = compute_jaccard_distance(
                        all_complexes_block[i], all_complexes_block[j]
                    )
                else:
                    val = np.nan
                jaccard_matrix_block[i, j] = val
                jaccard_matrix_block[j, i] = val
        save_jaccard_data(jaccard_matrix_block, {}, str(jaccard_full_file))
    else:
        print(f"Loaded existing full Jaccard matrix from {jaccard_full_file}")

    # Save full Jaccard matrix to CSV
    import csv

    jaccard_csv = output_dir / "jaccard_matrix_full.csv"
    with open(jaccard_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + all_labels_block)
        for label, row in zip(all_labels_block, jaccard_matrix_block):
            writer.writerow(
                [label] + [f"{val:.6f}" if not np.isnan(val) else "" for val in row]
            )

    plt.figure(figsize=(10, 8))
    im = plt.imshow(jaccard_matrix_block, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Jaccard distance")
    plt.title(f"Jaccard distance: {ctx.jaccard_subject} classifier", fontsize=14)
    plt.xticks(
        np.arange(n_block), all_labels_block, rotation=45, ha="right", fontsize=12
    )
    plt.yticks(np.arange(n_block), all_labels_block, fontsize=12)
    for i in range(n_block):
        for j in range(n_block):
            val = jaccard_matrix_block[i, j]
            if not np.isnan(val):
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.5 else "black",
                    fontsize=12,
                )
    plt.tight_layout()
    jaccard_pdf = figure_dir / "jaccard_matrix_full.pdf"
    plt.savefig(str(jaccard_pdf))
    plt.close()

    # 5.4 Full by-dimension Jaccard distance matrices (all DSCs)
    for dim in all_dims:
        jaccard_full_dim_file = output_dir / f"jaccard_matrix_full_dim{dim}.json"
        jaccard_matrix_block_dim, _ = load_jaccard_matrix(
            jaccard_full_dim_file, (n_block, n_block)
        )

        if jaccard_matrix_block_dim is None:
            print(f"Computing full symmetric Jaccard distance matrix for dimension {dim}...")
            jaccard_matrix_block_dim = np.zeros((n_block, n_block))
            for i in range(n_block):
                for j in range(i, n_block):
                    if (
                        all_complexes_block[i] is not None
                        and all_complexes_block[j] is not None
                    ):
                        jaccard_dict = compute_jaccard_distance(
                            all_complexes_block[i],
                            all_complexes_block[j],
                            by_dimension=True,
                        )
                        val = jaccard_dict.get(dim, 0) # type: ignore
                    else:
                        val = np.nan
                    jaccard_matrix_block_dim[i, j] = val
                    jaccard_matrix_block_dim[j, i] = val
            save_jaccard_data(jaccard_matrix_block_dim, {}, str(jaccard_full_dim_file))
        else:
            print(f"Loaded existing Jaccard matrix for dimension {dim} from {jaccard_full_dim_file}")
        plt.figure(figsize=(10, 8))
        im = plt.imshow(jaccard_matrix_block_dim, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, label="Jaccard distance")
        plt.title(f"Jaccard distance (dimension {dim})", fontsize=14)
        plt.xticks(
            np.arange(n_block), all_labels_block, rotation=45, ha="right", fontsize=12
        )
        plt.yticks(np.arange(n_block), all_labels_block, fontsize=12)
        for i in range(n_block):
            for j in range(n_block):
                val = jaccard_matrix_block_dim[i, j]
                if not np.isnan(val):
                    plt.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white" if val < 0.5 else "black",
                        fontsize=12,
                    )
        jaccard_pdf = figure_dir / f"jaccard_matrix_full_dim{dim}.pdf"
        plt.savefig(str(jaccard_pdf))
        plt.close()


    # ---- probe-point scatter plots ----
    # 6.1 summary plot for all selected points (one-hot, two-hot, random)
    twohot_colors = get_twohot_colors(colors, twohot_indices)

    # Plot: all chosen points (one-hot, two-hot, random) at once
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        class_indices = y == i
        plt.scatter(
            X[class_indices, 0].numpy(),
            X[class_indices, 1].numpy(),
            color="lightgray",
            alpha=1,
            label=None,
        )
    for i, points in enumerate(onehot_data):
        plt.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            color=colors[i],
            marker=markers[i],
            alpha=1,
            label=f"Q{i + 1} one-hot",
        )
    for idx, points in enumerate(twohot_data):
        plt.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            color=[twohot_colors[idx]],
            marker="^",
            alpha=1,
            label=labels_twohot[idx],
        )
    plt.scatter(
        random_points[:, 0].numpy(),
        random_points[:, 1].numpy(),
        color="black",
        marker=random_marker,
        alpha=1,
        label="Random",
    )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    plt.legend(new_handles, new_labels, loc="upper left")
    plt.tight_layout()
    all_chosen_pdf = figure_dir / f"{ctx.prefix}_all_chosen.pdf"
    plt.savefig(str(all_chosen_pdf))
    plt.close()

    # -------------------------------------------------------------------------
    # 6.2 Visualization: scatter plot of predicted class, highlight misclassified
    print(
        "Plotting predicted class scatter plot with misclassifications highlighted..."
    )
    with torch.no_grad():
        outputs = net(X)
        predicted = torch.argmax(outputs, dim=1)

    plt.figure(figsize=(8, 8))
    pred_colors = colors  # Use same color scheme as before
    for i in range(n_classes):
        idx = predicted == i
        plt.scatter(
            X[idx, 0].numpy(),
            X[idx, 1].numpy(),
            color=pred_colors[i],
            marker="o",
            alpha=0.7,
            label=f"Predicted Q{i + 1}",
        )
    # Highlight misclassified points
    misclassified = predicted != y
    if misclassified.any():
        plt.scatter(
            X[misclassified, 0].numpy(),
            X[misclassified, 1].numpy(),
            # An unfilled marker takes its colour from `color`; with
            # color="none" the crosses were invisible.
            color="k",
            marker="x",
            s=80,
            linewidths=2,
            label="Misclassified",
        )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    plt.legend(new_handles, new_labels, loc="upper left")
    plt.tight_layout()
    pred_scatter_pdf = figure_dir / f"{ctx.prefix}_predicted_scatter.pdf"
    plt.savefig(str(pred_scatter_pdf))
    plt.close()

    # -------------------------------------------------------------------------
    # 6.3 Visualization: scatter plot of only Q1 one-hot points highlighted
    plt.figure(figsize=(8, 8))
    # Plot all data in light gray
    for i in range(n_classes):
        class_indices = y == i
        plt.scatter(
            X[class_indices, 0].numpy(),
            X[class_indices, 1].numpy(),
            color="lightgray",
            alpha=1,
            label=None,
        )
    markers = get_onehot_markers(n_classes)
    colors = ["r", "g", "b", "purple"]
    # Highlight Q1 one-hot points (saved index 0)
    plt.scatter(
        onehot_data[0][:, 0].numpy(),
        onehot_data[0][:, 1].numpy(),
        color=colors[0],
        marker=markers[0],
        alpha=1,
        label="Q1 one-hot",
    )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.tight_layout()
    q0_onehot_pdf = figure_dir / f"{ctx.prefix}_q0_onehot.pdf"
    plt.savefig(str(q0_onehot_pdf))
    plt.close()

    # -------------------------------------------------------------------------


def main(run):
    requested = sys.argv[1:]
    contexts = [c for c in CONTEXTS if not requested or c.name in requested]
    unknown = set(requested) - {c.name for c in CONTEXTS}
    if unknown:
        raise SystemExit(f"Unknown context(s): {sorted(unknown)}. "
                         f"Known: {[c.name for c in CONTEXTS]}")

    X, y = generate_quadrant_data()

    drawn, skipped = [], []
    for ctx in contexts:
        run.section(ctx.name)
        loaded = load_context(ctx)
        if loaded is None:
            print(f"  no results in {ctx.output_dir} -- skipping")
            skipped.append(ctx.name)
            continue
        plot_context(ctx, loaded, X, y)
        print(f"  figures written to {ctx.figure_dir}")
        drawn.append(ctx.name)

    print(f"\nDrew {len(drawn)} context(s): {drawn}")
    if skipped:
        print(f"Skipped {len(skipped)} with no results on disk: {skipped}")


if __name__ == "__main__":
    with RunTimer("quadrant figures") as run:
        main(run)
