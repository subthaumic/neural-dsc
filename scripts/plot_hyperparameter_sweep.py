#!/usr/bin/env python3
"""Draw the aggregate figures for the trajectory/probe hyperparameter sweep.

The compute script writes detailed figures inside each architecture/class
directory. This script answers the cross-context questions for which the sweep
was run:

  membership  where is the *exact* simplex set stable in every context?
  dt          how much do filtration values and their ordering move with dt?
  tolerance   which parts of the filtered complex depend on the probe region?
  probes      what purity/availability cost does a tighter probe region impose?
  default     what structures and probe regions does the chosen configuration use?

Exact simplex identities and filtration values are read from the per-context
pickles. The aggregate CSVs alone contain only dimension counts, which are not
enough to establish that two complexes agree. No complexes are rebuilt here.

USAGE:
  python scripts/plot_hyperparameter_sweep.py
  python scripts/plot_hyperparameter_sweep.py dt tolerance

Outputs are written as PDF and PNG to ``figs/hyperparameter_sweep/``.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

from neural_dsc.utils.jaccard_index import filtration_jaccard_distance

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "hyperparameter_sweep"
OUTPUT_DIR = REPO_ROOT / "figs" / "hyperparameter_sweep"

DEFAULT_DT = 0.01
DEFAULT_TOLERANCE = 0.01
IMMEDIATE_THRESHOLD = 0.01

MATCH = "#2f7d4f"
DIFFER = "#b23a3a"
ABSENT = "#8a8a8a"
DIMENSION_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
]


NETWORK_NAMES = {"mono_handcrafted": "monosemantic", "poly_handcrafted": "polysemantic"}


def context_label(context):
    """Label for an (architecture, class) pair, as in the article: "monosemantic $Q_1$"."""
    architecture, class_idx = context
    return f"{NETWORK_NAMES.get(architecture, architecture)} $Q_{class_idx + 1}$"


def context_palette(contexts):
    """Use warm colours for mono contexts and cool colours for poly contexts."""
    mono = plt.cm.Oranges(np.linspace(0.45, 0.9, 4))
    poly = plt.cm.Blues(np.linspace(0.45, 0.9, 4))
    return {
        context: (mono[context[1]] if context[0].startswith("mono") else poly[context[1]])
        for context in contexts
    }


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_payload(results_dir):
    """Load aggregate diagnostics plus the complexes needed for exact comparisons."""
    table_paths = {
        "dt_table": results_dir / "dt_sweep_table_all_contexts.csv",
        "tolerance_table": results_dir / "probe_tolerance_sweep_table_all_contexts.csv",
        "probe_diagnostics": results_dir / "probe_selection_diagnostics.csv",
    }
    missing = [str(path) for path in table_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing sweep outputs:\n  " + "\n  ".join(missing))

    payload = {name: pd.read_csv(path) for name, path in table_paths.items()}
    dt_table = payload["dt_table"]
    contexts = sorted(
        {(str(row.architecture), int(row["class"])) for _, row in dt_table.iterrows()},
        key=lambda context: (0 if context[0].startswith("mono") else 1, context[1]),
    )
    payload["contexts"] = contexts
    payload["dt_values"] = sorted(float(value) for value in dt_table["dt"].unique())
    payload["tolerance_values"] = sorted(
        float(value) for value in payload["tolerance_table"]["probe_tolerance"].unique()
    )
    payload["dt_results"] = {}
    payload["tolerance_results"] = {}

    for architecture, class_idx in contexts:
        context_dir = results_dir / architecture / f"class_{class_idx}"
        dt_path = context_dir / "dt_sweep_results.pkl"
        tolerance_path = context_dir / "probe_tolerance_sweep_results.pkl"
        if not dt_path.exists() or not tolerance_path.exists():
            raise SystemExit(
                f"Missing per-context pickles under {context_dir}; exact-set and "
                "filtration plots require the complete compute output."
            )
        payload["dt_results"][(architecture, class_idx)] = load_pickle(dt_path)
        payload["tolerance_results"][(architecture, class_idx)] = load_pickle(
            tolerance_path
        )

    return payload


def lookup(results, value):
    """Retrieve a float-keyed result without relying on identical parsing."""
    return next(
        (result for key, result in results.items() if np.isclose(float(key), value)),
        None,
    )


def is_available(result):
    return result is not None and not result.get("skipped", False)


def simplex_map(result):
    """Map (dimension, simplex) to filtration, excluding fixed vertices."""
    return {
        (int(dimension), tuple(simplex)): float(filtration)
        for dimension, rows in result["dsc"].complex.items()
        if int(dimension) > 0
        for simplex, filtration in rows
    }


def simplex_set(result, threshold=None):
    filtrations = simplex_map(result)
    if threshold is None:
        return set(filtrations)
    return {
        simplex for simplex, filtration in filtrations.items()
        if filtration <= threshold + 1e-12
    }


def count_signature(result):
    """Counts by positive dimension, with trailing empty dimensions removed."""
    counts = result["counts"]
    dimensions = sorted(int(dimension) for dimension in counts if int(dimension) > 0)
    signature = [int(counts.get(dimension, 0)) for dimension in dimensions]
    while signature and signature[-1] == 0:
        signature.pop()
    return tuple(signature)


def signature_label(result):
    signature = count_signature(result)
    return "/".join(str(count) for count in signature) if signature else "--"


def comparison(reference, current):
    """Membership, filtration displacement, and ordering against a reference."""
    reference_map = simplex_map(reference)
    current_map = simplex_map(current)
    reference_set = set(reference_map)
    current_set = set(current_map)
    union = reference_set | current_set
    common = sorted(reference_set & current_set)
    differences = [abs(current_map[s] - reference_map[s]) for s in common]

    comparable_pairs = 0
    ties = 0
    inversions = 0
    for i, simplex_a in enumerate(common):
        for simplex_b in common[i + 1:]:
            reference_difference = reference_map[simplex_a] - reference_map[simplex_b]
            if reference_difference == 0:
                continue
            comparable_pairs += 1
            current_difference = current_map[simplex_a] - current_map[simplex_b]
            if current_difference == 0:
                ties += 1
            elif reference_difference * current_difference < 0:
                inversions += 1

    return {
        "jaccard_distance": filtration_jaccard_distance(
            reference_map, current_map
        ),
        "exact_membership": reference_set == current_set,
        "max_shift": max(differences, default=0.0),
        "mean_shift": float(np.mean(differences)) if differences else 0.0,
        "inversion_rate": inversions / comparable_pairs if comparable_pairs else 0.0,
        "tie_rate": ties / comparable_pairs if comparable_pairs else 0.0,
    }


def tightest_available(results, values):
    """The finest/tightest row that actually produced a complex."""
    for value in values:
        result = lookup(results, value)
        if is_available(result):
            return value, result
    raise ValueError("sweep contains no completed rows")


def draw_default_outline(ax, values, default):
    if any(np.isclose(default, value) for value in values):
        index = next(i for i, value in enumerate(values) if np.isclose(default, value))
        ax.add_patch(
            Rectangle(
                (index - 0.5, -0.5), 1, ax.get_ylim()[0] + 0.5,
                facecolor="none", edgecolor="black", linewidth=2.2, zorder=5,
            )
        )


def style_parameter_axis(ax, values, xlabel, show_ylabels, contexts):
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"{value:g}" for value in values])
    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels(
        [context_label(context) for context in contexts] if show_ylabels else []
    )
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("pdf", 300), ("png", 150)):
        path = outdir / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def annotated_heatmap(
    ax,
    values,
    contexts,
    parameters,
    title,
    xlabel,
    cmap,
    vmin,
    vmax,
    formatter,
    default,
    show_ylabels,
):
    """Draw a context-by-parameter matrix with values printed in every cell."""
    masked = np.ma.masked_invalid(values)
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad(ABSENT)
    image = ax.imshow(masked, aspect="auto", cmap=colour_map, vmin=vmin, vmax=vmax)
    normalise = Normalize(vmin=vmin, vmax=vmax)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                text, colour = "", "white"
            else:
                rgba = colour_map(normalise(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text = formatter(value)
                colour = "black" if luminance > 0.56 else "white"
            ax.text(
                column, row, text, ha="center", va="center",
                fontsize=7.5, color=colour, fontweight="bold",
            )

    ax.set_ylim(len(contexts) - 0.5, -0.5)
    style_parameter_axis(ax, parameters, xlabel, show_ylabels, contexts)
    ax.set_title(title, fontsize=10)
    draw_default_outline(ax, parameters, default)
    return image


def membership_panel(ax, results_by_context, contexts, values, default, title, xlabel):
    """Exact membership agreement, annotated by dimension-count signature."""
    for row, context in enumerate(contexts):
        results = results_by_context[context]
        reference_value, reference = tightest_available(results, values)
        reference_set = simplex_set(reference)
        for column, value in enumerate(values):
            result = lookup(results, value)
            if not is_available(result):
                colour, text = ABSENT, ""
            else:
                colour = MATCH if simplex_set(result) == reference_set else DIFFER
                text = signature_label(result)
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5), 1, 1,
                    facecolor=colour, edgecolor="white", linewidth=1.5,
                )
            )
            ax.text(
                column, row, text, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold",
            )
        if not np.isclose(reference_value, values[0]):
            ax.text(
                -0.65, row, f"ref {reference_value:g}", ha="right", va="center",
                fontsize=7, color="0.35",
            )

    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_ylim(len(contexts) - 0.5, -0.5)
    style_parameter_axis(ax, values, xlabel, True, contexts)
    ax.set_title(title, fontsize=10)
    draw_default_outline(ax, values, default)


def plot_membership(payload, outdir):
    """Exact simplex membership across both sweep axes."""
    contexts = payload["contexts"]
    dt_values = payload["dt_values"]
    tolerance_values = payload["tolerance_values"]
    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, 4.7),
        gridspec_kw={"width_ratios": [len(dt_values), len(tolerance_values)]},
    )
    membership_panel(
        axes[0], payload["dt_results"], contexts, dt_values, DEFAULT_DT,
        f"Integration step (reference $\\Delta t={min(dt_values):g}$)",
        "integration step $\\Delta t$",
    )
    membership_panel(
        axes[1], payload["tolerance_results"], contexts, tolerance_values,
        DEFAULT_TOLERANCE,
        "Selection tolerance (reference: smallest $\\eta$ with 128 inputs)",
        "selection tolerance $\\eta$",
    )
    axes[1].set_yticklabels([])
    handles = [
        Patch(facecolor=MATCH, label="same simplex set as the reference"),
        Patch(facecolor=DIFFER, label="different simplex set"),
        Patch(facecolor=ABSENT, label="fewer than 128 inputs accepted"),
        Patch(facecolor="none", edgecolor="black", linewidth=2.2, label="chosen value"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.01), fontsize=9,
    )
    fig.suptitle(
        "Simplex sets  (cells show the number of simplices in dimensions 1/2/3/...)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "sweep_stability_map", outdir)


def plot_dt(payload, outdir):
    """Filtration-value drift and ordering stability as the integrator is coarsened."""
    contexts = payload["contexts"]
    values = payload["dt_values"]
    palette = context_palette(contexts)
    max_shift = np.zeros((len(contexts), len(values)))
    inversion_rate = np.zeros_like(max_shift)

    for row, context in enumerate(contexts):
        results = payload["dt_results"][context]
        _, reference = tightest_available(results, values)
        for column, value in enumerate(values):
            metrics = comparison(reference, lookup(results, value))
            max_shift[row, column] = metrics["max_shift"]
            inversion_rate[row, column] = metrics["inversion_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for row, context in enumerate(contexts):
        axes[0].plot(
            values, max_shift[row], marker="o", markersize=4, linewidth=1.6,
            color=palette[context], label=context_label(context),
        )
    axes[0].axvline(DEFAULT_DT, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_xticks(values)
    axes[0].set_xticklabels([f"{value:g}" for value in values])
    axes[0].set_xlabel("integration step $\\Delta t$")
    axes[0].set_ylabel(f"shift relative to $\\Delta t={min(values):g}$")
    axes[0].set_title("Largest shift of a filtration value")
    axes[0].grid(alpha=0.25, which="both")

    image = annotated_heatmap(
        axes[1], 100 * inversion_rate, contexts, values,
        "Reversed pairs of simplices", "integration step $\\Delta t$", "YlOrRd", 0,
        max(1.0, float(np.nanmax(100 * inversion_rate))),
        lambda value: f"{value:.1f}%" if value else "0",
        DEFAULT_DT, True,
    )
    fig.colorbar(image, ax=axes[1], label="reversed pairs (%)")
    handles = [
        Line2D([], [], color=palette[context], marker="o", label=context_label(context))
        for context in contexts
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=8,
    )
    # No title: the article's caption describes the figure.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "dt_filtration_stability", outdir)


def plot_tolerance(payload, outdir):
    """Membership and filtration sensitivity to the selected probe region."""
    contexts = payload["contexts"]
    values = payload["tolerance_values"]
    shape = (len(contexts), len(values))
    full_jaccard_distance = np.full(shape, np.nan)
    immediate_jaccard_distance = np.full(shape, np.nan)
    max_shift = np.full(shape, np.nan)
    inversion_rate = np.full(shape, np.nan)

    for row, context in enumerate(contexts):
        results = payload["tolerance_results"][context]
        _, reference = tightest_available(results, values)
        reference_immediate = simplex_set(reference, IMMEDIATE_THRESHOLD)
        for column, value in enumerate(values):
            current = lookup(results, value)
            if not is_available(current):
                continue
            metrics = comparison(reference, current)
            current_immediate = simplex_set(current, IMMEDIATE_THRESHOLD)
            union = reference_immediate | current_immediate
            full_jaccard_distance[row, column] = metrics["jaccard_distance"]
            immediate_jaccard_distance[row, column] = (
                1.0 - len(reference_immediate & current_immediate) / len(union)
                if union
                else 0.0
            )
            max_shift[row, column] = metrics["max_shift"]
            inversion_rate[row, column] = metrics["inversion_rate"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.0), sharex=True)
    images = []
    images.append(annotated_heatmap(
        axes[0, 0], full_jaccard_distance, contexts, values,
        "Jaccard distance $d_J$", "", "YlOrRd", 0, 1,
        lambda value: f"{value:.2f}", DEFAULT_TOLERANCE, True,
    ))
    images.append(annotated_heatmap(
        axes[0, 1], immediate_jaccard_distance, contexts, values,
        f"Jaccard distance, filtration values $\\leq {IMMEDIATE_THRESHOLD:g}$",
        "", "YlOrRd", 0, 1, lambda value: f"{value:.2f}",
        DEFAULT_TOLERANCE, False,
    ))
    shift_max = max(0.01, float(np.nanmax(max_shift)))
    images.append(annotated_heatmap(
        axes[1, 0], max_shift, contexts, values,
        "Largest shift of a filtration value", "selection tolerance $\\eta$",
        "YlOrRd", 0, shift_max, lambda value: f"{value:.3f}",
        DEFAULT_TOLERANCE, True,
    ))
    inversion_max = max(0.01, float(np.nanmax(inversion_rate)))
    images.append(annotated_heatmap(
        axes[1, 1], 100 * inversion_rate, contexts, values,
        "Reversed pairs of simplices", "selection tolerance $\\eta$",
        "YlOrRd", 0, 100 * inversion_max,
        lambda value: f"{value:.1f}%" if value else "0",
        DEFAULT_TOLERANCE, False,
    ))
    fig.colorbar(images[0], ax=axes[0, 0], label="$d_J$")
    fig.colorbar(images[1], ax=axes[0, 1], label="Jaccard distance")
    fig.colorbar(images[2], ax=axes[1, 0], label="largest shift")
    fig.colorbar(images[3], ax=axes[1, 1], label="reversed pairs (%)")
    # No title: the article's caption describes the figure and its reference.
    fig.tight_layout(rect=(0, 0, 1, 1))
    save(fig, "probe_tolerance_stability", outdir)


def parameter_axis(ax, values):
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xticks(values)
    ax.set_xticklabels(
        [f"{value:g}" for value in values], rotation=30, ha="right", fontsize=8
    )
    ax.axvline(DEFAULT_TOLERANCE, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("selection tolerance $\\eta$")
    ax.grid(alpha=0.25, which="both")


def plot_probes(payload, outdir):
    """Probe availability versus the purity sacrificed by widening tolerance."""
    contexts = payload["contexts"]
    values = payload["tolerance_values"]
    palette = context_palette(contexts)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    for context in contexts:
        results = payload["tolerance_results"][context]
        diagnostics = [lookup(results, value)["probe_diagnostics"] for value in values]
        acceptance = [entry["acceptance_rate"] for entry in diagnostics]
        attempts = [entry["n_attempts"] for entry in diagnostics]
        min_purity = [entry["min_purity"] for entry in diagnostics]
        skipped = [not entry["converged"] for entry in diagnostics]

        for ax, series in zip(axes, (acceptance, attempts, min_purity)):
            ax.plot(
                values, series, marker="o", markersize=4, linewidth=1.5,
                color=palette[context], label=context_label(context),
            )
        for column, was_skipped in enumerate(skipped):
            if was_skipped:
                axes[0].scatter(
                    values[column], acceptance[column], marker="x", s=65,
                    linewidth=2, color=palette[context], zorder=5,
                )
                axes[1].scatter(
                    values[column], attempts[column], marker="x", s=65,
                    linewidth=2, color=palette[context], zorder=5,
                )
                axes[2].scatter(
                    values[column], min_purity[column], marker="x", s=65,
                    linewidth=2, color=palette[context], zorder=5,
                )

    axes[0].set_yscale("log")
    axes[0].set_ylabel("accepted draws / attempted draws")
    axes[0].set_title("Acceptance rate")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("uniform draws attempted")
    axes[1].set_title("Draws needed for 128 inputs")
    axes[2].set_ylim(0.89, 1.005)
    axes[2].set_ylabel("smallest selection purity")
    axes[2].set_title("Selection purity of the accepted inputs")
    for ax in axes:
        parameter_axis(ax, values)

    handles = [
        Line2D([], [], color=palette[context], marker="o", label=context_label(context))
        for context in contexts
    ]
    handles.append(Line2D([], [], color="0.3", marker="x", linestyle="none",
                          markersize=7, label="fewer than 128 inputs accepted"))
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.05), fontsize=8,
    )
    fig.suptitle("Cost of input selection against selection purity", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    save(fig, "probe_selection_tradeoff", outdir)


def plot_default(payload, outdir):
    """Structure and probe diagnostics at the repository defaults."""
    contexts = payload["contexts"]
    palette = context_palette(contexts)
    positions = np.arange(len(contexts))
    dt_results = [lookup(payload["dt_results"][context], DEFAULT_DT) for context in contexts]
    positive_dimensions = sorted({
        int(dimension)
        for result in dt_results
        for dimension, count in result["counts"].items()
        if int(dimension) > 0 and count > 0
    })

    diagnostics = payload["probe_diagnostics"].set_index(["architecture", "class"])
    acceptance = np.array([
        float(diagnostics.loc[context, "acceptance_rate"]) for context in contexts
    ])
    off_class_fraction = np.array([
        (
            float(diagnostics.loc[context, "dataset_accepted_off_class_count"])
            / float(diagnostics.loc[context, "dataset_accepted_count"])
        ) if diagnostics.loc[context, "dataset_accepted_count"] else 0.0
        for context in contexts
    ])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bottom = np.zeros(len(contexts))
    for colour, dimension in zip(DIMENSION_COLORS, positive_dimensions):
        values = np.array([
            int(result["counts"].get(dimension, 0)) for result in dt_results
        ])
        axes[0].bar(
            positions, values, bottom=bottom, color=colour,
            label=f"{dimension}-simplices",
        )
        bottom += values
    axes[0].set_ylabel("non-vertex simplices")
    axes[0].set_title("Complex structure")
    axes[0].legend(frameon=False, fontsize=8)

    bars = axes[1].bar(positions, acceptance, color=[palette[c] for c in contexts])
    axes[1].bar_label(bars, labels=[f"{100*v:.1f}%" for v in acceptance],
                      padding=2, fontsize=7, rotation=90)
    axes[1].set_ylim(0, max(acceptance) * 1.28)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylabel("acceptance rate")
    axes[1].set_title("Input availability")

    bars = axes[2].bar(
        positions, off_class_fraction, color=[palette[c] for c in contexts]
    )
    axes[2].bar_label(
        bars, labels=[f"{100*v:.1f}%" for v in off_class_fraction],
        padding=2, fontsize=7, rotation=90,
    )
    axes[2].set_ylim(0, max(0.02, max(off_class_fraction) * 1.28))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_ylabel("off-class among accepted reference points")
    axes[2].set_title("Off-class points in the selection region")

    labels = [context_label(context) for context in contexts]
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Chosen configuration  ($\\Delta t={DEFAULT_DT:g}$, "
        f"$\\eta={DEFAULT_TOLERANCE:g}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "default_context_summary", outdir)


FIGURES = {
    "membership": plot_membership,
    "dt": plot_dt,
    "tolerance": plot_tolerance,
    "probes": plot_probes,
    "default": plot_default,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures", nargs="*", metavar="FIGURE",
        help=f"figures to draw, from {sorted(FIGURES)} or 'all' (default: all)",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = [name for name in args.figures if name != "all" and name not in FIGURES]
    if unknown:
        raise SystemExit(f"Unknown figures {unknown}; available: {sorted(FIGURES)}")
    selected = list(FIGURES) if not args.figures or "all" in args.figures else args.figures

    payload = load_payload(args.results)
    print(f"Results: {args.results}")
    print(f"Figures: {args.outdir}")
    print(f"Contexts: {len(payload['contexts'])}")
    for name in selected:
        FIGURES[name](payload, args.outdir)


if __name__ == "__main__":
    main()
