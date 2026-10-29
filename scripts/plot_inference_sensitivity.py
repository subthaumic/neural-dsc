"""Draw the figures for the joint decision-rule calibration.

Reads the JSON payload written by `compute_inference_sensitivity.py` and turns
it into the figures behind appendix section "Choosing the decision rule". No
complexes are rebuilt here: everything is read off disk, so this can be run
locally against a payload copied back from the server.

The payload carries six views of the same grid, and each gets a figure:

  probe count      does the complex stop changing as B grows?
  margin           what does opening the null region cost in structure?
  effect size      where does a candidate epsilon sit in the |theta|/sigma spectrum?
  correction       do the five corrections agree once epsilon is chosen?
  trajectory       does the complex stop changing as K grows?
  null admission   how many admitted edges are inside the null by the study's
                   own point estimate?

A seventh figure plots the measured per-unit build cost, which is what makes
the grid affordable or not.
"""

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from neural_dsc.utils.progress import RunTimer

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "inference_sensitivity" / "inference_sensitivity.json"
OUTPUT_DIR = REPO_ROOT / "figs" / "inference_sensitivity"

# The slices the summary figures are drawn at. Every one of these is a free
# choice among values the payload actually contains; --alpha, --correction,
# --probes and --k move them.
REFERENCE_ALPHA = 0.01
REFERENCE_CORRECTION = "bonferroni"
REFERENCE_MARGIN = 0.001

# Corrections over the family of simplices tested at one dimension. Membership
# is decided at full ablation alone, so each simplex contributes exactly one
# test and there is nothing to spend across the trajectory.
CORRECTION_LABELS = {
    "none": "uncorrected",
    "bh": "Benjamini--Hochberg",
    "by": "Benjamini--Yekutieli",
    "bonferroni": "Bonferroni",
}
CORRECTION_COLORS = {
    "none": "0.35",
    "bh": "tab:blue",
    "by": "tab:green",
    "bonferroni": "tab:red",
}
PERCENTILE_COLORS = {
    "50": "0.7", "75": "0.55", "90": "tab:blue",
    "95": "tab:green", "99": "tab:orange", "100": "tab:red",
}
SPECTRUM_PERCENTILES = ["50", "75", "90", "95", "99", "100"]


# --- LOADING ---

def load_payload(path):
    """Read the payload and re-key the string-encoded tuples into real tuples."""
    with open(path) as handle:
        payload = json.load(handle)

    payload["series"] = {
        ast.literal_eval(key): value for key, value in payload["series"].items()
    }
    payload["build_seconds"] = {
        ast.literal_eval(key): value for key, value in payload["build_seconds"].items()
    }
    return payload


def contexts(config):
    """The (architecture, class) pairs the run covered, in report order."""
    return [(arch, cls) for arch in config["architectures"] for cls in config["classes"]]


NETWORK_NAMES = {"mono_handcrafted": "monosemantic", "poly_handcrafted": "polysemantic"}


def context_label(arch, cls):
    """Panel name as in the article's text, e.g. "monosemantic $Q_1$"."""
    return f"{NETWORK_NAMES.get(arch, arch)} $Q_{cls + 1}$"


def panel_grid(config, height=2.6, width=3.4):
    """One panel per context, laid out architecture-per-row."""
    n_arch = len(config["architectures"])
    n_cls = len(config["classes"])
    fig, axes = plt.subplots(
        n_arch, n_cls,
        figsize=(width * n_cls, height * n_arch),
        squeeze=False,
    )
    return fig, {
        (arch, cls): axes[i][j]
        for i, arch in enumerate(config["architectures"])
        for j, cls in enumerate(config["classes"])
    }


def shared_legend(fig, handles, labels, ncol):
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))


def save(fig, name, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


# --- FIGURES ---

def plot_probe_count_convergence(payload, outdir, alpha, correction, k):
    """Total simplices against B, one line per margin.

    A margin whose line has gone flat by the largest B is one whose complex the
    probe count no longer decides. Flat but at zero is not a pass -- the dotted
    line marks the margin-0 count, so a curve far below it has bought its
    stability by finding nothing.
    """
    config = payload["config"]
    probe_counts = config["probe_counts"]
    margins = config["margins"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(margins)))

    fig, axes = panel_grid(config)
    for (arch, cls), ax in axes.items():
        for margin, color in zip(margins, colors):
            series = payload["series"].get((arch, cls, alpha, correction, margin, k))
            if series is None:
                continue
            totals = [series["total_simplices_by_count"][str(b)] for b in probe_counts]
            style = ":" if margin == 0.0 else "-"
            ax.plot(probe_counts, totals, style, color=color, marker="o",
                    markersize=3, linewidth=1.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xticks(probe_counts)
        ax.set_xticklabels([str(b) for b in probe_counts], fontsize=7)
        ax.set_title(context_label(arch, cls), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("sample size $B$", fontsize=8)
        ax.set_ylabel("number of simplices", fontsize=8)

    # The article's caption states the parameters and that vertices are not
    # counted, so the figure carries no title.
    handles = [Line2D([], [], color=c, marker="o", markersize=3,
                      linestyle=":" if m == 0.0 else "-",
                      label=f"$\\varepsilon = {m:g}$")
               for m, c in zip(margins, colors)]
    shared_legend(fig, handles, [h.get_label() for h in handles], ncol=len(margins))
    save(fig, "probe_count_convergence", outdir)


def plot_margin_collapse(payload, outdir, alpha, k, probes):
    """Total simplices against the margin, one line per correction.

    This is the price of the interval null. The corrections are drawn together
    because a margin that only stabilises the complex under one of them has
    been fitted rather than calibrated.
    """
    config = payload["config"]
    margins = config["margins"]

    fig, axes = panel_grid(config)
    for (arch, cls), ax in axes.items():
        for correction in config["fdr_methods"]:
            totals = []
            for margin in margins:
                series = payload["series"].get((arch, cls, alpha, correction, margin, k))
                totals.append(
                    series["total_simplices_by_count"][str(probes)] if series else np.nan
                )
            ax.plot(margins, totals, marker="o", markersize=3, linewidth=1.6,
                    color=CORRECTION_COLORS[correction])
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xscale("symlog", linthresh=0.025)
        ax.set_xticks(margins)
        ax.set_xticklabels([f"{m:g}" for m in margins], fontsize=7)
        ax.set_title(context_label(arch, cls), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"margin $\varepsilon$ (units of $\sigma$)", fontsize=8)
        ax.set_ylabel("total simplices", fontsize=8)

    handles = [Line2D([], [], color=CORRECTION_COLORS[m], marker="o", markersize=3,
                      label=CORRECTION_LABELS[m]) for m in config["fdr_methods"]]
    fig.suptitle(
        f"Effect of the margin on complex size  ($B={probes}$, $\\alpha={alpha}$, $K={k}$)",
        fontsize=11,
    )
    shared_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    save(fig, "margin_collapse", outdir)


def plot_effect_size_spectrum(payload, outdir, which="terminal"):
    """Percentiles of the standardised effect |theta|/sigma against B.

    Drawn at full ablation by default, which is where membership is decided; pass
    "peak" for the largest shift anywhere on the trajectory, whose gap against
    the terminal value is how far the coupling is from monotone.

    The horizontal lines are the swept margins. Where a margin sits above the
    99th percentile it is discarding all but the extreme tail; where it sits
    below the median it is not separating anything. This view is independent of
    alpha, the correction and K -- it is a property of the data alone.
    """
    config = payload["config"]
    probe_counts = config["probe_counts"]

    fig, axes = panel_grid(config)
    for (arch, cls), ax in axes.items():
        spectrum = payload["effect_size_spectrum"].get(f"{arch}/class_{cls}")
        if spectrum is None:
            continue
        for pct in SPECTRUM_PERCENTILES:
            values = [spectrum[str(b)].get(which, {}).get(pct, np.nan)
                      for b in probe_counts]
            ax.plot(probe_counts, values, marker="o", markersize=3, linewidth=1.5,
                    color=PERCENTILE_COLORS[pct])
        for margin in config["margins"]:
            if margin == 0.0:
                continue
            ax.axhline(margin, color="crimson", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(probe_counts)
        ax.set_xticklabels([str(b) for b in probe_counts], fontsize=7)
        ax.set_title(context_label(arch, cls), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("probe points $B$", fontsize=8)
        ax.set_ylabel(r"$|\theta|/\sigma$", fontsize=8)

    handles = [Line2D([], [], color=PERCENTILE_COLORS[p], marker="o", markersize=3,
                      label=f"p{p}") for p in SPECTRUM_PERCENTILES]
    handles.append(Line2D([], [], color="crimson", linestyle="--", linewidth=0.7,
                          label="swept margins"))
    at = "at full ablation" if which == "terminal" else "at its peak over the trajectory"
    fig.suptitle(f"Standardised effect-size spectrum {at} "
                 f"(independent of the decision rule)", fontsize=11)
    shared_legend(fig, handles, [h.get_label() for h in handles],
                  ncol=len(SPECTRUM_PERCENTILES) + 1)
    save(fig, "effect_size_spectrum", outdir)


def plot_correction_agreement(payload, outdir):
    """Largest pairwise Jaccard distance, as a context-by-margin grid.

    0.0 means all corrections returned the same filtered complex, so the choice of
    correction is not doing any work at that margin. Cells are annotated with
    the value; the run's own containment check is reported alongside.
    """
    config = payload["config"]
    margins = config["margins"]
    ctx = contexts(config)

    grid = np.full((len(ctx), len(margins)), np.nan)
    for i, (arch, cls) in enumerate(ctx):
        for j, margin in enumerate(margins):
            entry = payload["correction_agreement"].get(
                f"{arch}/class_{cls}|margin={margin}"
            )
            if entry is None:
                continue
            worst = [dim_entry["jaccard_distance"]
                     for pair in entry["pairwise"].values()
                     for dim_entry in pair.values()]
            if worst:
                grid[i, j] = max(worst)

    fig, ax = plt.subplots(figsize=(1.1 * len(margins) + 3.5, 0.45 * len(ctx) + 2.2))
    image = ax.imshow(grid, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(ctx)):
        for j in range(len(margins)):
            if np.isnan(grid[i, j]):
                continue
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_xticks(range(len(margins)))
    ax.set_xticklabels([f"{m:g}" for m in margins])
    ax.set_yticks(range(len(ctx)))
    ax.set_yticklabels([context_label(a, c) for a, c in ctx], fontsize=8)
    ax.set_xlabel(r"margin $\varepsilon$ (units of $\sigma$)")
    ax.set_title("Largest pairwise Jaccard distance across corrections\n"
                 f"($B={max(config['probe_counts'])}$, "
                 f"$\\alpha={max(config['alphas'])}$, $K={config['max_iterations']}$)",
                 fontsize=10)
    fig.colorbar(image, ax=ax, label="Jaccard distance")
    fig.tight_layout()
    save(fig, "correction_agreement", outdir)


def plot_trajectory_resolution(payload, outdir, margin):
    """Total simplices against the ablation-grid resolution K, one line per correction.

    A correctness check rather than a sensitivity curve. Membership is decided
    at full ablation and every ablation grid contains full ablation, so every line
    here must be flat; a slope is a bug. K only sets how precisely the last
    crossing -- the filtration value -- can be located.
    """
    config = payload["config"]
    lengths = config["max_iterations_values"]

    fig, axes = panel_grid(config)
    for (arch, cls), ax in axes.items():
        for correction in config["fdr_methods"]:
            entry = payload["ablation_grid_resolution"].get(
                f"{arch}/class_{cls}|margin={margin}|{correction}"
            )
            if entry is None:
                continue
            totals = [entry["total_by_length"][str(k)] for k in lengths]
            ax.plot(lengths, totals, marker="o", markersize=4, linewidth=1.6,
                    color=CORRECTION_COLORS[correction],
                    linestyle="-" if entry["membership_invariant"] else "--")
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(k) for k in lengths], fontsize=7)
        ax.set_title(context_label(arch, cls), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("ablation-grid resolution $K$", fontsize=8)
        ax.set_ylabel("total simplices", fontsize=8)

    handles = [Line2D([], [], color=CORRECTION_COLORS[m], marker="o", markersize=4,
                      label=CORRECTION_LABELS[m]) for m in config["fdr_methods"]]
    handles.append(Line2D([], [], color="0.4", linestyle="--",
                          label="dashed: membership depends on $K$ (a bug)"))
    fig.suptitle(
        f"Invariance in ablation-grid resolution  ($\\varepsilon={margin:g}$, "
        f"$B={max(config['probe_counts'])}$, $\\alpha={max(config['alphas'])}$)",
        fontsize=11,
    )
    shared_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    save(fig, "trajectory_resolution", outdir)


def plot_null_admission(payload, outdir, alpha, k, probes):
    """Share of admitted 1-simplices whose full-ablation |theta|/sigma is inside the null.

    A plug-in diagnostic rather than an unbiased FDR estimate, but every
    correction is scored against the same point estimates, so it compares them
    fairly. It is undefined at margin 0, where the null region is empty -- which
    is the point of including a point null in the sweep at all.
    """
    config = payload["config"]
    margins = [m for m in config["margins"] if m > 0]

    fig, axes = panel_grid(config)
    for (arch, cls), ax in axes.items():
        for correction in config["fdr_methods"]:
            rates = []
            for margin in margins:
                series = payload["series"].get((arch, cls, alpha, correction, margin, k))
                admission = series.get("null_admission") if series else None
                rates.append(
                    admission["rate"] if admission and admission.get("n_checked") else np.nan
                )
            ax.plot(margins, rates, marker="o", markersize=3, linewidth=1.6,
                    color=CORRECTION_COLORS[correction])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        ax.set_xticks(margins)
        ax.set_xticklabels([f"{m:g}" for m in margins], fontsize=7)
        ax.set_title(context_label(arch, cls), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"margin $\varepsilon$ (units of $\sigma$)", fontsize=8)
        ax.set_ylabel("admitted inside null", fontsize=8)

    handles = [Line2D([], [], color=CORRECTION_COLORS[m], marker="o", markersize=3,
                      label=CORRECTION_LABELS[m]) for m in config["fdr_methods"]]
    fig.suptitle(
        f"Null-region admission rate  ($B={probes}$, $\\alpha={alpha}$, $K={k}$)",
        fontsize=11,
    )
    shared_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    save(fig, "null_admission", outdir)


def plot_build_cost(payload, outdir):
    """Measured single-core construction time per work unit, against B.

    Cost is a steep function of how small the margin is: at a point null the
    inductive construction has many more surviving faces to extend, and the
    number of candidate simplices grows combinatorially. This figure is what
    makes the wall clock of the next sweep predictable.
    """
    config = payload["config"]
    probe_counts = config["probe_counts"]
    margins = config["margins"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(margins)))

    fig, axes = panel_grid(config)
    for (arch, cls), ax in axes.items():
        for margin, color in zip(margins, colors):
            times = [payload["build_seconds"].get((arch, cls, b, margin), np.nan)
                     for b in probe_counts]
            ax.plot(probe_counts, times, marker="o", markersize=3, linewidth=1.6,
                    color=color, linestyle=":" if margin == 0.0 else "-")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(probe_counts)
        ax.set_xticklabels([str(b) for b in probe_counts], fontsize=7)
        ax.set_title(context_label(arch, cls), fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("probe points $B$", fontsize=8)
        ax.set_ylabel("build time (s, one core)", fontsize=8)

    handles = [Line2D([], [], color=c, marker="o", markersize=3,
                      linestyle=":" if m == 0.0 else "-", label=f"$\\varepsilon$ = {m:g}")
               for m, c in zip(margins, colors)]
    fig.suptitle(
        f"Construction cost per work unit  ($K={config['max_iterations']}$, "
        f"$\\Delta t={config['dt']}$, single core)", fontsize=11,
    )
    shared_legend(fig, handles, [h.get_label() for h in handles], ncol=len(margins))
    save(fig, "build_cost", outdir)


# --- SUMMARY ---

def print_summary(payload, alpha, correction, k, probes):
    """The decision the figures are drawn to support, as a table."""
    config = payload["config"]
    print()
    print(f"Reference slice: alpha={alpha}, correction={correction}, "
          f"K={k}, B={probes}")
    print(f"{'context':<24} {'margin':>7} {'total':>7} {'topdim':>7} "
          f"{'convB':>6} {'null/n':>9}")
    for arch, cls in contexts(config):
        for margin in config["margins"]:
            series = payload["series"].get((arch, cls, alpha, correction, margin, k))
            if series is None:
                continue
            admission = series.get("null_admission") or {}
            checked = admission.get("n_checked")
            share = (f"{admission.get('n_null_admitted')}/{checked}"
                     if checked else "-")
            print(f"{context_label(arch, cls):<24} {margin:>7g} "
                  f"{series['total_simplices_by_count'][str(probes)]:>7} "
                  f"{series['top_dimension_by_count'][str(probes)]:>7} "
                  f"{str(series['converged_in_probe_count']):>6} {share:>9}")

    violations = {key: entry["containment_violations"]
                  for key, entry in payload["correction_agreement"].items()
                  if entry["containment_violations"]}
    print()
    if violations:
        print(f"WARNING: {len(violations)} contexts have containment violations "
              f"(a correction admitted a simplex the uncorrected test rejected):")
        for key, entry in violations.items():
            print(f"  {key}: {entry}")
    else:
        print("Containment holds everywhere: no correction admitted a simplex "
              "the uncorrected test rejected.")

    total_build = payload["total_build_seconds"]
    worst_key, worst = max(payload["build_seconds"].items(), key=lambda kv: kv[1])
    print(f"\nWall clock {payload['runtime_seconds'] / 3600:.2f} h against "
          f"{total_build / 3600:.2f} CPU-h over {config['n_workers']} workers "
          f"(efficiency {total_build / payload['runtime_seconds']:.1f}x).")
    print(f"Worst single unit: {worst_key} at {worst:.0f} s "
          f"({100 * worst / total_build:.0f}% of all CPU time).")


FIGURES = {
    "probe_count": lambda p, o, a: plot_probe_count_convergence(
        p, o, a.alpha, a.correction, a.k),
    "margin": lambda p, o, a: plot_margin_collapse(p, o, a.alpha, a.k, a.probes),
    "spectrum": lambda p, o, a: plot_effect_size_spectrum(p, o),
    "agreement": lambda p, o, a: plot_correction_agreement(p, o),
    "trajectory": lambda p, o, a: plot_trajectory_resolution(p, o, a.margin),
    "null": lambda p, o, a: plot_null_admission(p, o, a.alpha, a.k, a.probes),
    "cost": lambda p, o, a: plot_build_cost(p, o),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", metavar="FIGURE",
                        help=f"which figures to draw, from {sorted(FIGURES)} "
                             f"or 'all' (default: all)")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=REFERENCE_ALPHA)
    parser.add_argument("--correction", default=REFERENCE_CORRECTION)
    parser.add_argument("--margin", type=float, default=REFERENCE_MARGIN)
    parser.add_argument("--probes", type=int, default=None,
                        help="reference probe count (default: the largest swept)")
    parser.add_argument("--k", type=int, default=None,
                        help="reference trajectory length (default: the longest swept)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.results.exists():
        raise SystemExit(
            f"No payload at {args.results}. Run compute_inference_sensitivity.py "
            f"first, or point --results at a copy from the server."
        )

    payload = load_payload(args.results)
    config = payload["config"]

    # Defaults that depend on the payload, and a check that the requested slice
    # is one the run actually covered -- a silently empty figure is worse than
    # a refusal.
    if args.probes is None:
        args.probes = max(config["probe_counts"])
    if args.k is None:
        args.k = max(config["max_iterations_values"])
    for name, value, allowed in (
        ("alpha", args.alpha, config["alphas"]),
        ("correction", args.correction, config["fdr_methods"]),
        ("margin", args.margin, config["margins"]),
        ("probes", args.probes, config["probe_counts"]),
        ("k", args.k, config["max_iterations_values"]),
    ):
        if value not in allowed:
            raise SystemExit(f"--{name}={value} is not in the payload; swept: {allowed}")

    print(f"Payload: {args.results}")
    print(f"Generated: {payload['generated_at']}")
    print(f"Figures:  {args.outdir}")

    unknown = [name for name in args.figures if name != "all" and name not in FIGURES]
    if unknown:
        raise SystemExit(f"Unknown figures {unknown}; available: {sorted(FIGURES)}")

    selected = sorted(FIGURES) if not args.figures or "all" in args.figures else args.figures
    for name in selected:
        FIGURES[name](payload, args.outdir, args)

    print_summary(payload, args.alpha, args.correction, args.k, args.probes)


if __name__ == "__main__":
    with RunTimer("inference sensitivity figures"):
        main()
