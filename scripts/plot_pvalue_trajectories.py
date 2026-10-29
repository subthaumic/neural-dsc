"""Draw the ablation-response behaviour of the p-values along CCA trajectories.

Reads the caches written by `compute_pvalue_trajectories.py`. Each record is one
candidate simplex and carries its whole p-value sequence over the ablation axis
s in [0, 1], where s = 1 - sum_S a_i(gamma(s)) / sum_S a_i(x0) and s = 1 is the
fully ablated counterfactual.

Two things are drawn, and they answer the two separate questions the decision
rule asks of a trajectory:

  * Whether it is monotone. Membership is settled at s = 1, but the filtration
    is the *last crossing* -- the smallest ablation from which the test stays
    significant to the end -- and calling that a minimum effective ablation, the
    dose-response literature's minimum effective dose with the ablation fraction
    in place of a dose, is only honest if the test, once significant, stays
    significant. Where it does, the
    first crossing and the last coincide and the trajectory is a clean
    ablation-response curve. Where it does not, the filtration marks where the
    flickering stopped rather than where the effect began, and the gap between
    the two crossings is what that costs.
  * Where the cutoff falls. The correction applies to the family of simplices
    tested at one dimension, not to the ablation axis, so each dimension has exactly
    one cutoff and it is a horizontal line. Reading the ablation axis top-down needs
    no adjustment of its own: that is the fixed-sequence (step-down) MED
    procedure, which controls the familywise error over ablations for free.

The cutoffs are read out of `family_cutoff` and the crossings out of
`persistent_crossing_index` -- the same functions the construction calls -- so
the figure cannot drift away from the rule applied to the complex.

Outputs, under `figs/pvalue_trajectories/`:

  pvalue_trajectories_<arch>_class_<c>.pdf/png    p against ablation, per dimension
  pvalue_ablation_response_<arch>_class_<c>.pdf/png   crossings and their spread
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT / "src"))

from neural_dsc.directed_simplicial_complex import build_dsc_from_cached_pvalues
from neural_dsc.io.persistence import load_pvalues
from neural_dsc.utils.fdr_correction import family_cutoff, persistent_crossing_index
from neural_dsc.utils.progress import RunTimer

from compute_pvalue_trajectories import (  # noqa: E402  (path set above)
    DT,
    MAX_DIMENSIONS,
    MODEL_FACTORIES,
)

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "pvalue_trajectories"
OUTPUT_DIR = REPO_ROOT / "figs" / "pvalue_trajectories"

ALPHA = 0.01  # must match compute_pvalue_trajectories.py
DECISIVE = "bonferroni"  # the correction the complexes are built under
FLOOR = 1e-12  # p-values of exactly 0 would vanish on a log axis

TRAJECTORY_COLOR = "steelblue"  # admitted, and monotone in ablation
RAGGED_COLOR = "tab:orange"     # admitted, but the first crossing is not the last
REJECTED_COLOR = "0.75"         # never significant at s = 1, so never admitted

# The correction the complexes are built under comes first, with the three
# references it is worth reading against. BH is the classical FDR procedure, and
# the gap to BY is the cost of not being able to assume PRDS. Bonferroni needs
# no dependence assumption either, but controls the familywise error rather than
# the false discovery rate -- a strictly stronger guarantee, so it is not simply
# "more conservative than BY": its threshold alpha/m is flat, while BY's rises
# with the number of rejections, so BY is stricter than Bonferroni when few
# hypotheses are rejected (k < H_m) and more permissive once many are.
METHODS = ["none", "by", "bh", "bonferroni"]
METHOD_LABELS = {
    "none": "uncorrected",
    "by": "Benjamini\u2013Yekutieli",
    "bh": "Benjamini\u2013Hochberg",
    "bonferroni": "Bonferroni",
}
METHOD_COLORS = {
    "none": "0.35", "by": "tab:green", "bh": "tab:blue", "bonferroni": "tab:red",
}
METHOD_STYLES = {"none": "--", "by": "-", "bh": ":", "bonferroni": "-."}


def families_actually_posed(records, architecture, alpha, method):
    """Replay the construction and return what it evaluated at each dimension.

    Not a regrouping of the cache by dimension, and the difference matters.
    Construction is inductive: a k-simplex is only tested when all of its
    (k-1)-faces are in the complex, so a stricter correction that drops one edge
    also silences every triangle that edge would have supported. Those triangles
    are never posed as hypotheses and have no place in a figure describing the
    run -- however small the p-value sitting in the cache for them, which was
    computed by whichever more permissive build wrote it.

    Returns {dimension: [(simplex, ablation_grid, p_values)]}, empty dimensions dropped.
    """
    model = MODEL_FACTORIES[architecture]()
    model.eval()
    _, posed = build_dsc_from_cached_pvalues(
        records, model, MAX_DIMENSIONS[architecture], DT, alpha, method,
        verbose=False, return_families=True,
    )
    return {dim: entry["family"] for dim, entry in sorted(posed.items())
            if entry["family"]}


def cutoffs_for(family, alpha):
    """Each method's threshold on one dimension's family.

    Every method is scored on the *same* family -- the one the construction
    actually posed under the decisive correction -- so the lines are comparable.
    `family_cutoff` scores the final p-values, because that is where membership
    is decided, and returns the critical value the step-up stopped at. It comes
    back as -inf when a method rejects nothing, which is drawn as an absent line
    rather than as a line at zero.
    """
    finals = np.array([float(p_vals[-1]) for _, _, p_vals in family])
    out = {}
    for method in METHODS:
        cut = family_cutoff(finals, alpha=alpha, method=method)
        out[method] = None if not np.isfinite(cut) else float(cut)
    # "none" comes back as the largest float below alpha, which is alpha for
    # every drawing purpose and reads better on the axis.
    out["none"] = alpha
    return out


def crossings(p_values, cutoff):
    """(first, last) crossing indices under one cutoff, or (None, None).

    The last crossing is the filtration value. The first is drawn beside it
    only to expose the gap: they differ exactly when the trajectory dips back
    above the cutoff after having crossed it.
    """
    flags = np.asarray(p_values) <= cutoff
    last = persistent_crossing_index(flags)
    if last is None:
        return None, None
    return int(np.argmax(flags)), last


def plot_trajectories(families, cutoffs, label, outdir, stem):
    """p against ablation, one panel per dimension, with the family cutoffs over it."""
    dims = list(families)
    fig, axes = plt.subplots(1, len(dims), figsize=(6.5 * len(dims), 5),
                             squeeze=False)
    axes = axes[0]
    any_ragged = False

    for ax, dim in zip(axes, dims):
        family = families[dim]
        cuts = cutoffs[dim]
        decisive = cuts[DECISIVE] if cuts[DECISIVE] is not None else -np.inf
        n_ragged = 0

        for _simplex, _ablation_grid, raw in family:
            p_values = np.clip(np.asarray(raw, dtype=float), FLOOR, 1.0)
            ablation_grid = np.linspace(0.0, 1.0, p_values.size)
            first, last = crossings(raw, decisive)
            ragged = last is not None and first != last
            n_ragged += bool(ragged)
            if last is None:
                style = dict(color=REJECTED_COLOR, alpha=0.6, linewidth=0.6, zorder=1)
            elif ragged:
                style = dict(color=RAGGED_COLOR, alpha=0.85, linewidth=1.3, zorder=3)
            else:
                style = dict(color=TRAJECTORY_COLOR, alpha=0.55, linewidth=0.9, zorder=2)
            ax.plot(ablation_grid, p_values, **style)
            if last is not None:
                # The filtration value: where this simplex enters the complex.
                ax.plot(ablation_grid[last], p_values[last], marker="o", markersize=4,
                        color="crimson", zorder=5)
                if ragged:
                    ax.plot(ablation_grid[first], p_values[first], marker="x",
                            markersize=5, color=RAGGED_COLOR, zorder=5)

        for method in METHODS:
            cut = cuts[method]
            if cut is None:
                continue
            ax.axhline(max(cut, FLOOR), linestyle=METHOD_STYLES[method],
                       linewidth=1.5, color=METHOD_COLORS[method], zorder=4)

        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("ablation $s$")
        ax.set_ylabel("p-value")
        admitted = sum(1 for _, _, p_vals in family if float(p_vals[-1]) <= decisive)
        title = f"dimension {dim}: {len(family)} candidates, {admitted} admitted"
        if n_ragged:
            title += f", {n_ragged} crossing back"
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        any_ragged = any_ragged or n_ragged > 0

    # Terms as in the article's caption. The entries for curves that cross back
    # over the cutoff appear only when the figure has such a curve. No title:
    # the caption says which network and input set a figure shows.
    handles = [
        Line2D([], [], color=REJECTED_COLOR, linewidth=1.0, label="rejected"),
        Line2D([], [], color=TRAJECTORY_COLOR, linewidth=1.2, label="admitted"),
    ]
    if any_ragged:
        handles.append(Line2D([], [], color=RAGGED_COLOR, linewidth=1.4,
                              label="admitted, crosses back over the cutoff"))
    handles.append(Line2D([], [], color="crimson", marker="o", markersize=4, linestyle="",
                          label=f"$s^*$, last crossing of the {METHOD_LABELS[DECISIVE]} cutoff"))
    if any_ragged:
        handles.append(Line2D([], [], color=RAGGED_COLOR, marker="x", markersize=5,
                              linestyle="", label="first crossing, where it differs"))
    for method in METHODS:
        if any(cutoffs[d][method] is not None for d in dims):
            handles.append(Line2D([], [], color=METHOD_COLORS[method],
                                  linestyle=METHOD_STYLES[method], linewidth=1.5,
                                  label=f"{METHOD_LABELS[method]} cutoff"))
    fig.legend(handles, [h.get_label() for h in handles], fontsize=8,
               loc="lower center", ncol=4, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, f"pvalue_trajectories_{stem}", outdir)


def plot_ablation_response(families, cutoffs, label, outdir, stem):
    """How significance accumulates with ablation, and where the filtration lands.

    The left panel is the fraction of candidates significant *at* ablation s, not
    by ablation s. The distinction is the whole point of the last-crossing rule:
    under the old reading a rejection was absorbing, so the curve could only
    rise; here a trajectory that crosses back pulls it down again, and any dip
    is an ablation at which the complex would have been different.

    The right panel is the distribution of last crossings -- the filtration
    values the complex actually carries -- with the first crossings drawn
    behind them. Where the two histograms coincide the ablation-response is
    monotone and 'minimum effective ablation' is exact; where the first crossings
    sit to the left, that gap is how much of the ablation axis the raggedness costs.
    """
    dims = list(families)
    fig, axes = plt.subplots(2, len(dims), figsize=(6.0 * len(dims), 8),
                             squeeze=False)

    summary = {}
    for column, dim in enumerate(dims):
        family = families[dim]
        cuts = cutoffs[dim]
        decisive = cuts[DECISIVE] if cuts[DECISIVE] is not None else -np.inf
        matrix = np.asarray([p_vals for _, _, p_vals in family], dtype=float)
        ablation_grid = np.linspace(0.0, 1.0, matrix.shape[1])

        ax_rate = axes[0][column]
        for method in METHODS:
            cut = cuts[method]
            if cut is None:
                continue
            ax_rate.plot(ablation_grid, (matrix <= cut).mean(axis=0), linewidth=1.6,
                         color=METHOD_COLORS[method],
                         linestyle=METHOD_STYLES[method],
                         label=METHOD_LABELS[method])
        ax_rate.set_xlim(0, 1)
        ax_rate.set_ylim(0, 1)
        ax_rate.set_xlabel("fraction ablated, $s$")
        ax_rate.set_ylabel(f"fraction significant at $s$ (of {len(family)})")
        ax_rate.set_title(f"dimension {dim}: significance against ablation", fontsize=10)
        ax_rate.grid(alpha=0.3)
        ax_rate.legend(fontsize=8, loc="upper left")

        first_ablations, last_ablations = [], []
        for _simplex, _ablation_grid, p_vals in family:
            first, last = crossings(p_vals, decisive)
            if last is None:
                continue
            span = max(1, len(p_vals) - 1)
            first_ablations.append(first / span)
            last_ablations.append(last / span)

        ax_hist = axes[1][column]
        if last_ablations:
            bins = np.linspace(0, 1, 41)
            ax_hist.hist(first_ablations, bins=bins, color=RAGGED_COLOR, alpha=0.55,
                         label="first crossing")
            ax_hist.hist(last_ablations, bins=bins, color="crimson", alpha=0.65,
                         label="last crossing (the filtration)")
            n_ragged = sum(1 for f, l in zip(first_ablations, last_ablations) if f != l)
            gaps = [l - f for f, l in zip(first_ablations, last_ablations)]
            summary[dim] = {
                "n_admitted": len(last_ablations),
                "n_ragged": n_ragged,
                "max_gap": max(gaps) if gaps else 0.0,
                "median_filtration": float(np.median(last_ablations)),
            }
            ax_hist.set_title(
                f"dimension {dim}: {n_ragged} of {len(last_ablations)} ragged, "
                f"largest gap {max(gaps):.3f}", fontsize=10)
            ax_hist.legend(fontsize=8)
        else:
            ax_hist.set_title(f"dimension {dim}: nothing admitted", fontsize=10)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel("ablation $s$ at the crossing")
        ax_hist.set_ylabel("candidates")
        ax_hist.grid(alpha=0.3)

    fig.suptitle(f"Ablation response and where the filtration lands — {label}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"pvalue_ablation_response_{stem}", outdir)
    return summary


def save(fig, name, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--decisive", default=DECISIVE, choices=METHODS,
                        help="which correction the last-crossing markers are "
                             "drawn for (default: the one the complexes use)")
    return parser.parse_args()


def main():
    args = parse_args()
    globals()["ALPHA"] = args.alpha
    globals()["DECISIVE"] = args.decisive
    caches = sorted(args.results.glob("pvalues_*.json"))
    if not caches:
        raise SystemExit(
            f"No p-value caches in {args.results}. Run "
            f"compute_pvalue_trajectories.py first."
        )

    for cache in caches:
        stem = cache.stem[len("pvalues_"):]
        match = re.fullmatch(r"(.+)_class_(\d+)", stem)
        if not match or match.group(1) not in MODEL_FACTORIES:
            print(f"\nskipping {cache.name}: no model factory for {stem}")
            continue
        architecture = match.group(1)
        label = re.sub(
            r"_class_(\d)",
            lambda m: f" / Q{int(m.group(1)) + 1}",
            stem.replace("_handcrafted", ""),
        )
        records = load_pvalues(str(cache))
        families = families_actually_posed(
            records, architecture, args.alpha, args.decisive
        )
        if not families:
            print(f"\n{label}: {METHOD_LABELS[args.decisive]} admits nothing at "
                  f"any dimension; no figure drawn")
            continue
        posed = sum(len(f) for f in families.values())
        print(f"\n{label}: {posed} candidates posed under "
              f"{METHOD_LABELS[args.decisive]} over {len(families)} dimensions "
              f"({len(records)} in the cache), "
              f"{len(records[0]['all_pvalues'])} ablation points")

        cutoffs = {d: cutoffs_for(f, args.alpha) for d, f in families.items()}
        for dim, cuts in cutoffs.items():
            shown = "  ".join(
                f"{METHOD_LABELS[m]}: "
                + ("rejects nothing" if cuts[m] is None else f"p <= {cuts[m]:.3g}")
                for m in METHODS
            )
            print(f"  dimension {dim} (m = {len(families[dim])}): {shown}")

        plot_trajectories(families, cutoffs, label, args.outdir, stem)
        summary = plot_ablation_response(families, cutoffs, label, args.outdir, stem)
        for dim, row in summary.items():
            print(f"  dimension {dim}: {row['n_admitted']} admitted, "
                  f"{row['n_ragged']} ragged, largest first-to-last gap "
                  f"{row['max_gap']:.3f}, median filtration "
                  f"{row['median_filtration']:.3f}")


if __name__ == "__main__":
    with RunTimer("p-value trajectory figures"):
        main()
