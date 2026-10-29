"""
Sensitivity of the DSC construction to the parameters of its hypothesis test.

Sweeps the constants that decide which simplices enter the complex --

  * epsilon (`relevance_margin`), which defines the null, H0: |theta| <= eps
  * alpha, the evidence threshold
  * the multiplicity correction over the family of simplices tested at each
    dimension ("none", Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli)
  * K (`max_iterations`), the resolution of the ablation grid

-- together with the probe count B, over both quadrant architectures and every
class context. They are swept jointly rather than in sequence because the first
three are not independent: epsilon is calibrated under a correction, and a
correction is chosen at some epsilon.

THE DECISION RULE THIS SCRIPT IS TESTING
----------------------------------------
Each candidate simplex carries a whole trajectory of tests, one per ablation
s in [0, 1], where s = 1 - sum_S a_i(gamma(s)) / sum_S a_i(x0) and s = 1 is the
fully ablated counterfactual. Two separate things are read off it:

  * Membership is decided at s = 1 alone. The family a correction applies to is
    therefore the set of simplices tested at one dimension -- not the ablation grid.
    K does not enter the correction at all.
  * The filtration value is the *last crossing*: the smallest ablation from which
    the test stays significant all the way to s = 1. This is the minimum
    effective ablation, and reading the ablation axis top-down like this is the
    fixed-sequence (step-down) MED procedure, which controls the familywise
    error over ablations with no further adjustment -- which is the formal reason
    the correction is applied over the simplex family only.

Both readings make predictions this script checks. Because every ablation grid
contains s = 1, membership must be exactly invariant in K; a K-dependence is a
bug, not a finding. And the last crossing is the smallest *grid* ablation from which
significance holds, so a coarser grid can only report it at or after where a
finer one does -- when the trajectory is monotone, which the ablation-response
diagnostic measures. Filtration values are therefore non-increasing in K, and
the shortfall is bounded by the coarse grid's spacing.

Every quantity is computed on the whole complex, not on its 1-skeleton, because
construction is inductive: losing one edge can remove many higher simplices, so
a margin that looks harmless at dimension 1 can flatten the complex at dimension
4. Neuron indices are fixed, so two complexes are compared by set equality on
their per-dimension simplex lists together with their filtration values.

Cost. The p-values a causal test produces depend on the probe set, the
trajectory and epsilon -- not on alpha or on the correction. One construction
per (context, epsilon, B) is therefore re-scored under every (alpha, correction,
K) triple via `build_dsc_from_cached_pvalues`. K comes along for free because a
coarser ablation grid is a subsample of a finer one, taken so that it still contains
the endpoint where the decision is made. The cache must be collected with
`fdr_correction="none"` at the largest alpha in the grid: construction being
inductive, a more permissive setting demands p-values a stricter one would never
have computed.

What is left after that is not spread evenly: eps = 0 admits enough at each
dimension to keep extending, so its constructions cost one to two orders of
magnitude more than the rest, and the run ends up bounded by its single longest
unit rather than by total work. Those units therefore get a second tier of
parallelism -- a few at a time, each construction parallel over the candidates
tested at its dimension -- while the cheap majority keeps one unit per worker.
Both tiers together occupy N_WORKERS cores; see `run_phase` in main().

Reported alongside the sweep:

  * The effect-size spectrum: the distribution of standardised effect
    |theta| / sigma over all (predictor, target) pairs, where theta is the median
    paired activation shift and sigma the target's within-class SD. Reported
    both at full ablation -- the quantity the decision rule actually sees -- and at
    its peak over the trajectory, whose gap against the terminal value is how
    far the coupling is from monotone. It depends only on the model and the
    probe set, so it presupposes no choice of epsilon, alpha or correction. Note
    that the median is not the test's own location functional -- see
    `null_admission_rate` and `relevance_test_pvalue`.
  * The ablation-response diagnostic: how often the first crossing differs from the
    last one. They coincide exactly when the p-value trajectory is monotone in
    ablation, which is what makes "minimum effective ablation" a well-posed reading of
    the filtration. A high disagreement rate says the ablation-response is ragged
    and the filtration is being set by where the raggedness stops rather than by
    where the effect begins.
  * The feasibility floor: the smallest B at which the one-sided signed-rank test
    can fire at all. Its smallest attainable p-value is 2^-B, so comparing that
    against the cutoff a correction allows bounds B from below whatever the
    effect size. It is a function of alpha and of how many simplices the
    dimension's family holds, which is what the correction divides alpha across.

Outputs go to `results/inference_sensitivity/` as JSON, alongside a stdout log.
No figures -- those are `plot_inference_sensitivity.py`.

USAGE:
  python compute_inference_sensitivity.py

To change run parameters, edit the CONFIGURATION block below. Completed work
units are checkpointed as they finish and reused on the next run.
"""

import os

# MUST run before numpy/scipy/torch are imported, here and in every spawned
# child (which re-imports this module top-to-bottom). OpenBLAS reads its thread
# count when the shared library loads, so setting these later is too late.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import itertools
import json
import multiprocessing as mp
import pickle
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from neural_dsc.utils.progress import RunTimer
from neural_dsc.directed_simplicial_complex import (
    DirectedSimplicialComplex,
    build_dsc_from_cached_pvalues,
    pin_worker_threads,
)
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.utils.fdr_correction import FAMILY_METHODS, persistent_crossing_index
from neural_dsc.utils.probe_selection import draw_probe_points, onehot_target, probe_seed
from neural_dsc.utils.jaccard_index import (
    JACCARD_FILTRATION_INTERVALS,
    JACCARD_FILTRATION_POINTS,
    JACCARD_METHOD_ID,
    filtration_jaccard_distance,
)

# Path setup
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# CONFIGURATION
OUTPUT_DIR = RESULTS_DIR / "inference_sensitivity"
CHECKPOINT = OUTPUT_DIR / "work_units.pkl"
LOG_FILE = OUTPUT_DIR / "inference_sensitivity.log"
RESUME = True

ARCHITECTURES = ["mono_handcrafted", "poly_handcrafted"]
CLASSES = [0, 1, 2, 3]

PROBE_COUNTS = [8, 16, 32, 64, 128, 256, 512]

# Each margin costs one full construction per (context, probe count), and the
# cost is a steep function of how small the margin is: at eps = 0 the point null
# admits everything the probe set can resolve, so the inductive construction has
# far more surviving faces to extend. Measured on this grid, eps = 0 cost 1.90
# CPU-h against ~0.71 CPU-h for every other margin -- 23% of the run in 10% of
# the units -- and its worst single unit ran 21.7 min against ~85 s for the same
# unit at eps = 0.025. That asymmetry is what the heavy tier below exists for.
#
# The grid is dense below 0.025 because that is where the answer changes: on the
# previous run every margin from 0.025 up collapsed every context to dimension 2,
# while eps = 0 reached dimension 5 -- so all of the structure, and all of the
# cost, lives in a gap the old grid stepped straight over.
MARGINS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4]

# Re-scored from the cached p-values, so these three axes are nearly free.
ALPHAS = [0.001, 0.01, 0.05]

# Resolution of the ablation grid. Not a horizon: the ablation reaches s = 1 by
# construction, so K only sets how finely the interval [0, 1] is sampled, and
# therefore how precisely the last crossing can be located. Because every grid
# contains the endpoint, membership must not depend on K at all -- that is an
# invariance this script checks rather than an axis it explores. The cache is
# built once at max(MAX_ITERATIONS_VALUES) and subsampled.
MAX_ITERATIONS_VALUES = [250, 500, 1000]

# Corrections over the family of simplices tested at one dimension. BY is the
# conservative default: simplices sharing a predictor set are scored from one
# shared trajectory, so their p-values are dependent in a way that is not
# obviously PRDS, and BY holds under arbitrary dependence.
FDR_METHODS = list(FAMILY_METHODS)

# Family sizes to tabulate the feasibility floor against, spanning the range a
# dimension of these complexes actually reaches. The observed sizes are reported
# with the results, from the cache.
FAMILY_SIZES_FOR_FLOOR = [1, 10, 100, 1000]

MAX_DIMENSIONS = {"mono_handcrafted": 8, "poly_handcrafted": 7}  # architecture ceilings
MAX_ITERATIONS = max(MAX_ITERATIONS_VALUES)  # what is actually integrated
DT = 0.01
RELEVANCE_SCALE = "std"
PROBE_TOLERANCE = 0.01
SEED = 20260819
N_WORKERS = -1  # -1 or None = all CPUs but four, floored at 1

# Second tier of parallelism, for the units that would otherwise set the wall
# clock on their own. A unit counts as heavy when its margin is at or below
# HEAVY_MARGIN and its probe count at or above HEAVY_PROBES; heavy units are
# dispatched HEAVY_LANES at a time, each construction parallel over the
# candidates tested at its dimension, so the two tiers together still occupy
# N_WORKERS cores. See `run_phase` in main() for why this is worth the trouble.
#
# The rule is a cost predictor, not a property of the result: it selects
# eps = 0 at B >= 64, which on the previous run was 32 of the 560 units and 1.69
# of the 1.90 CPU-h that eps = 0 cost. Widening it past the genuinely long units
# only pays inner-pool startup on units that finish in seconds.
HEAVY_MARGIN = 0.0
HEAVY_PROBES = 64
HEAVY_LANES = 4

# Laptop-sized override for a smoke run.
QUICK = False
if QUICK:
    ARCHITECTURES = ["mono_handcrafted"]
    CLASSES = [0]
    PROBE_COUNTS = [16, 32]
    MARGINS = [0.1, 0.2]
    ALPHAS = [0.01]
    MAX_ITERATIONS_VALUES = [25, 50]
    MAX_ITERATIONS = max(MAX_ITERATIONS_VALUES)

MODEL_FACTORIES = {
    "mono_handcrafted": create_quadrants_mono,
    "poly_handcrafted": create_quadrants_poly,
}


def is_heavy(unit):
    """
    Whether a unit is long enough to be worth the second tier of parallelism.

    A prediction from the two axes cost actually depends on, made before the
    unit runs. Nothing about the result depends on it, so it is deliberately
    absent from `config_key`: retiering does not invalidate a checkpoint.
    """
    _, _, n_probes, margin = unit
    return margin <= HEAVY_MARGIN and n_probes >= HEAVY_PROBES


def feasibility_floor(alpha, family_size, fdr_method):
    """
    Smallest probe count at which the one-sided signed-rank test can fire at all.

    Its smallest attainable p-value is 2^-B; compare that against the most
    permissive cutoff the correction can produce on a family of `family_size`
    tests. "none" and Benjamini-Hochberg both allow alpha itself at the top
    rank; Bonferroni allows alpha / m; BY allows alpha / (m * H_m) only in its
    worst case, but at the top rank it allows alpha / H_m, which is the bound
    that matters for feasibility.

    The ablation resolution K does not enter. Each simplex contributes one test,
    taken at full ablation, so nothing divides alpha across the trajectory.
    """
    m = max(1, int(family_size))
    if fdr_method in ("none", "bh"):
        threshold = alpha
    elif fdr_method == "bonferroni":
        threshold = alpha / m
    elif fdr_method == "by":
        threshold = alpha / float(np.sum(1.0 / np.arange(1, m + 1)))
    else:
        raise ValueError(f"unknown correction {fdr_method!r}")
    b = 1
    while 2.0 ** -b > threshold:
        b += 1
    return b, threshold


def build_probes(architecture, class_idx, n_probes):
    """Deterministically rebuild a model and its probe set inside a worker."""
    model = MODEL_FACTORIES[architecture]()
    model.eval()
    result = draw_probe_points(
        model,
        onehot_target(4, class_idx),
        n_probes,
        tolerance=PROBE_TOLERANCE,
        seed=probe_seed(SEED, class_idx, n_probes),
        label=f"{architecture}/class_{class_idx} n={n_probes}",
        verbose=False,
    )
    return model, result


def as_simplex_map(complex_dict):
    """{dimension: {simplex tuple: filtration}}, the comparable form."""
    out = {}
    for dim, entries in complex_dict.items():
        if dim == 0 or not entries:
            continue
        out[int(dim)] = {tuple(s): float(f) for s, f in entries}
    return out


def run_work_unit(unit, inner_workers=1):
    """
    One (architecture, class, probe count, margin) cell.

    Owns exactly one full construction -- the expensive part -- and re-scores it
    under every (alpha, correction, trajectory length) triple from the cached
    p-values.

    `inner_workers` is the second tier: with more than one, the construction
    parallelises over the candidates tested at each dimension rather than
    running them in sequence. It needs a non-daemonic parent, which is why the
    scheduler dispatches through ProcessPoolExecutor.
    """
    pin_worker_threads()
    architecture, class_idx, n_probes, margin = unit

    # mp.Pool workers are daemonic and cannot have children, and build() catches
    # the resulting pool-creation failure and quietly falls back to sequential.
    # That failure mode is invisible in the log and costs a factor of two, so
    # refuse it here instead of discovering it in the wall clock.
    if inner_workers > 1 and mp.current_process().daemon:
        raise RuntimeError(
            f"cannot use {inner_workers} inner workers from a daemonic process: "
            "dispatch the heavy tier with ProcessPoolExecutor, not mp.Pool"
        )

    model, probe_result = build_probes(architecture, class_idx, n_probes)
    max_dimension = MAX_DIMENSIONS[architecture]

    started = time.time()
    sc = DirectedSimplicialComplex()
    # Collected at the most permissive setting so the cache is a superset of
    # every corrected complex; see build_dsc_from_cached_pvalues.
    result = sc.build(
        model=model,
        probe_data=probe_result.points,
        max_dimension=max_dimension,
        dt=DT,
        max_iterations=MAX_ITERATIONS,
        significance_level=max(ALPHAS),
        n_workers=inner_workers,
        fdr_correction="none",
        collect_pvalues=True,
        relevance_margin=margin,
        relevance_scale=RELEVANCE_SCALE,
    )
    sc, pvalues = result if isinstance(result, tuple) else (result, [])
    build_seconds = time.time() - started

    complexes = {}
    for alpha, fdr_method, n_iter in itertools.product(
        ALPHAS, FDR_METHODS, MAX_ITERATIONS_VALUES
    ):
        if fdr_method == "none" and alpha == max(ALPHAS) and n_iter == MAX_ITERATIONS:
            rebuilt = sc  # already exactly this configuration
        else:
            rebuilt = build_dsc_from_cached_pvalues(
                pvalues, model, max_dimension, DT, alpha, fdr_method,
                n_steps=n_iter + 1, verbose=False,
            )
        complexes[(alpha, fdr_method, n_iter)] = as_simplex_map(rebuilt.complex)

    return (
        architecture, class_idx, n_probes, margin, complexes,
        {
            "ablation_response": ablation_response_diagnostic(pvalues, ALPHAS),
            "family_sizes": {
                int(d): sum(1 for r in pvalues if r["dimension"] == d)
                for d in sorted({r["dimension"] for r in pvalues})
            },
            "build_seconds": build_seconds,
            "n_pvalue_records": len(pvalues),
            "min_purity": probe_result.diagnostics["min_purity"],
            "acceptance_rate": probe_result.acceptance_rate,
        },
    )


def ablation_response_diagnostic(pvalues, alphas):
    """
    How far the p-value trajectories are from monotone in ablation.

    The filtration is the last crossing, and calling that a *minimum effective
    ablation* is only honest if the test, once significant, stays significant. When
    it does, the first crossing and the last coincide. When it does not, the
    filtration is set by where the trajectory stops flickering rather than by
    where the effect appears, and the gap between the two is how much of the
    ablation axis that costs.

    Scored at the uncorrected alpha, which is the most permissive cutoff any
    correction in the grid can produce and therefore exercises the most of each
    trajectory; a stricter cutoff can only shorten the significant region.

    Returns {alpha: {counts and ablation gaps}}, over every (predictor, target) the
    construction actually tested.
    """
    out = {}
    for alpha in alphas:
        n_significant = 0
        n_ragged = 0
        gaps = []
        for record in pvalues:
            p_vals = np.asarray(record["all_pvalues"], dtype=float)
            if p_vals.size == 0:
                continue
            flags = p_vals <= alpha
            last = persistent_crossing_index(flags)
            if last is None:
                continue  # never admitted, so it has no filtration to argue about
            n_significant += 1
            first = int(np.argmax(flags))
            if first != last:
                n_ragged += 1
                span = max(1, p_vals.size - 1)
                gaps.append((last - first) / span)
        out[str(alpha)] = {
            "n_significant_at_full_ablation": n_significant,
            "n_ragged": n_ragged,
            "ragged_fraction": n_ragged / n_significant if n_significant else float("nan"),
            "mean_ablation_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_ablation_gap": float(np.max(gaps)) if gaps else 0.0,
        }
    return out


def run_spectrum_unit(unit):
    """
    One (architecture, class, probe count) cell of the effect-size spectrum.

    Kept separate from the complex-building units because the spectrum depends
    only on the model and the probe set, so it would otherwise be recomputed
    identically for every margin. Restricted to single-neuron predictors: the
    spectrum is a statement about pairwise coupling.
    """
    pin_worker_threads()
    architecture, class_idx, n_probes = unit

    model, probe_result = build_probes(architecture, class_idx, n_probes)
    baseline = model.get_activations(probe_result.points)
    baseline_np = baseline.cpu().numpy()

    effects = {"terminal": [], "peak": []}
    by_pair = {}
    n_zero_sigma = 0
    for predictor in range(model.internal_neurons):
        # The same CCA trajectory the complexes are built from, so the spectrum
        # describes the intervention the decision rule actually sees.
        acts_over_time, _ = model.ablation_trajectory(
            probe_result.points, [predictor], n_steps=MAX_ITERATIONS, dt=DT
        )
        for target in range(model.internal_neurons):
            if target == predictor:
                continue
            sigma = float(np.std(baseline_np[:, target]))
            if sigma <= 0:
                # Excluded rather than counted as zero: such a pair can still
                # become a simplex via the epsilon floor, so the spectrum is not
                # a complete census and says so.
                n_zero_sigma += 1
                continue
            y_obs = baseline[:, target]
            shifts = [
                abs(float(np.median((y_obs - acts_over_time[step][:, target]).cpu().numpy())))
                for step in range(acts_over_time.shape[0])
            ]
            # Terminal is the decision point: membership is settled at full ablation
            # alone. Peak is kept beside it because their ratio is a
            # model-level, test-free measure of how non-monotone the coupling
            # is -- a pair with a large peak and a terminal near zero is one the
            # rule is designed to reject, and it is worth knowing how many there
            # are.
            terminal = shifts[-1] / sigma
            peak = max(shifts) / sigma
            effects["terminal"].append(terminal)
            effects["peak"].append(peak)
            # Keyed by the 1-simplex it would become, so an admitted edge can be
            # checked against the null region it was supposed to exclude.
            by_pair[(predictor, target)] = {"terminal": terminal, "peak": peak}

    return architecture, class_idx, n_probes, effects, n_zero_sigma, by_pair


def report_effect_spectrum(effects, n_zero_sigma):
    """Print the |theta|/sigma percentiles at full ablation and at the peak."""
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    out = {}
    for which in ("terminal", "peak"):
        arr = np.array(sorted(e for e in effects[which] if np.isfinite(e)))
        if arr.size == 0:
            print(f"    ({which}: no finite effect sizes)")
            continue
        pct = {str(q): float(np.percentile(arr, q)) for q in qs}
        out[which] = pct
        label = "at s=1" if which == "terminal" else " peak "
        print(f"    |theta|/sigma {label}: "
              + "  ".join(f"p{q}={pct[str(q)]:.3f}" for q in qs))
    if n_zero_sigma:
        print(f"    ({n_zero_sigma} pairs excluded: target has zero within-class SD, "
              f"so no standardised effect is defined -- they can still become "
              f"simplices via the epsilon floor)")
    return out


def score_series(by_count):
    """
    Turn one (context, alpha, correction, margin) series over probe counts into
    complex-level stability numbers.

    `by_count` maps probe count -> {dimension: {simplex: filtration}}.

    Everything is reported per dimension, since the margin's effect compounds
    upward through the inductive construction.
    """
    counts = sorted(by_count)
    largest = counts[-1]
    dims = sorted({d for m in by_count.values() for d in m})

    per_dim = {}
    for dim in dims:
        filtrations = {b: by_count[b].get(dim, {}) for b in counts}
        sets = {b: set(filtrations[b]) for b in counts}
        upper = counts[len(counts) // 2:]
        stable = set.intersection(*(sets[b] for b in upper)) if upper else set()

        means = []
        for b in upper:
            vals = [by_count[b][dim][s] for s in stable if s in by_count[b].get(dim, {})]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        finite = [m for m in means if m == m and m > 0]

        per_dim[dim] = {
            "counts": {b: len(sets[b]) for b in counts},
            "jaccard_distance_min_vs_max": filtration_jaccard_distance(
                filtrations[counts[0]], filtrations[largest]
            ),
            "jaccard_distance_prev_vs_max": (
                filtration_jaccard_distance(
                    filtrations[counts[-2]], filtrations[largest]
                )
                if len(counts) > 1
                else float("nan")
            ),
            "n_stable": len(stable),
            "mean_filtration_by_count": dict(zip(upper, means)),
            "filtration_drift": max(finite) / min(finite) if len(finite) > 1 else float("nan"),
            "min_probes_matching_largest": next(
                (b for b in counts if sets[b] and sets[b] == sets[largest]), None
            ),
        }

    total_by_count = {
        b: sum(len(by_count[b].get(d, {})) for d in dims) for b in counts
    }
    top_dim_by_count = {
        b: max((d for d in dims if by_count[b].get(d)), default=0) for b in counts
    }
    return {
        "per_dimension": per_dim,
        "total_simplices_by_count": total_by_count,
        "top_dimension_by_count": top_dim_by_count,
        # Support convergence remains a set-membership check; the Jaccard
        # distance above additionally detects filtration drift.
        "converged_in_probe_count": all(
            set(by_count[counts[-2]].get(d, {}))
            == set(by_count[largest].get(d, {}))
            for d in dims
        ) if dims else False,
    }


def null_admission_rate(simplex_map, effects_by_pair, margin):
    """
    Fraction of admitted 1-simplices that lie inside the null region.

    In standardised units the interval null H0: |theta| <= eps is exactly
    |theta| / sigma <= margin, because eps = margin * sigma by construction, so
    an edge whose standardised effect sits at or below the margin looks like a
    false discovery.

    Scored at full ablation, not at the peak, because that is where membership is
    decided. An edge with a large peak and a terminal effect inside the null
    region is precisely what the decision rule is meant to exclude, so counting
    it as a discovery would score the rule against a hypothesis it does not
    test.

    Read it as an upper bound on those, not a count. The spectrum standardises
    the *median* paired shift, while the signed-rank test estimates the
    pseudomedian; on the zero-inflated deltas that ablation produces the two
    diverge, and an edge can sit at median zero while the test correctly rejects
    (see `relevance_test_pvalue`). A high rate therefore flags a distribution
    whose effect is concentrated on a subset of probe points, which is a
    substantive property of the coupling rather than an error. It is also a
    plug-in estimate: the peak is measured on the same probe set, so a genuinely
    borderline pair can land on either side by chance. It remains comparable
    across corrections, which are all scored against the same estimates.

    Restricted to dimension 1, since the effect spectrum is computed for
    single-neuron predictors and higher simplices have no standardised effect to
    compare against.

    Returns (rate, n_null_admitted, n_admitted), or (nan, 0, 0) if nothing was
    admitted or the margin is zero (under which the null region is empty).
    """
    edges = simplex_map.get(1, {})
    if not edges or margin <= 0:
        return float("nan"), 0, len(edges)
    checked = [
        effects_by_pair[e]["terminal"] for e in edges
        if e in effects_by_pair
    ]
    if not checked:
        return float("nan"), 0, len(edges)
    n_null = sum(1 for v in checked if v <= margin)
    return n_null / len(checked), n_null, len(checked)


def compare_trajectory_lengths(by_length):
    """
    Effect of the ablation-grid resolution at one (context, margin, alpha, correction, B).

    `by_length` maps K -> {dimension: {simplex: filtration}}.

    K is no longer a parameter of the decision. Membership is settled at s = 1,
    every grid contains s = 1, so the simplex sets must agree exactly across K.
    `membership_invariant` records whether they did; a False here is a bug in the
    subsampling or in the decision rule, not a property of the data, and the
    offending dimensions are listed so it can be chased.

    What K does control is where the last crossing can be located. It is the
    smallest *grid* ablation from which the test stays significant, so on a coarser
    grid it lands on the first available point at or after the true crossing:
    for a monotone trajectory a coarse grid reports the same value or a later
    one, short by less than one grid spacing. `filtration_monotone_in_K` checks
    that the values are non-increasing in K and `resolution_bound_respected`
    that the shortfall stays inside the coarse spacing. Both can legitimately
    fail on a *non*-monotone trajectory, where coarsening can also skip the dip
    that was holding the crossing back and hand back an earlier value -- so read
    a violation here against the ablation-response table before calling it a bug.
    """
    lengths = sorted(by_length)
    longest = lengths[-1]
    dims = sorted({d for m in by_length.values() for d in m})

    per_dim = {}
    membership_violations = []
    monotonicity_violations = []
    bound_violations = []
    shifts = []
    for dim in dims:
        filtrations = {k: by_length[k].get(dim, {}) for k in lengths}
        sets = {k: set(filtrations[k]) for k in lengths}
        ref = sets[longest]
        if any(sets[k] != ref for k in lengths):
            membership_violations.append(
                {"dimension": dim,
                 "counts": {k: len(sets[k]) for k in lengths},
                 "n_symmetric_difference": {k: len(sets[k] ^ ref) for k in lengths}}
            )
        # Filtration comparison over the simplices every K agrees on, which is
        # all of them unless membership is already broken.
        common = set.intersection(*(sets[k] for k in lengths)) if lengths else set()
        dim_shift = 0.0
        for simplex in common:
            values = [by_length[k][dim][simplex] for k in lengths]
            # Non-increasing in K: the coarse grid reports the crossing at or
            # after the fine grid does.
            if any(a < b - 1e-9 for a, b in zip(values, values[1:])):
                monotonicity_violations.append(
                    {"dimension": dim, "simplex": simplex,
                     "filtration_by_length": dict(zip(lengths, values))}
                )
            # ... and by less than one spacing of whichever grid is coarser.
            for k, value in zip(lengths, values):
                if value - values[-1] > 1.0 / k + 1e-9:
                    bound_violations.append(
                        {"dimension": dim, "simplex": simplex, "K": k,
                         "excess": value - values[-1] - 1.0 / k}
                    )
            dim_shift = max(dim_shift, abs(values[-1] - values[0]))
        shifts.append(dim_shift)
        per_dim[dim] = {
            "counts": {k: len(sets[k]) for k in lengths},
            "jaccard_distance_to_longest": {
                k: filtration_jaccard_distance(
                    filtrations[k], filtrations[longest]
                )
                for k in lengths
            },
            "n_common": len(common),
            "max_filtration_shift": dim_shift,
        }

    totals = [sum(len(by_length[k].get(d, {})) for d in dims) for k in lengths]
    return {
        "per_dimension": per_dim,
        "total_by_length": dict(zip(lengths, totals)),
        "membership_invariant": not membership_violations,
        "membership_violations": membership_violations,
        "filtration_monotone_in_K": not monotonicity_violations,
        # Capped: a systematic failure would otherwise write out every simplex.
        "monotonicity_violations": monotonicity_violations[:20],
        "n_monotonicity_violations": len(monotonicity_violations),
        "resolution_bound_respected": not bound_violations,
        "n_resolution_bound_violations": len(bound_violations),
        "max_filtration_shift": max(shifts) if shifts else 0.0,
    }


def compare_corrections(by_correction):
    """
    Pairwise agreement between corrections at one (context, margin, alpha, B).

    `by_correction` maps correction name -> {dimension: {simplex: filtration}}.

    Reports two things. Per-dimension Jaccard distance, i.e. how much the choice of
    correction changes the complex at all; and containment, checking the
    structural expectation that a stricter correction admits a subset of a more
    permissive one. A containment violation indicates a bug rather than a
    finding, so it is surfaced rather than averaged away.
    """
    names = list(by_correction)
    dims = sorted({d for m in by_correction.values() for d in m})
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            per_dim = {}
            for dim in dims:
                fa = by_correction[a].get(dim, {})
                fb = by_correction[b].get(dim, {})
                sa = set(fa)
                sb = set(fb)
                per_dim[dim] = {
                    "jaccard_distance": filtration_jaccard_distance(fa, fb),
                    "only_in_first": len(sa - sb),
                    "only_in_second": len(sb - sa),
                }
            pairwise[f"{a}|{b}"] = per_dim

    # "none" is the most permissive by construction; everything must sit inside it.
    violations = []
    if "none" in by_correction:
        for name, simplex_map in by_correction.items():
            if name == "none":
                continue
            for dim in dims:
                extra = set(simplex_map.get(dim, {})) - set(by_correction["none"].get(dim, {}))
                if extra:
                    violations.append(
                        {"correction": name, "dimension": dim, "n_extra": len(extra)}
                    )
    return {"pairwise": pairwise, "containment_violations": violations}


class _Tee:
    """Mirror stdout into a log file, so tables survive a lost ssh session."""

    def __init__(self, stream, path):
        self.stream = stream
        self.log = open(path, "a", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return self.stream.isatty()


def main(run):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, LOG_FILE)
    print(f"\n{'#' * 78}\n# run started {datetime.now().isoformat(timespec='seconds')}\n{'#' * 78}")
    started = time.time()
    torch.manual_seed(SEED)

    config_key = (
        "v5-full-ablation-decision",  # bump when a work unit's return shape changes
        tuple(ARCHITECTURES), tuple(CLASSES), tuple(PROBE_COUNTS), tuple(MARGINS),
        tuple(ALPHAS), tuple(FDR_METHODS), tuple(MAX_ITERATIONS_VALUES),
        tuple(sorted(MAX_DIMENSIONS.items())),
        MAX_ITERATIONS, DT, RELEVANCE_SCALE, PROBE_TOLERANCE, SEED,
    )

    n_workers = N_WORKERS
    if n_workers is None or n_workers == -1:
        n_workers = max(1, mp.cpu_count() - 4)
    soft_nproc, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"Workers: {n_workers} of {mp.cpu_count()} CPUs | RLIMIT_NPROC soft = {soft_nproc} "
          f"| BLAS threads pinned to {os.environ.get('OPENBLAS_NUM_THREADS')}")

    print("\nSigned-rank feasibility floors: the smallest probe count at which a "
          "test can fire\nat all, since its smallest attainable p-value is 2^-B. "
          "It depends on alpha and on\nhow many simplices the dimension's family "
          "holds, which is what alpha is divided\nacross:")
    print(f"  {'correction':<14}{'family m':>9}" + "".join(f"{f'a={a}':>16}" for a in ALPHAS))
    floors = {}
    for method in FDR_METHODS:
        for m in FAMILY_SIZES_FOR_FLOOR:
            cells = []
            for alpha in ALPHAS:
                b, thr = feasibility_floor(alpha, m, method)
                floors[(method, m, alpha)] = b
                cells.append(f"B>={b} ({thr:.1e})")
            print(f"  {method:<14}{m:>9}" + "".join(f"{c:>16}" for c in cells))
    worst_floor = max(floors.values())
    below = [b for b in PROBE_COUNTS if b < worst_floor]
    if below:
        print(f"  NOTE: probe counts {below} sit below the worst floor B>={worst_floor} "
              f"and cannot fire\n        under every (alpha, correction, family size) "
              f"in this grid, whatever the\n        effect size. Observed family sizes "
              f"are reported with the results.")

    # Longest-processing-time-first, which is what the second tier needs too:
    # cost rises with the probe count and falls with the margin, in that order
    # of influence, so the largest B and the smallest margins go out first and
    # the long poles are already running while workers are still free.
    units = [
        (arch, cls, n, margin)
        for n in sorted(PROBE_COUNTS, reverse=True)
        for margin in sorted(MARGINS)
        for arch in ARCHITECTURES
        for cls in CLASSES
    ]
    n_heavy = sum(1 for u in units if is_heavy(u))
    print(f"\n{len(units)} work units (one full construction each), dispatched "
          f"most-expensive-first.\nCost is dominated by the probe count and by how "
          f"small the margin is, and spans three orders of\nmagnitude across the "
          f"grid, so the reported ETA runs pessimistic until the long poles clear.\n"
          f"{n_heavy} of them (margin <= {HEAVY_MARGIN:g}, B >= {HEAVY_PROBES}) run in "
          f"the heavy tier: {max(1, min(HEAVY_LANES, n_workers))} at a time, each "
          f"construction internally\nparallel, since a single unit of that size "
          f"otherwise sets the wall clock on its own.")

    # The spectrum depends only on the model and probe set, so it is one unit
    # per (context, probe count) rather than one per margin.
    spectrum_units = [
        (arch, cls, n)
        for arch in ARCHITECTURES for cls in CLASSES for n in PROBE_COUNTS
    ]

    results = []
    spectrum_results = []
    done = set()
    spectrum_done = set()
    if RESUME and CHECKPOINT.exists():
        with open(CHECKPOINT, "rb") as f:
            cached = pickle.load(f)
        if cached.get("config_key") == config_key:
            results = cached["results"]
            spectrum_results = cached.get("spectrum_results", [])
            done = {(r[0], r[1], r[2], r[3]) for r in results}
            spectrum_done = {(r[0], r[1], r[2]) for r in spectrum_results}
            print(f"  Resuming: {len(done)}/{len(units)} complex units and "
                  f"{len(spectrum_done)}/{len(spectrum_units)} spectrum units already done.")
        else:
            print(f"  Ignoring {CHECKPOINT.name}: written under a different configuration.")
    pending = [u for u in units if (u[0], u[1], u[2], u[3]) not in done]
    pending_spectrum = [u for u in spectrum_units if u not in spectrum_done]

    def checkpoint():
        tmp = CHECKPOINT.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({
                "config_key": config_key,
                "results": results,
                "spectrum_results": spectrum_results,
            }, f)
        tmp.replace(CHECKPOINT)  # atomic, so a kill mid-write cannot corrupt it

    # Spectrum first: it is cheap, and it is the one result that presupposes no
    # choice of eps, alpha or correction, so it is worth having even if the
    # expensive phase is interrupted.
    if pending_spectrum:
        run.section(f"effect-size spectrum ({len(pending_spectrum)} units)")
        if n_workers > 1 and len(pending_spectrum) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers, initializer=pin_worker_threads) as pool:
                for res in pool.imap_unordered(run_spectrum_unit, pending_spectrum):
                    spectrum_results.append(res)
        else:
            for unit in pending_spectrum:
                spectrum_results.append(run_spectrum_unit(unit))
        checkpoint()
        print(f"  done ({time.time() - started:.0f}s elapsed)")

    # Two tiers, because one unit per worker cannot beat the longest unit and on
    # the previous run that unit *was* the wall clock: 8.3 CPU-h over 48 lanes
    # is ten minutes of work, but one eps = 0 construction ran 21.7 minutes on
    # its own with every other lane idle behind it. So the heavy units go a few
    # at a time with the construction parallel over the candidates at each
    # dimension, and everything else keeps the one-unit-per-worker shape.
    #
    # Inner width is capped by the family size at a dimension, not by the unit's
    # total cost -- dimensions are built in sequence, and the worst unit tests
    # only 42/98/92/48 candidates across dimensions 1-4. Past about 16 the tail
    # dimension runs out of candidates to spread. Scheduling the recorded times
    # says 4 lanes x 12 is flat-optimal at ~11 min against the 10.3 min floor,
    # and that both extremes lose: 48x1 is today's 22 min, while 1x48 is 13 min
    # because most eps = 0 units are small and serialising them wastes lanes.
    heavy = [u for u in pending if is_heavy(u)]
    light = [u for u in pending if not is_heavy(u)]
    heavy_lanes = max(1, min(HEAVY_LANES, n_workers))
    inner_workers = max(1, n_workers // heavy_lanes)

    def run_phase(label, phase_units, lanes, inner):
        """
        Dispatch one tier.

        ProcessPoolExecutor rather than mp.Pool: its workers are not daemonic,
        so a unit may open a pool of its own. Submission order is dispatch
        order, so the longest-first ordering survives.
        """
        if not phase_units:
            return
        run.section(
            f"{label} ({len(phase_units)} units, {lanes} lanes"
            + (f" x {inner} inner workers)" if inner > 1 else ", one build each)")
        )
        every = max(1, len(phase_units) // 20)
        if lanes > 1 and len(phase_units) > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=lanes, mp_context=ctx, initializer=pin_worker_threads
            ) as pool:
                futures = [pool.submit(run_work_unit, u, inner) for u in phase_units]
                for i, future in enumerate(as_completed(futures), 1):
                    results.append(future.result())
                    if i % every == 0 or i == len(phase_units):
                        checkpoint()
                        run.eta(i, len(phase_units), "units")
        else:
            for i, unit in enumerate(phase_units, 1):
                results.append(run_work_unit(unit, inner))
                if i % every == 0 or i == len(phase_units):
                    checkpoint()
                    run.eta(i, len(phase_units), "units")
        checkpoint()

    run_phase("complex construction, heavy tier", heavy, heavy_lanes, inner_workers)
    run_phase("complex construction", light, n_workers, 1)

    run.section("scoring + reporting")

    # (context, alpha, fdr, margin, K) -> probe count -> {dim: {simplex: filtration}}
    collected = {}
    build_times = {}
    ablation_response = {}
    family_sizes = {}
    for arch, cls, n_probes, margin, complexes, meta in results:
        build_times[(arch, cls, n_probes, margin)] = meta["build_seconds"]
        ablation_response[(arch, cls, n_probes, margin)] = meta.get("ablation_response", {})
        for dim, count in meta.get("family_sizes", {}).items():
            family_sizes[int(dim)] = max(family_sizes.get(int(dim), 0), int(count))
        for (alpha, fdr_method, n_iter), simplex_map in complexes.items():
            collected.setdefault(
                (arch, cls, alpha, fdr_method, margin, n_iter), {}
            )[n_probes] = simplex_map

    # (arch, class, probe count) -> {(predictor, target): peak |theta|/sigma}
    effects_by_pair = {
        (r[0], r[1], r[2]): r[5] for r in spectrum_results
    }

    summary = {}
    spectra = {}
    fdr_comparison = {}
    for arch in ARCHITECTURES:
        for cls in CLASSES:
            label = f"{arch}/class_{cls}"
            print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

            print("\n  Effect-size spectrum (independent of margin, alpha and correction):")
            spectra[label] = {}
            for n_probes in PROBE_COUNTS:
                match = [r for r in spectrum_results
                         if r[0] == arch and r[1] == cls and r[2] == n_probes]
                if not match:
                    continue
                _, _, _, effects, n_zero, _pairs = match[0]
                print(f"  B={n_probes}")
                spectra[label][str(n_probes)] = report_effect_spectrum(effects, n_zero)

            for fdr_method in FDR_METHODS:
                for alpha in ALPHAS:
                    print(f"\n  correction={fdr_method}  alpha={alpha}  K={MAX_ITERATIONS}")
                    print(f"  {'margin':>7} | {'simplices by dim at B=' + str(max(PROBE_COUNTS)):<34}"
                          f"| {'top':>4} | {'conv':>5} | {'dJ1':>5} | {'drift1':>7} | {'null/n':>7}")
                    for margin in MARGINS:
                        key = (arch, cls, alpha, fdr_method, margin, MAX_ITERATIONS)
                        if key not in collected:
                            continue
                        row = score_series(collected[key])
                        biggest = max(PROBE_COUNTS)
                        dims = sorted(row["per_dimension"])
                        shape = " ".join(
                            f"d{d}:{row['per_dimension'][d]['counts'][biggest]}"
                            for d in dims
                            if row["per_dimension"][d]["counts"][biggest]
                        ) or "(empty)"
                        d1 = row["per_dimension"].get(1, {})
                        rate, n_null, n_adm = null_admission_rate(
                            collected[key][biggest],
                            effects_by_pair.get((arch, cls, biggest), {}),
                            margin,
                        )
                        row["null_admission"] = {
                            "rate": rate, "n_null_admitted": n_null, "n_checked": n_adm,
                        }
                        summary[str(key)] = row
                        print(f"  {margin:>7} | {shape:<34}| "
                              f"{row['top_dimension_by_count'][biggest]:>4} | "
                              f"{str(row['converged_in_probe_count']):>5} | "
                              f"{d1.get('jaccard_distance_min_vs_max', float('nan')):>5.2f} | "
                              f"{d1.get('filtration_drift', float('nan')):>7.2f} | "
                              f"{(f'{n_null}/{n_adm}' if n_adm else '-'):>7}")

    # How much the choice of correction actually changes the complex.
    print(f"\n{'=' * 78}\nCORRECTION AGREEMENT "
          f"(at B={max(PROBE_COUNTS)}, alpha={max(ALPHAS)}, K={MAX_ITERATIONS})\n{'=' * 78}")
    for arch in ARCHITECTURES:
        for cls in CLASSES:
            label = f"{arch}/class_{cls}"
            for margin in MARGINS:
                by_correction = {
                    m: collected[
                        (arch, cls, max(ALPHAS), m, margin, MAX_ITERATIONS)
                    ][max(PROBE_COUNTS)]
                    for m in FDR_METHODS
                    if (arch, cls, max(ALPHAS), m, margin, MAX_ITERATIONS) in collected
                }
                if len(by_correction) < 2:
                    continue
                cmp = compare_corrections(by_correction)
                fdr_comparison[f"{label}|margin={margin}"] = cmp
                dims = sorted({d for m in by_correction.values() for d in m})
                worst = max(
                    (per_dim[d]["jaccard_distance"]
                     for per_dim in cmp["pairwise"].values()
                     for d in dims
                     if per_dim[d]["jaccard_distance"]
                     == per_dim[d]["jaccard_distance"]),
                    default=float("nan"),
                )
                flag = ""
                if cmp["containment_violations"]:
                    flag = (f"  !! {len(cmp['containment_violations'])} containment "
                            f"violations (a stricter correction admitted a simplex "
                            f"'none' did not -- this is a bug, not a finding)")
                print(f"  {label:<28} margin={margin:<6} "
                      f"largest pairwise Jaccard distance across dims = {worst:.2f}{flag}")

    # K sets the resolution of the ablation grid, nothing else. Membership must be
    # exactly invariant in it; only the filtration values may move, and only
    # earlier as the grid coarsens.
    print(f"\n{'=' * 78}\nABLATION-GRID RESOLUTION "
          f"(at B={max(PROBE_COUNTS)}, alpha={max(ALPHAS)})\n{'=' * 78}")
    length_comparison = {}
    n_membership_broken = 0
    n_monotonicity_broken = 0
    n_bound_broken = 0
    worst_shift = 0.0
    for arch in ARCHITECTURES:
        for cls in CLASSES:
            label = f"{arch}/class_{cls}"
            for margin in MARGINS:
                header_done = False
                for fdr_method in FDR_METHODS:
                    by_length = {
                        k: collected[
                            (arch, cls, max(ALPHAS), fdr_method, margin, k)
                        ][max(PROBE_COUNTS)]
                        for k in MAX_ITERATIONS_VALUES
                        if (arch, cls, max(ALPHAS), fdr_method, margin, k) in collected
                    }
                    if len(by_length) < 2:
                        continue
                    cmp = compare_trajectory_lengths(by_length)
                    length_comparison[f"{label}|margin={margin}|{fdr_method}"] = cmp
                    worst_shift = max(worst_shift, cmp["max_filtration_shift"])
                    flags = ""
                    if not cmp["membership_invariant"]:
                        n_membership_broken += 1
                        flags += "  !! membership depends on K"
                    if not cmp["filtration_monotone_in_K"]:
                        n_monotonicity_broken += 1
                        flags += (f"  !! {cmp['n_monotonicity_violations']} filtrations "
                                  f"fell as K fell")
                    if not cmp["resolution_bound_respected"]:
                        n_bound_broken += 1
                        flags += (f"  !! {cmp['n_resolution_bound_violations']} exceed "
                                  f"the grid spacing")
                    if not header_done:
                        print(f"\n  {label}  margin={margin}")
                        print(f"    {'correction':<14}"
                              + "".join(f"{f'K={k}':>10}" for k in MAX_ITERATIONS_VALUES)
                              + f"{'inv':>6}{'mono':>6}{'shift':>9}")
                        header_done = True
                    cells = "".join(
                        f"{cmp['total_by_length'][k]:>10}" for k in MAX_ITERATIONS_VALUES
                    )
                    print(f"    {fdr_method:<14}{cells}"
                          f"{str(cmp['membership_invariant']):>6}"
                          f"{str(cmp['filtration_monotone_in_K']):>6}"
                          f"{cmp['max_filtration_shift']:>9.4f}{flags}")
    print(f"\n  Across every cell: {n_membership_broken} membership violations, "
          f"{n_monotonicity_broken} monotonicity\n  violations, {n_bound_broken} "
          f"resolution-bound violations; largest filtration shift {worst_shift:.4f} "
          f"(one\n  spacing of the coarsest grid is "
          f"{1.0 / min(MAX_ITERATIONS_VALUES):.4f}).")

    # Whether "minimum effective ablation" is a well-posed reading of the filtration.
    print(f"\n{'=' * 78}\nABLATION-RESPONSE MONOTONICITY "
          f"(first crossing vs last, at B={max(PROBE_COUNTS)})\n{'=' * 78}")
    print(f"  {'context':<28}{'margin':>7}" +
          "".join(f"{f'a={a}':>22}" for a in ALPHAS))
    for arch in ARCHITECTURES:
        for cls in CLASSES:
            label = f"{arch}/class_{cls}"
            for margin in MARGINS:
                diag = ablation_response.get((arch, cls, max(PROBE_COUNTS), margin))
                if not diag:
                    continue
                cells = []
                for alpha in ALPHAS:
                    d = diag.get(str(alpha))
                    if not d:
                        cells.append(f"{'-':>22}")
                        continue
                    cells.append(
                        f"{d['n_ragged']}/{d['n_significant_at_full_ablation']} "
                        f"(gap<={d['max_ablation_gap']:.2f})".rjust(22)
                    )
                print(f"  {label:<28}{margin:>7}" + "".join(cells))

    total_build = sum(build_times.values())
    print(f"\n{'=' * 78}\nHOW TO READ THIS\n{'=' * 78}")
    print("  Pick the smallest margin for which, in EVERY context:")
    print("    - conv = True (the whole complex, not just its 1-skeleton, has")
    print("      stopped changing as the probe count grows), and")
    print("    - the shape has not collapsed relative to margin 0 -- a margin")
    print("      that buys stability by finding nothing is not a good margin.")
    print("  Then read the same table across corrections and alphas: a margin")
    print("  that only stabilises under one correction has not been calibrated,")
    print("  it has been fitted.")
    print("  The ablation-grid table is a correctness check, not a choice: membership")
    print("  is settled at full ablation and every grid contains full ablation, so inv must")
    print("  be True everywhere -- a False there is a bug to chase, not a result.")
    print("  mono says the filtration only moved later as the grid coarsened, which")
    print("  is what landing on the first available grid point does, and shift is")
    print("  the resolution cost, which should sit at the 1/K scale. Those two can")
    print("  fail honestly, but only where the ablation-response table shows ragged")
    print("  trajectories: check there before treating a False as a bug.")
    print("  The monotonicity table counts admitted simplices whose test fires,")
    print("  stops, and fires again. Those are the ones for which 'minimum")
    print("  effective ablation' is a stretch: the filtration marks where the")
    print("  flickering stopped, and gap is how much of the ablation axis lies between")
    print("  that and the first crossing. If this is near zero the ablation-response is")
    print("  effectively monotone and the fixed-sequence reading is clean; if it is")
    print("  not, that is an argument for a trend test on the trajectory rather")
    print("  than for a different correction.")
    print("  The null/n column counts admitted 1-simplices whose |theta|/sigma AT")
    print("  FULL ABLATION sits inside the null region |theta|/sigma <= margin -- false")
    print("  discoveries by the study's own point estimate. Full ablation, not peak,")
    print("  because that is where membership is decided. It is a plug-in")
    print("  diagnostic rather than an unbiased FDR estimate, but every correction")
    print("  is scored against the same estimates, so it compares them fairly.")
    print("  It is undefined at margin 0, where the null region is empty -- which")
    print("  is exactly why a point null makes the comparison vacuous.")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": time.time() - started,
        "total_build_seconds": total_build,
        "jaccard_method": {
            "id": JACCARD_METHOD_ID,
            "quantity": "distance",
            "filtration_domain": [0.0, 1.0],
            "filtration_intervals": JACCARD_FILTRATION_INTERVALS,
            "filtration_points": JACCARD_FILTRATION_POINTS,
            "vertices_included": False,
        },
        "config": {
            "architectures": ARCHITECTURES, "classes": CLASSES,
            "probe_counts": PROBE_COUNTS, "margins": MARGINS,
            "alphas": ALPHAS, "fdr_methods": FDR_METHODS,
            "max_iterations_values": MAX_ITERATIONS_VALUES,
            "max_dimensions": MAX_DIMENSIONS, "max_iterations": MAX_ITERATIONS,
            "dt": DT, "relevance_scale": RELEVANCE_SCALE,
            "probe_tolerance": PROBE_TOLERANCE, "seed": SEED, "n_workers": n_workers,
        },
        "build_seconds": {str(k): v for k, v in build_times.items()},
        "family_sizes_by_dimension": family_sizes,
        "feasibility_floors": {
            f"{m}|{size}|{a}": floors[(m, size, a)] for (m, size, a) in floors
        },
        "effect_size_spectrum": spectra,
        "ablation_response": {str(k): v for k, v in ablation_response.items()},
        "correction_agreement": fdr_comparison,
        "ablation_grid_resolution": length_comparison,
        "series": summary,
    }
    out = OUTPUT_DIR / "inference_sensitivity.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Saved to {out}")
    print(f"  Total runtime: {time.time() - started:.1f} s "
          f"({total_build:.0f} s of it in construction)")


if __name__ == "__main__":
    with RunTimer("inference sensitivity") as run:
        main(run)
