"""Multiplicity correction over the family of simplices tested at one dimension.

The family is the dimension, never the ablation axis. Each candidate simplex
contributes a single test, taken at full ablation, so there is nothing to spend
across a trajectory: reading the ablation axis from the top down is the
fixed-sequence (step-down) MED procedure, which controls the familywise error
over ablations with no adjustment of its own.

`family_cutoff` turns one dimension's p-values into a rejection cutoff, and
`persistent_crossing_index` turns one trajectory into the filtration value.
"""

import math

import numpy as np

FAMILY_METHODS = ("none", "bonferroni", "bh", "by")


def family_cutoff(p_values, alpha=0.01, method="by"):
    """Rejection cutoff for one family of simplex tests, at one dimension.

    The family is every (predictor simplex, target) pair tested while building a
    given dimension -- those are the hypotheses the construction actually poses.
    It is emphatically not the trajectory: the ablation axis carries one hypothesis
    evaluated at increasing ablation_grid, not many hypotheses, and correcting across it
    penalises resolution rather than multiplicity. See
    `persistent_crossing_index` for why the ablation axis needs no correction at all.

    Args:
        p_values: p-value per family member, taken at full ablation (s = 1).
        alpha: level the family is controlled at.
        method: "none" (per-comparison), "bonferroni" (FWER), "bh" (FDR under
            independence or PRDS), or "by" (FDR under arbitrary dependence, at
            the cost of a log factor). Simplices sharing a predictor set reuse
            one trajectory, so their p-values are dependent by construction and
            PRDS is not established; "by" is the honest default and "bh" is
            reported alongside it to show what the assumption buys.

    Returns:
        float threshold c, with member i rejected exactly when p_i <= c. Fixed
        threshold methods return the value just below their nominal level, so
        `p <= c` reproduces the strict `p < alpha` convention used elsewhere.

        For the step-up methods this is the critical value the procedure stopped
        at, level * k / m, *not* the largest p-value it rejected. The two select
        exactly the same family members -- k is the largest index with
        p_(k) <= t_k, and t_k < t_(k+1) < p_(k+1), so nothing lies between
        p_(k) and t_k -- but they are different numbers, and only the threshold
        is meaningful away from s = 1. Dating a simplex by when its trajectory
        first drops below some *other* simplex's final p-value would make the
        filtration depend on an unrelated hypothesis; dating it by when the
        trajectory reaches the level the family was controlled at is what the
        fixed-sequence reading actually licenses.

        A family that rejects nothing yields -inf.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return -np.inf

    if method == "none":
        return float(np.nextafter(alpha, -np.inf))
    if method == "bonferroni":
        return float(np.nextafter(alpha / m, -np.inf))
    if method in ("bh", "by"):
        level = alpha
        if method == "by":
            # Benjamini-Yekutieli: the harmonic factor buys validity under any
            # dependence structure whatsoever.
            level = alpha / np.sum(1.0 / np.arange(1, m + 1))
        ordered = np.sort(p)
        thresholds = level * np.arange(1, m + 1) / m
        below = np.flatnonzero(ordered <= thresholds)
        if below.size == 0:
            return -np.inf
        return float(thresholds[below[-1]])
    raise ValueError(f"unknown correction {method!r}; expected one of {FAMILY_METHODS}")


def signed_rank_floor(n_probes):
    """
    Smallest two-sided p-value a Wilcoxon signed-rank test on `n_probes` pairs
    can return.

    Every paired difference has to fall on the same side for the statistic to
    reach its extreme, and there are 2^n sign assignments, so no configuration
    of the data can produce a p below 2^-n. This is a property of the test, not
    of the effect: a real effect of any size is unreportable below it.
    """
    return 2.0 ** -int(n_probes)


def min_probes_for_family(n_hypotheses, alpha=0.01, method="bonferroni"):
    """
    Fewest probes at which a family of this size can admit anything at all.

    A correction divides the budget among the hypotheses tested together, and a
    signed-rank test cannot return a p-value below `signed_rank_floor`. When the
    floor sits above the corrected threshold the dimension is decided before it
    is computed: nothing can clear the cutoff, whatever the network does.

    This bites at scale rather than in the small runs. The quadrant complexes
    test 56 pairs at dimension 1, where 13 probes suffice; an MLP with 240
    internal neurons tests 57,360, which needs 23, and the MNIST script's
    historical B = 20 is below that -- it would have spent the entire budget to
    admit an empty dimension.

    The step-up procedures need a configuration to have a threshold at all: BH
    and BY compare the k-th smallest p against a cutoff that grows with k, so a
    dimension where everything is significant is corrected far less harshly
    than one where a single simplex is. The count returned is therefore the
    requirement at the *most stringent rank* -- enough probes for a lone
    discovery to be reportable when nothing else in the family is. That is the
    conservative reading and the useful one: a family that cannot report its
    first discovery cannot report any.

    Args:
        n_hypotheses: size of the family corrected over.
        alpha: significance level.
        method: correction. "none" ignores the family size; "bonferroni" and
            "bh" agree at rank 1; "by" is stricter by the harmonic factor.

    Returns:
        Smallest integer n with `signed_rank_floor(n) <= threshold`, where the
        threshold is the one a lone discovery faces.
    """
    if method not in FAMILY_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {FAMILY_METHODS}")
    if n_hypotheses < 1:
        raise ValueError(f"a family holds at least one hypothesis, got {n_hypotheses}")

    if method == "none":
        threshold = alpha
    elif method == "bonferroni":
        threshold = alpha / n_hypotheses
    elif method == "bh":
        # Rank 1, where BH and Bonferroni coincide. At higher ranks BH is more
        # permissive, but a family that cannot clear rank 1 never gets there.
        threshold = alpha / n_hypotheses
    else:  # "by"
        harmonic = sum(1.0 / k for k in range(1, n_hypotheses + 1))
        threshold = alpha / (n_hypotheses * harmonic)

    return int(math.ceil(-math.log2(threshold)))


def persistent_crossing_index(flags):
    """Index from which the test stays significant to the end of the ablation axis.

    The filtration value is the smallest ablation at which the effect is
    established -- the minimum effective dose of the dose-response literature,
    with the ablation fraction in place of a dose. So it has to be the point
    past which the effect is present and stays present, not the first place the
    test happens to fire. Reading the last crossing rather than the
    first is what makes the reading a *minimum*: an isolated rejection at a low ablation
    that does not persist is noise along the trajectory, and taking it would
    date the simplex earlier than the evidence supports.

    Read from the largest ablation downward this is the classical fixed-sequence
    step-down procedure for a minimum effective dose: test the highest level,
    and only if it rejects test the next one down, stopping at the first
    failure. Because each test is reached only when every higher one has already
    rejected, the family-wise error over the ablation axis is controlled at alpha
    with no multiplicity adjustment. That is the formal reason the correction
    belongs to the simplex family and not to the trajectory.

    Args:
        flags: boolean array over the ablation grid, ordered by increasing ablation.

    Returns:
        int index where the final run of rejections begins, or None if the test
        does not reject at full ablation -- in which case there is no ablation at which
        the effect is established and the simplex is not admitted.
    """
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0 or not flags[-1]:
        return None
    misses = np.flatnonzero(~flags)
    return int(misses[-1] + 1) if misses.size else 0
