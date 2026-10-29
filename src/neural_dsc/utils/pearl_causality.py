import numpy as np
import torch
from scipy.stats import wilcoxon

RELEVANCE_SCALE_MODES = ("std", "iqr", "median_abs", "class_pooled_std")


def relevance_epsilon(y_obs, baseline_acts, relevance_margin,
                      relevance_scale="std", scale_floor_frac=0.05):
    """
    Per-target relevance margin epsilon, in activation units.

    Neurons do not share a scale, so a single absolute epsilon would mean very
    different things for a neuron operating near 0.01 and one operating near 10,
    and would bias which simplices survive. Epsilon is therefore expressed as a
    multiple of the target's own activation scale on the *unperturbed*
    distribution: with the default mode, `relevance_margin` is read in standard
    deviations of the target's baseline activation across the probe set.

    The scale is deliberately **class-conditional**. The question the DSC asks is
    whether a neuron behaves differently than it normally would *for this class*,
    so the yardstick is that neuron's spread within the class, not its dynamic
    range over the whole input domain. A single global `relevance_margin` is
    multiplied by each target's own within-class SD, so the margin is comparable
    across neurons while the units remain those of the conditional distribution.

    Scale choice matters more than it looks. On high-purity probe sets a ReLU
    network leaves many neurons off for the entire class, so any location-based
    scale collapses: for the handcrafted quadrant nets the *median* absolute
    activation is exactly 0 for three neurons of mono class 0 and, pooled over
    the network, is 0 outright in three of four contexts. A zero scale silently
    restores the point null for precisely the neurons where the interval null
    matters most. Standard deviation degenerates only for neurons that are
    exactly constant, and the floor below catches those.

    The floor is a fraction of the mean per-neuron standard deviation, which is
    positive whenever anything in the network varies at all.

    Args:
        y_obs: baseline activations of the target neuron, shape (B,).
        baseline_acts: baseline activations of all neurons, shape (B, N), on
            *this class's* probe set. Used for the floor and for the
            "class_pooled_std" mode; every scale derived from it is therefore
            class-conditional.
        relevance_margin: epsilon in units of the target's scale. 0 disables
            relevance testing and recovers the point null exactly.
        relevance_scale: "std" (SD of the target's unperturbed activation),
            "iqr" (its interquartile range), "median_abs" (its typical
            magnitude — degenerates on sparse ReLU activations, see above), or
            "class_pooled_std" (one scale shared by every target, still taken
            within the class).
        scale_floor_frac: floor on the target scale, as a fraction of the mean
            per-neuron standard deviation.

    Returns:
        Float epsilon (0.0 when relevance testing is disabled).
    """
    if relevance_margin <= 0:
        return 0.0
    if relevance_scale not in RELEVANCE_SCALE_MODES:
        raise ValueError(
            f"relevance_scale must be one of {RELEVANCE_SCALE_MODES}, got {relevance_scale!r}"
        )

    baseline_acts = np.asarray(baseline_acts)
    # Mean of the per-neuron SDs: positive as long as any neuron varies, unlike
    # a pooled median absolute activation, which is 0 on sparse ReLU networks.
    class_scale = float(np.mean(np.std(baseline_acts, axis=0)))

    if relevance_scale == "std":
        scale = float(np.std(y_obs))
    elif relevance_scale == "iqr":
        scale = float(np.subtract(*np.percentile(y_obs, [75, 25])))
    elif relevance_scale == "median_abs":
        scale = float(np.median(np.abs(y_obs)))
    else:
        scale = class_scale

    scale = max(scale, scale_floor_frac * class_scale)
    return relevance_margin * scale


def _signed_rank_p(values, alternative):
    """
    One-sided Wilcoxon signed-rank p-value, tolerant of degenerate input.

    scipy raises when every value is zero (all observations dropped by the
    zero-handling), which is a "no evidence" case rather than an error.
    """
    if np.allclose(values, 0):
        return 1.0
    try:
        _, p_value = wilcoxon(values, alternative=alternative)
    except ValueError:
        return 1.0
    return float(p_value)


def relevance_test_pvalue(delta, epsilon):
    """
    p-value for the relevance test H0: |theta| <= epsilon against H1: |theta| > epsilon,
    where theta is the *pseudomedian* (Hodges-Lehmann location) of the paired
    differences `delta` -- the median of the Walsh averages (d_i + d_j)/2, which
    is the location functional the signed-rank test actually estimates.

    The distinction matters here rather than being pedantry. Counterfactual
    ablation produces a zero-inflated `delta`: on many probe points the
    intervention changes the target not at all, and on the rest it changes it a
    lot. The signed-rank statistic ranks by magnitude, so a substantial minority
    of large deviations outranks a majority of ties at -epsilon, and the test
    rejects while the sample median is exactly zero. Concretely, 212 differences
    of 1.0 against 300 zeros gives a sample median of 0 and p = 1e-10 at
    epsilon = 0.01, because the pseudomedian is 0.5.

    So an admitted simplex asserts that ablation matters on a substantial
    fraction of the class's probe points, not that the typical effect exceeds
    epsilon. That is the intended reading for a causal link -- a coupling active
    on half the inputs is a real coupling -- but it is a different claim from
    the one a median-based null would make, and the fraction at which the test
    turns over is implied by the rank statistic rather than chosen.

    Also known as a minimum-effect test. It is the mirror image of an
    equivalence (TOST) test: equivalence puts the *interval* in the alternative
    and establishes negligibility, whereas here the interval is the null and
    rejecting establishes that the effect is at least epsilon. The burden of
    proof therefore falls the way we want it for building a complex — absent
    evidence, a simplex is not added.

    The composite null is rejected if either one-sided test clears alpha:
    signed-rank on `delta - epsilon` against the alternative theta > epsilon, or
    on `delta + epsilon` against theta < -epsilon. No multiplicity adjustment is
    needed because the two alternatives are disjoint: at the boundary theta = eps
    the upper p-value is uniform while the lower one is ~1, so the size of the
    combined test is alpha, attained at the boundaries of the null interval.

    With epsilon = 0 this reduces to the ordinary two-sided signed-rank test.

    Args:
        delta: paired differences y_obs - y_counterfactual, shape (B,).
        epsilon: relevance margin in activation units (>= 0).

    Returns:
        Float p-value.
    """
    if np.allclose(delta, 0):
        return 1.0
    if epsilon <= 0:
        return _signed_rank_p(delta, "two-sided")
    p_upper = _signed_rank_p(delta - epsilon, "greater")
    p_lower = _signed_rank_p(delta + epsilon, "less")
    return min(p_upper, p_lower)


def pearl_causal_test(
    predictor_neurons,
    target_neurons,
    model,
    probe_data,
    dt=0.01,
    max_iterations=100,
    significance_level=0.01,
    relevance_margin=0.0,
    relevance_scale="std",
    return_ablation=False,
):
    """
    Performs counterfactual causal tests comparing baseline target neuron
    activations with those along the counterfactual ablation of the predictors.

    Supports both single target neurons and multiple target neurons.

    With `relevance_margin > 0` the test is a *relevance test* (also known as a
    minimum-effect test): the null is the interval H0: |theta| <= epsilon rather
    than the point H0: theta = 0, where theta is the median paired activation
    shift. This matters because the point null is false for essentially every
    (predictor set, target) pair in a densely connected network, so power tends
    to 1 for every candidate as the probe count grows and the complex fills in
    without limit. Against an interval null the detection threshold is pinned at
    epsilon by construction, and more probe points sharpen the estimate instead
    of lowering the threshold.

    With `relevance_margin = 0` (the default) the behaviour is exactly the
    previous two-sided point-null test.

        Parameters:
            predictor_neurons: index or list of indices for predictor neurons.
            target_neurons: index or list of indices for target neuron(s).
            model: an instance of the MLP class.
            probe_data: input data tensor (must contain at least 2 samples).
            dt: the ablation integrator's step size. It sets how accurately
                the descent path is followed, not where it ends.
            max_iterations: resolution of the ablation grid the trajectory is
                resampled onto, so t_values has max_iterations + 1 points
                spanning s = 0 to s = 1.
            significance_level: significance threshold for the test.
            relevance_margin: epsilon in units of the target neuron's
                class-conditional activation scale, i.e. one global margin
                multiplied by each target's own SD over *this* probe set.
                0 disables relevance testing.
            relevance_scale: how that per-target scale is measured; see
                `relevance_epsilon`.

    Returns:
      If target_neurons is a single index:
        tuple (t_values, significance_flags, p_values), with t_values the ablation
        grid over [0, 1].
      If target_neurons is a list of indices:
        Dictionary mapping each target neuron index to that tuple.
      With return_ablation=True, a (result, ablation_report) pair instead.
    """
    # Handle single predictor and multiple predictors
    single_predictor = not hasattr(predictor_neurons, "__iter__") or isinstance(
        predictor_neurons, (int, np.integer)
    )
    predictors = [predictor_neurons] if single_predictor else predictor_neurons

    # Handle single target and multiple targets
    single_target = not hasattr(target_neurons, "__iter__") or isinstance(
        target_neurons, (int, np.integer)
    )
    targets = [target_neurons] if single_target else target_neurons

    # Ensure probe_data is batched
    if probe_data.ndim == 1:
        probe_data = probe_data.unsqueeze(0)

    # Generate additional samples if fewer than 2 are provided
    if probe_data.shape[0] < 2:
        target_output = model(probe_data).detach()
        probe_data = model.generate_batch_from_target(target_output)

    # Compute the baseline activations for all neurons
    baseline_acts = model.get_activations(probe_data)  # shape: (B, total_neurons)

    # Run the counterfactual ablation and read the filtration off it. The
    # filtration value is the ablation: the fraction of the predictors' baseline
    # activation removed so far, so the axis is s in [0, 1] by construction and
    # s = 1 is the fully ablated counterfactual. `dt` sets the integrator's
    # step; `max_iterations` is the resolution of the ablation grid, not a horizon
    # the ablation may fail to reach within, because the descent runs until the
    # predictors are actually zero.
    acts_over_time, ablation_report = model.ablation_trajectory(
        probe_data, predictors, n_steps=max_iterations, dt=dt,
    )
    t_values = (np.arange(max_iterations + 1, dtype=np.float32)
                / float(max_iterations))

    # For each target neuron, evaluate significance at each ablation
    baseline_acts_np = baseline_acts.cpu().numpy()
    results = {}
    for target in targets: # type: ignore
        y_obs = baseline_acts[:, target]  # Baseline for this target
        significance_flags = []
        p_values_list = []

        # Relevance margin is a property of the target neuron, not of the step,
        # so it is computed once per target from the unablated activations.
        epsilon = relevance_epsilon(
            y_obs.cpu().numpy(), baseline_acts_np,
            relevance_margin, relevance_scale,
        )

        num_iterations = acts_over_time.shape[0]
        for step in range(num_iterations):
            acts_cf = acts_over_time[step]  # shape: (B, total_neurons)
            y_cf = acts_cf[:, target]
            delta = y_obs - y_cf
            delta_np = delta.cpu().numpy()

            p_value = relevance_test_pvalue(delta_np, epsilon)

            p_values_list.append(p_value)
            significance_flags.append(p_value < significance_level) # type: ignore

        significance_flags = np.array(significance_flags)
        p_values_array = np.array(p_values_list)
        results[target] = (t_values, significance_flags, p_values_array)


    # For single targets, return the tuple directly for backward compatibility
    out = results[target_neurons] if single_target else results
    if return_ablation:
        # Opt-in, because the results mapping is keyed by target and callers
        # iterate it; the report tells an unreachable counterfactual apart from
        # a genuinely absent effect, which is not something a target key can say.
        return out, ablation_report
    return out
