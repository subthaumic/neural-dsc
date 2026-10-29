"""
Purity-based probe point selection.

The top-k selection in `neural_dsc.utils.select` couples two things that ought to be
independent: *how many* probe points we draw (the sample size of the causal
test) and *how class-pure* they are. Because the candidate pool is a fixed
finite sample, raising k walks the selection down the cosine ranking and into
the decision boundary. At k = 252 the sweep diagnostics record a worst selected
cosine of 0.29 and 69/247 probes drawn from other classes for the polysemantic
class 3, so the class-conditional complex is no longer conditioned on its class.

Here the probe set is instead defined by a *purity criterion*, and points are
drawn by rejection sampling from the input domain until the requested number of
probes is accepted. Sample size and purity are then separate knobs: `n_points`
does not affect which region of input space the probes come from, and
`tolerance` does not affect how many there are.

Purity is measured as the cosine similarity between a chosen representation of
the model output and the target vector. Handcrafted quadrant models have
nonnegative, directly interpretable outputs and use them unchanged. Trained
classifiers emit signed logits, whose cosine is not identifiable: adding the
same value to every logit leaves the classifier unchanged but changes the
cosine. Their callers therefore transform logits to softmax probabilities
before measuring purity. Cosine then extends unchanged to the two-hot targets
used for the mixed-class complexes, where "equally strong on both classes" is
a direction statement.

The acceptance rule is absolute: a point is accepted when its cosine is at
least 1 - tolerance, i.e. within `tolerance` of a perfectly pure output. For
every target used in the article the networks attain cosine 1.0 on the input
domain, so the rule needs no calibration. (A tolerance of exactly 0 is a
different matter: the handcrafted polysemantic net attains cosine 1.0 for
exactly one point in class 1 and none at all in class 3.)

Earlier versions measured the threshold against an estimate of the largest
cosine the network attains, taken over 20,000 uniform calibration draws. For
the article's targets that estimate is exactly 1.0, so both rules accept the
same points. The calibration draws are still made when no reference is given,
because they come from the same generator as the candidates: dropping them
would shift the random stream and change which points are drawn, and the
stored input sets would no longer be reproduced.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import torch


@dataclass
class ProbeSelectionResult:
    """Probe points plus the diagnostics needed to audit the selection."""

    points: torch.Tensor
    achievable_purity: float
    threshold: float
    purities: torch.Tensor
    n_attempts: int
    acceptance_rate: float
    converged: bool
    diagnostics: Dict[str, float] = field(default_factory=dict)
    # Positions of the drawn points in the pool they came from. Only the
    # finite-pool sampler can fill this in; a point drawn from a continuous
    # domain has no index anywhere. Callers that need to point back at the
    # dataset -- to mark probes on an embedding, say -- must take them from
    # here rather than searching the pool for matching rows, which on MNIST is
    # a (pool x probes x 784) comparison and hundreds of megabytes per class.
    indices: Optional[torch.Tensor] = None


def uniform_box_sampler(low=-3.0, high=3.0, dim=2, generator=None):
    """
    Return a sampler drawing uniformly from a box, matching `generate_quadrant_data`.

    Args:
        low, high: box bounds (scalar, applied to every coordinate).
        dim: input dimensionality.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Callable n -> tensor of shape (n, dim).
    """

    def sample(n):
        raw = torch.rand(n, dim, generator=generator)
        return (high - low) * raw + low

    return sample


def target_purity(outputs, target_vec):
    """
    Cosine similarity between each model output and the target vector.

    Works for one-hot and two-hot targets alike.
    """
    target = target_vec.to(dtype=outputs.dtype, device=outputs.device)
    return torch.nn.functional.cosine_similarity(outputs, target.unsqueeze(0), dim=1)


def softmax_probabilities(logits):
    """Convert classifier logits to the shift-invariant probability simplex."""
    return torch.nn.functional.softmax(logits, dim=-1)


def save_probe_diagnostics(path, labeled_results, *, output_space):
    """Persist the acceptance audit for one resumable probe-data group."""
    payload = {
        "output_space": output_space,
        "contexts": [
            {"context": label, **result.diagnostics}
            for label, result in labeled_results
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _model_outputs(model, inputs, output_transform=None):
    outputs = model(inputs)
    return output_transform(outputs) if output_transform is not None else outputs


def estimate_achievable_purity(model, target_vec, sampler, n_calibration=20000,
                               batch_size=4096, output_transform=None):
    """
    Estimate the highest purity the model attains for this target.

    This is the calibration constant the acceptance threshold is measured
    against. It is estimated by sampling rather than optimisation so that it
    reflects purity attainable on the *input distribution*, not at some isolated
    optimum the sampler will never hit.

    Args:
        model: MLP instance.
        target_vec: one-hot or two-hot target (1D tensor).
        sampler: callable n -> inputs.
        n_calibration: total calibration points to draw.
        batch_size: forward-pass batch size.

    Returns:
        Float maximum observed purity.
    """
    best = -1.0
    drawn = 0
    while drawn < n_calibration:
        n = min(batch_size, n_calibration - drawn)
        inputs = sampler(n)
        with torch.no_grad():
            purities = target_purity(
                _model_outputs(model, inputs, output_transform), target_vec
            )
        best = max(best, float(purities.max()))
        drawn += n
    return best


def sample_probe_points(model, target_vec, sampler, n_points, tolerance=0.01,
                        achievable=None, max_attempts=None, batch_size=4096,
                        n_calibration=20000, output_transform=None):
    """
    Draw `n_points` probe points whose output purity is within `tolerance` of
    a perfectly pure output.

    Acceptance rule:

        cos(model(x), target_vec) >= reference - tolerance,  reference = 1

    Rejection sampling from `sampler` means the accepted points are distributed
    as the input distribution restricted to the accepted region -- unbiased
    within that region, and independent of `n_points`.

    Args:
        model: MLP instance.
        target_vec: one-hot or two-hot target (1D tensor).
        sampler: callable n -> inputs.
        n_points: number of probe points to accept.
        tolerance: purity slack below the reference.
        achievable: reference purity. None means 1.0 after the calibration
            draws (see below); the trained-network scripts pass 1.0 directly
            and make no calibration draws.
        max_attempts: abort after this many draws (default 1000 * n_points).
        batch_size: draw size per rejection round.
        n_calibration: calibration draws made when `achievable` is None. They
            no longer set the threshold; they are kept so that the candidates
            come from the same position in the random stream as in the runs
            that produced the stored input sets. Their maximum cosine is
            discarded.

    Returns:
        ProbeSelectionResult. `converged` is False if the attempt budget ran out
        before `n_points` were accepted, in which case `points` holds whatever
        was accepted -- callers should treat that as a failed context rather than
        silently proceeding on a short probe set.
    """
    if achievable is None:
        estimate_achievable_purity(
            model, target_vec, sampler,
            n_calibration=n_calibration, batch_size=batch_size,
            output_transform=output_transform,
        )
        achievable = 1.0
    threshold = achievable - tolerance
    if max_attempts is None:
        max_attempts = 1000 * n_points

    accepted = []
    accepted_purity = []
    n_accepted = 0
    n_eligible = 0
    attempts = 0

    while n_accepted < n_points and attempts < max_attempts:
        n = min(batch_size, max_attempts - attempts)
        inputs = sampler(n)
        with torch.no_grad():
            purities = target_purity(
                _model_outputs(model, inputs, output_transform), target_vec
            )
        attempts += n

        mask = purities >= threshold
        n_eligible += int(mask.sum())
        if bool(mask.any()):
            take = int(min(int(mask.sum()), n_points - n_accepted))
            idx = torch.nonzero(mask, as_tuple=False).flatten()[:take]
            accepted.append(inputs[idx])
            accepted_purity.append(purities[idx])
            n_accepted += take

    if accepted:
        points = torch.cat(accepted)
        purities = torch.cat(accepted_purity)
    else:
        points = torch.empty((0, *sampler(1).shape[1:]))
        purities = torch.empty((0,))

    diagnostics = {
        "n_requested": float(n_points),
        "n_accepted": float(points.shape[0]),
        "achievable_purity": float(achievable),
        "threshold": float(threshold),
        "tolerance": float(tolerance),
        "min_purity": float(purities.min()) if purities.numel() else float("nan"),
        "mean_purity": float(purities.mean()) if purities.numel() else float("nan"),
        "n_attempts": float(attempts),
        # Fraction of *drawn* points clearing the threshold, i.e. the measure of
        # the accepted region under the input distribution. Counted before the
        # final batch is truncated to n_points, so it does not depend on how
        # many probes were requested.
        "acceptance_rate": n_eligible / attempts if attempts else 0.0,
    }

    return ProbeSelectionResult(
        points=points,
        achievable_purity=float(achievable),
        threshold=float(threshold),
        purities=purities,
        n_attempts=attempts,
        acceptance_rate=diagnostics["acceptance_rate"],
        converged=points.shape[0] == n_points,
        diagnostics=diagnostics,
    )


def sample_probe_points_from_pool(model, target_vec, pool, n_points,
                                 tolerance=0.01, eligible=None, generator=None,
                                 output_transform=None, min_mixing=None):
    """
    Draw `n_points` probes from a finite pool, by the same rule as the
    continuous-domain sampler.

    Rejection sampling from a box accepts points of the input distribution that
    clear a purity threshold, and takes them without further preference. On a
    dataset the input distribution is the empirical one, so the same
    construction is: keep the pool members clearing the threshold, then draw
    uniformly among them. Uniformly is the operative word -- ranking the
    accepted set and taking the top `n_points` would put sample size and purity
    back on one knob, which is the coupling `select_topk_by_cosine_similarity`
    was replaced to avoid.

    On MNIST the tie-breaking matters more than it sounds. Measured on the
    trained classifier, softmax purity saturates: every digit has between two
    and five thousand correctly classified points within 1e-9 of the achievable
    maximum. A top-k over that set is decided entirely by the sort's treatment
    of ties, which is dataset order -- so "the 128 purest sevens" would in fact
    be "the first 128 sevens in MNIST".

    Args:
        model: MLP instance.
        target_vec: one-hot or two-hot target (1D tensor).
        pool: candidate inputs, shape (n, ...).
        n_points: number of probes to draw.
        tolerance: purity slack below the achievable maximum, measured over the
            pool rather than over the whole input domain.
        eligible: optional boolean mask restricting the pool before purity is
            measured, e.g. to points carrying the class's own label.
        generator: torch.Generator for the subsample; None leaves the global
            RNG in charge.
        output_transform: applied to model outputs before purity is measured.
            Trained classifiers emit signed logits, whose cosine is not
            identifiable, so their callers pass `softmax_probabilities`.
        min_mixing: for a multi-hot target only, the smallest share every one of
            its classes must carry before a point is eligible at all. None
            leaves the criterion as it was, which is correct for one-hot targets
            and unsafe for the rest -- see below.

    Purity is a cosine, so a *confident* output already scores 1/sqrt(k) against
    a k-hot target: 0.707 for a two-hot one, which is not small. The threshold
    is achievable-minus-tolerance and the achievable maximum is taken over the
    pool, so when the pool holds nothing genuinely mixed the maximum collapses
    to that floor and every confidently single-class point clears the bar. The
    result is a full probe set, no warning, and a "two-hot" context built
    entirely from unambiguous members of one class. Measured on MNIST, the pair
    6+7 admits 11,604 such points.

    `min_mixing` closes that off by restricting eligibility before the maximum
    is taken, so the achievable purity is computed over genuinely mixed points
    and the accepted set cannot contain anything else. If too few exist the
    caller's strict check fires, which is the outcome that can be acted on.

    Returns:
        ProbeSelectionResult. `converged` is False when the accepted set is
        smaller than `n_points`, in which case `points` holds all of it.
    """
    target_classes = torch.nonzero(target_vec, as_tuple=False).flatten()
    if min_mixing is not None and target_classes.numel() > 1:
        with torch.no_grad():
            shares = _model_outputs(model, pool, output_transform)
        mixed = (shares[:, target_classes] >= min_mixing).all(dim=1)
        if not bool(mixed.any()):
            classes = target_classes.tolist()
            raise ValueError(
                f"no point in the pool gives at least {min_mixing} to every one "
                f"of the target classes {classes}, so this multi-hot context "
                f"does not exist in the data. Without min_mixing the criterion "
                f"would have accepted confidently single-class points instead, "
                f"scoring 1/sqrt({len(classes)}) against the target, and "
                f"returned a full probe set with nothing mixed in it."
            )
        eligible = mixed if eligible is None else (eligible & mixed)

    candidates = pool if eligible is None else pool[eligible]
    n_eligible_pool = int(candidates.shape[0])
    if n_eligible_pool == 0:
        raise ValueError("the candidate pool is empty before purity is measured")

    with torch.no_grad():
        purities = target_purity(
            _model_outputs(model, candidates, output_transform), target_vec
        )
    achievable = float(purities.max())
    threshold = achievable - tolerance

    accepted_mask = purities >= threshold
    accepted_idx = torch.nonzero(accepted_mask, as_tuple=False).flatten()
    n_accepted_total = int(accepted_idx.numel())

    take = min(n_points, n_accepted_total)
    order = torch.randperm(n_accepted_total, generator=generator)[:take]
    chosen = accepted_idx[order]
    points = candidates[chosen]
    chosen_purities = purities[chosen]

    # Back to coordinates in the caller's pool, undoing the eligibility mask.
    if eligible is None:
        pool_indices = chosen
    else:
        pool_indices = torch.nonzero(eligible, as_tuple=False).flatten()[chosen]

    diagnostics = {
        "n_requested": float(n_points),
        "n_accepted": float(points.shape[0]),
        "achievable_purity": achievable,
        "threshold": threshold,
        "tolerance": float(tolerance),
        "min_purity": float(chosen_purities.min()) if chosen_purities.numel() else float("nan"),
        "mean_purity": float(chosen_purities.mean()) if chosen_purities.numel() else float("nan"),
        "n_attempts": float(n_eligible_pool),
        # Share of the eligible pool clearing the threshold: the finite-pool
        # counterpart of the accepted region's measure, and like it, counted
        # before the draw is truncated to n_points.
        "acceptance_rate": n_accepted_total / n_eligible_pool,
        "n_pool": float(n_eligible_pool),
        "n_accepted_total": float(n_accepted_total),
    }

    return ProbeSelectionResult(
        points=points,
        achievable_purity=achievable,
        threshold=threshold,
        purities=chosen_purities,
        n_attempts=n_eligible_pool,
        acceptance_rate=diagnostics["acceptance_rate"],
        converged=points.shape[0] == n_points,
        diagnostics=diagnostics,
        indices=pool_indices,
    )


def draw_probe_points_from_pool(model, target_vec, pool, n_points, tolerance=0.01,
                                eligible=None, seed=None, label="", strict=True,
                                verbose=True, min_mixing=None, **kwargs):
    """
    Seeded, audited wrapper over `sample_probe_points_from_pool`.

    The finite-pool counterpart of `draw_probe_points`, and it fails the same
    way for the same reason: a short probe set changes the sample size of every
    causal test in the run, so `strict` refuses it rather than quietly
    proceeding. Unlike the continuous case there is no attempt budget to raise
    -- if the pool cannot supply `n_points` at this tolerance, the choices are a
    wider tolerance or a smaller `n_points`, and both are decisions for the
    caller to make explicitly.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    result = sample_probe_points_from_pool(
        model, target_vec, pool, n_points, tolerance=tolerance,
        eligible=eligible, generator=generator, min_mixing=min_mixing, **kwargs
    )

    if verbose:
        d = result.diagnostics
        tag = f"{label}: " if label else ""
        print(
            f"  {tag}{int(d['n_accepted'])}/{n_points} probes drawn from "
            f"{int(d['n_accepted_total'])} accepted of {int(d['n_pool'])} pooled "
            f"(purity >= {d['threshold']:.6f} of achievable "
            f"{d['achievable_purity']:.6f}, min {d['min_purity']:.6f})"
        )

    if strict and not result.converged:
        raise RuntimeError(
            f"probe selection did not converge for {label or 'target'}: the pool "
            f"supplied {result.points.shape[0]} of {n_points} probes at tolerance "
            f"{tolerance} ({int(result.diagnostics['n_accepted_total'])} accepted "
            f"of {int(result.diagnostics['n_pool'])}). Raise the tolerance or "
            f"lower n_points -- do not proceed on a short probe set, which would "
            f"change the sample size of every test."
        )
    return result


def onehot_target(n_classes, class_idx, dtype=torch.float32):
    """One-hot target vector for a single class."""
    vec = torch.zeros(n_classes, dtype=dtype)
    vec[class_idx] = 1.0
    return vec


def twohot_target(n_classes, class_a, class_b, dtype=torch.float32):
    """Two-hot target vector for a mixed-class complex."""
    vec = torch.zeros(n_classes, dtype=dtype)
    vec[class_a] = 1.0
    vec[class_b] = 1.0
    return vec


def probe_seed(base_seed, class_idx, n_points):
    """Seed for one context's probe draw.

    Varying the seed with the context stops every class from being handed the
    same sampler stream, and varying it with the count stops a B=64 draw from
    being a prefix of a B=128 one, which would make the two look more alike than
    independent draws would.

    Every script that claims to use "the" probe set for a context has to derive
    its seed here. Two scripts with the same nominal configuration but different
    seeds return the same simplices with filtration values that differ by a few
    percent, which is a silent disagreement rather than a loud one.
    """
    return base_seed + 1000 * class_idx + n_points


def draw_probe_points(model, target_vec, n_points, tolerance=0.01, seed=None,
                      low=-3.0, high=3.0, dim=2, label="", strict=True,
                      verbose=True, **kwargs):
    """
    Draw a purity-filtered probe set from the input domain, seeded and audited.

    Convenience wrapper over `sample_probe_points` for the experiment scripts,
    which all need the same four things: a seeded uniform sampler matching
    `generate_quadrant_data`'s box, a printed diagnostic line, a hard failure
    when the requested number of probes could not be accepted, and a result
    object carrying the purity diagnostics into the saved outputs.

    Unlike `select_topk_by_cosine_similarity`, this does not draw from a fixed
    finite pool, so `n_points` is not capped by the dataset size and raising it
    does not walk the selection toward the decision boundary.

    Args:
        model: MLP instance.
        target_vec: one-hot or two-hot target (1D tensor).
        n_points: number of probe points to accept.
        tolerance: purity slack below the achievable maximum.
        seed: seed for the sampler; None leaves global RNG state in charge.
        low, high, dim: input-domain box, matching `generate_quadrant_data`.
        label: printed in the diagnostic line, e.g. "class 2".
        strict: raise if fewer than `n_points` probes were accepted. A short
            probe set silently changes the sample size of every causal test, so
            failing loudly is the safe default.
        verbose: print the diagnostic line.

    Returns:
        ProbeSelectionResult.

    Raises:
        RuntimeError: if `strict` and the attempt budget ran out first.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    sampler = uniform_box_sampler(low=low, high=high, dim=dim, generator=generator)

    result = sample_probe_points(
        model, target_vec, sampler, n_points, tolerance=tolerance, **kwargs
    )

    if verbose:
        d = result.diagnostics
        tag = f"{label}: " if label else ""
        print(
            f"  {tag}{int(d['n_accepted'])}/{n_points} probes accepted "
            f"(purity >= {d['threshold']:.4f} of achievable {d['achievable_purity']:.4f}, "
            f"min {d['min_purity']:.4f}, acceptance rate {d['acceptance_rate']:.4f})"
        )

    if strict and not result.converged:
        raise RuntimeError(
            f"probe selection did not converge for {label or 'target'}: accepted "
            f"{result.points.shape[0]} of {n_points} in {result.n_attempts} draws "
            f"(threshold {result.threshold:.4f}, acceptance rate "
            f"{result.acceptance_rate:.6f}). Raise `tolerance`, raise "
            f"`max_attempts`, or lower `n_points` -- do not proceed on a short "
            f"probe set, which would change the sample size of every test."
        )
    return result
