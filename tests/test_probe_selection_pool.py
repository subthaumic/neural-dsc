"""
Finite-pool probe selection: the dataset counterpart of rejection sampling.

The properties that matter are the ones the continuous sampler has and top-k
selection does not -- purity and sample size as separate knobs, and a draw that
does not depend on the order the pool happens to be stored in.
"""

import pytest
import torch

from neural_dsc.models.gen import create_quadrants_mono
from neural_dsc.utils.probe_selection import (
    draw_probe_points_from_pool,
    onehot_target,
    sample_probe_points_from_pool,
    softmax_probabilities,
)


def _pool(n=2000, seed=0):
    return 6.0 * torch.rand(n, 2, generator=torch.Generator().manual_seed(seed)) - 3.0


def test_every_drawn_point_clears_the_threshold():
    model = create_quadrants_mono()
    result = sample_probe_points_from_pool(
        model, onehot_target(4, 0), _pool(), 64, tolerance=0.01,
        generator=torch.Generator().manual_seed(1),
    )
    assert result.converged
    assert result.points.shape == (64, 2)
    assert bool((result.purities >= result.threshold).all())


def test_sample_size_does_not_change_which_region_is_accepted():
    """
    The coupling top-k has and this does not: raising the count must not lower
    the purity of the accepted set, only draw more from it.
    """
    model = create_quadrants_mono()
    small, large = (
        sample_probe_points_from_pool(
            model, onehot_target(4, 0), _pool(), n, tolerance=0.01,
            generator=torch.Generator().manual_seed(1),
        )
        for n in (16, 256)
    )
    assert small.threshold == large.threshold
    assert small.diagnostics["n_accepted_total"] == large.diagnostics["n_accepted_total"]
    assert large.purities.min() >= large.threshold


def test_the_draw_does_not_follow_pool_order():
    """
    Where purity saturates -- as it does for every MNIST digit, thousands of
    points within 1e-9 of the maximum -- a top-k is decided by tie-breaking,
    which is dataset order. A uniform draw is not.
    """
    model = create_quadrants_mono()
    pool = _pool(n=4000)
    result = sample_probe_points_from_pool(
        model, onehot_target(4, 0), pool, 64, tolerance=0.5,
        generator=torch.Generator().manual_seed(7),
    )
    prefix = pool[:64]
    assert not torch.equal(result.points, prefix)


def test_the_draw_is_reproducible_under_a_seed():
    model, pool, target = create_quadrants_mono(), _pool(), onehot_target(4, 1)
    a, b = (
        draw_probe_points_from_pool(
            model, target, pool, 32, tolerance=0.05, seed=99, verbose=False
        ).points
        for _ in range(2)
    )
    assert torch.equal(a, b)


def test_eligibility_mask_is_applied_before_purity_is_measured():
    """
    The achievable maximum is calibrated on the pool, so restricting the pool
    has to restrict the calibration with it -- otherwise the threshold is set by
    points that were never candidates.
    """
    model, pool, target = create_quadrants_mono(), _pool(), onehot_target(4, 2)
    mask = torch.zeros(len(pool), dtype=torch.bool)
    mask[:200] = True
    restricted = sample_probe_points_from_pool(
        model, target, pool, 8, tolerance=0.01, eligible=mask,
        generator=torch.Generator().manual_seed(0),
    )
    assert restricted.diagnostics["n_pool"] == 200.0
    assert all((point == pool[:200]).all(dim=1).any() for point in restricted.points)


def test_a_short_pool_fails_loudly():
    model, target = create_quadrants_mono(), onehot_target(4, 0)
    mask = torch.zeros(2000, dtype=torch.bool)
    mask[:4] = True
    with pytest.raises(RuntimeError, match="did not converge"):
        draw_probe_points_from_pool(
            model, target, _pool(), 128, tolerance=1e-9, eligible=mask,
            seed=0, verbose=False,
        )


def test_output_transform_is_threaded_through():
    """Trained classifiers measure purity on probabilities, not signed logits."""
    model, pool, target = create_quadrants_mono(), _pool(), onehot_target(4, 3)
    result = sample_probe_points_from_pool(
        model, target, pool, 16, tolerance=0.01,
        generator=torch.Generator().manual_seed(0),
        output_transform=softmax_probabilities,
    )
    assert result.achievable_purity <= 1.0
    assert result.points.shape == (16, 2)


def test_indices_point_back_at_the_pool():
    """
    The drawn rows are recoverable by index, not by searching the pool for
    matching values -- on MNIST that search is a (pool x probes x 784)
    comparison, hundreds of megabytes per class.
    """
    model, pool, target = create_quadrants_mono(), _pool(), onehot_target(4, 0)
    result = sample_probe_points_from_pool(
        model, target, pool, 32, tolerance=0.05,
        generator=torch.Generator().manual_seed(4),
    )
    assert result.indices is not None
    assert torch.equal(pool[result.indices], result.points)


def test_indices_survive_an_eligibility_mask():
    """The mask reindexes the candidates; the returned indices must not."""
    model, pool, target = create_quadrants_mono(), _pool(), onehot_target(4, 1)
    mask = torch.zeros(len(pool), dtype=torch.bool)
    mask[500:1500] = True
    result = sample_probe_points_from_pool(
        model, target, pool, 16, tolerance=0.05, eligible=mask,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.equal(pool[result.indices], result.points)
    assert bool(mask[result.indices].all())


def test_the_continuous_sampler_has_no_indices():
    from neural_dsc.utils.probe_selection import draw_probe_points

    result = draw_probe_points(
        create_quadrants_mono(), onehot_target(4, 0), 8, seed=0, verbose=False
    )
    assert result.indices is None


# --- the multi-hot degeneracy, and the guard against it ------------------------
#
# Purity is a cosine, so a confident output already scores 1/sqrt(k) against a
# k-hot target. The threshold is achievable-minus-tolerance with the maximum
# taken over the pool, so a pool holding nothing genuinely mixed collapses the
# maximum onto that floor and admits every confident single-class point instead.
# That mode is silent: it returns a full probe set. On MNIST it is what the pair
# 6+7 does, with 11,604 accepted and none of them ambiguous.


def _twohot(n_classes, a, b):
    target = torch.zeros(n_classes)
    target[a] = target[b] = 1.0
    return target


def test_a_two_hot_target_admits_unmixed_points_without_the_guard():
    """The degenerate mode is silent: a full probe set, nothing mixed in it.

    Q1 and Q3 are opposite quadrants, so no input is genuinely between them.
    The criterion should therefore have nothing to offer -- but it returns a
    full set anyway, because a confident output still scores 1/sqrt(2) against
    the two-hot target and the pool maximum has collapsed to around that floor.
    The exact floor depends on how saturated the network is, so the invariant
    tested here is the behaviour rather than the constant.
    """
    model, pool = create_quadrants_mono(), _pool()
    result = sample_probe_points_from_pool(
        model, _twohot(4, 0, 2), pool, 16, tolerance=0.01,
        generator=torch.Generator().manual_seed(3),
        output_transform=softmax_probabilities,
    )
    assert result.converged, "the degenerate mode is silent -- it returns a full set"
    probs = softmax_probabilities(model(result.points))
    mixed = (probs[:, [0, 2]] >= 0.3).all(dim=1)
    assert not bool(mixed.any()), "none of these probes is genuinely mixed"
    assert result.achievable_purity < 0.95, "and the maximum is nowhere near 1.0"


def test_min_mixing_refuses_the_degenerate_case():
    model, pool = create_quadrants_mono(), _pool()
    with pytest.raises(ValueError, match="does not exist in the data"):
        sample_probe_points_from_pool(
            model, _twohot(4, 0, 2), pool, 16, tolerance=0.01, min_mixing=0.3,
            generator=torch.Generator().manual_seed(3),
            output_transform=softmax_probabilities,
        )


def test_min_mixing_still_admits_a_genuinely_mixed_pair():
    """Adjacent quadrants do have points between them, so the guard passes."""
    model, pool = create_quadrants_mono(), _pool()
    result = sample_probe_points_from_pool(
        model, _twohot(4, 0, 1), pool, 16, tolerance=0.5, min_mixing=0.3,
        generator=torch.Generator().manual_seed(3),
        output_transform=softmax_probabilities,
    )
    assert result.converged
    probs = softmax_probabilities(model(result.points))
    assert bool((probs[:, [0, 1]] >= 0.3).all()), "every probe must be mixed"


def test_min_mixing_leaves_one_hot_selection_untouched():
    """One-hot targets have a single class, so the guard must not engage."""
    model, pool = create_quadrants_mono(), _pool()
    draw = lambda **kw: sample_probe_points_from_pool(
        model, onehot_target(4, 1), pool, 32, tolerance=0.01,
        generator=torch.Generator().manual_seed(11), **kw
    )
    assert torch.equal(draw().points, draw(min_mixing=0.3).points)
