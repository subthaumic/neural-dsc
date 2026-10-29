"""
When a dimension is decided before it is computed.

A correction splits the budget among the hypotheses tested together, and a
signed-rank test on B pairs cannot return a p below 2^-B. If the floor sits
above the corrected threshold, no configuration of the data admits anything --
the run's only possible output is an empty dimension, and it costs the full
compute to produce.
"""

import math

import pytest

from neural_dsc.utils.fdr_correction import (
    family_cutoff,
    min_probes_for_family,
    signed_rank_floor,
)


def test_floor_is_the_smallest_reachable_p():
    assert signed_rank_floor(1) == 0.5
    assert signed_rank_floor(10) == pytest.approx(2.0 ** -10)


@pytest.mark.parametrize("m", [1, 7, 56, 1000, 57360])
@pytest.mark.parametrize("method", ["none", "bonferroni", "bh", "by"])
def test_the_count_is_what_a_lone_discovery_needs(m, method):
    """
    One candidate at the floor against a family of nulls -- the hardest case,
    and the one the returned count is defined by. At `n` it is reportable, at
    `n - 1` it is not.

    Stated against a configuration rather than a formula because BH and BY have
    no configuration-free threshold: their cutoff grows with the rank, so a
    dimension where everything is significant is corrected far more gently.
    """
    n = min_probes_for_family(m, alpha=0.01, method=method)

    def lone(probes):
        return [signed_rank_floor(probes)] + [1.0] * (m - 1)

    assert signed_rank_floor(n) <= family_cutoff(lone(n), 0.01, method)
    if n > 1:
        assert signed_rank_floor(n - 1) > family_cutoff(lone(n - 1), 0.01, method)


def test_the_quadrant_runs_clear_it_and_the_old_mnist_setting_does_not():
    """
    56 pairs at dimension 1 for the eight-neuron quadrant nets; 57,360 for an
    MLP with 240 internal neurons. B = 128 clears both; the MNIST script's
    historical B = 20 clears neither the second nor its own requirement of 23.
    """
    assert min_probes_for_family(56, 0.01, "bonferroni") == 13
    assert min_probes_for_family(240 * 239, 0.01, "bonferroni") == 23
    assert signed_rank_floor(128) <= 0.01 / (240 * 239)
    assert signed_rank_floor(20) > 0.01 / (240 * 239)


def test_family_size_does_not_enter_without_a_correction():
    assert min_probes_for_family(10, 0.01, "none") == min_probes_for_family(
        10_000_000, 0.01, "none"
    )


def test_by_is_stricter_than_bonferroni_by_the_harmonic_factor():
    m = 1000
    harmonic = sum(1.0 / k for k in range(1, m + 1))
    assert min_probes_for_family(m, 0.01, "by") == math.ceil(
        -math.log2(0.01 / (m * harmonic))
    )
    assert min_probes_for_family(m, 0.01, "by") >= min_probes_for_family(
        m, 0.01, "bonferroni"
    )


def test_rejects_nonsense_input():
    with pytest.raises(ValueError, match="unknown method"):
        min_probes_for_family(10, method="holm")
    with pytest.raises(ValueError, match="at least one hypothesis"):
        min_probes_for_family(0)
