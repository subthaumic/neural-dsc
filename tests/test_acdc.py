"""
Consistency of the activation-patching machinery ACDC is built on.

The algorithm's edge scores are differences between patched forward passes, so
every score is meaningless unless the patch reconstructs the right network at
the two ends of its range: patching nothing must give the clean run, and
patching every edge must give the corrupted run. Both failed before -- the
patch read its source activations from an `nn.Linear` forward hook, which
returns the pre-activation rather than the value an outgoing edge multiplies.
"""

import torch

from neural_dsc.acdc.acdc import ACDC
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly


def _probe_points(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 6.0 * torch.rand(n, 2, generator=g) - 3.0


def test_layer_activations_are_post_relu():
    model = create_quadrants_mono()
    x = _probe_points()
    acts = ACDC(model)._get_layer_activations(x)
    _, forward_acts = model(x, return_activations=True)

    for i in forward_acts:
        assert torch.allclose(acts[i], forward_acts[i])
    assert (acts[0] >= 0).all()


def test_patching_nothing_reproduces_the_clean_run():
    model = create_quadrants_mono()
    acdc = ACDC(model)
    x = _probe_points()
    xc = acdc.create_corrupted_data(x, "flip", seed=1)

    out = acdc._run_with_activation_patching(
        x, xc, acdc._get_layer_activations(x), acdc._get_layer_activations(xc), set()
    )
    with torch.no_grad():
        assert torch.allclose(out, model(x), atol=1e-6)


def test_patching_every_edge_reproduces_the_corrupted_run():
    """
    The other end of the range. Only hidden-to-hidden edges exist, so patching
    all of them replaces every input to the second hidden layer with its
    corrupted value; from there the two runs coincide.
    """
    for factory in (create_quadrants_mono, create_quadrants_poly):
        model = factory()
        acdc = ACDC(model)
        x = _probe_points()
        xc = acdc.create_corrupted_data(x, "flip", seed=1)

        out = acdc._run_with_activation_patching(
            x, xc,
            acdc._get_layer_activations(x),
            acdc._get_layer_activations(xc),
            acdc.active_edges,
        )
        with torch.no_grad():
            assert torch.allclose(out, model(xc), atol=1e-6), factory.__name__


def test_corruption_is_reproducible_under_a_seed():
    model = create_quadrants_mono()
    acdc = ACDC(model)
    x = _probe_points()
    for kind in ("noise", "shuffle", "zero", "flip"):
        a = acdc.create_corrupted_data(x, kind, seed=7)
        b = acdc.create_corrupted_data(x, kind, seed=7)
        assert torch.equal(a, b), kind


def test_metric_does_not_scale_with_output_width():
    """
    `reduction='mean'` divides by batch *and* classes, so the same divergence
    would read four times smaller here than on a two-class head -- and the
    removal threshold is compared against exactly this number.
    """
    model = create_quadrants_mono()
    acdc = ACDC(model)
    clean = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    patched = torch.tensor([[0.0, 2.0, 0.0, 0.0]])

    expected = torch.nn.functional.kl_div(
        torch.log_softmax(patched, -1), torch.softmax(clean, -1), reduction="batchmean"
    ).item()
    assert acdc._compute_metric(clean, patched) == expected
