"""
Attribution patching: what the linear estimate gets right, and where it stops.

Two properties are worth pinning. The estimate has to track the patch it is an
estimate of, which it does while the perturbation stays inside one linear
region. And on ACDC's own metric the single-step version carries no signal at
all, which is not a defect of the implementation but the reason the
integrated-gradients variant exists.
"""

import torch

from neural_dsc.acdc.acdc import ACDC
from neural_dsc.acdc.eap import (
    edge_attribution_scores,
    kl_to_clean,
    target_logit_diff,
    top_k_circuit,
)
from neural_dsc.acdc.graph import hidden_edges
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly


def _clean_points(n=64, seed=0):
    return 6.0 * torch.rand(n, 2, generator=torch.Generator().manual_seed(seed)) - 3.0


def test_kl_carries_no_signal_at_the_clean_run():
    """
    KL to the clean output is minimised at the clean output, so its gradient
    there is zero and single-step EAP scores every edge at rounding noise --
    around 1e-8 against the 1e-1 the integrated version reports on the same
    edges. Read the module docstring before choosing `kl_to_clean` with
    `n_steps = 1`.
    """
    for factory in (create_quadrants_mono, create_quadrants_poly):
        model = factory()
        clean = _clean_points()
        corrupted = ACDC(model).create_corrupted_data(clean, "flip", seed=1)

        eap = edge_attribution_scores(model, clean, corrupted, kl_to_clean, n_steps=1)
        eap_ig = edge_attribution_scores(model, clean, corrupted, kl_to_clean, n_steps=16)

        assert set(eap) == hidden_edges(model)
        assert max(abs(v) for v in eap.values()) < 1e-6
        assert max(abs(v) for v in eap_ig.values()) > 1e-2


def test_task_metric_gives_single_step_eap_a_signal():
    model = create_quadrants_mono()
    clean = _clean_points()
    corrupted = ACDC(model).create_corrupted_data(clean, "flip", seed=1)
    scores = edge_attribution_scores(
        model, clean, corrupted, target_logit_diff(0), n_steps=1
    )
    assert max(abs(v) for v in scores.values()) > 1e-3


def test_estimate_tracks_the_patch_inside_one_linear_region():
    """
    Against an actual single-edge patch of the same edge. The counterfactual is
    kept small so the perturbation does not push any unit across its kink: that
    is the regime the first-order estimate is entitled to, and outside it the
    agreement is a question for the experiment rather than for a test.
    """
    model = create_quadrants_mono()
    clean = _clean_points()
    corrupted = clean + 1e-4 * torch.randn(
        clean.shape, generator=torch.Generator().manual_seed(3)
    )

    metric = target_logit_diff(0)
    acdc = ACDC(model)
    clean_acts = acdc._get_layer_activations(clean)
    corrupted_acts = acdc._get_layer_activations(corrupted)
    with torch.no_grad():
        clean_logits = model(clean)
        baseline = metric(clean_logits, clean_logits).item()

    scores = edge_attribution_scores(model, clean, corrupted, metric, n_steps=1)

    compared = 0
    for edge, estimate in scores.items():
        patched = acdc._run_with_activation_patching(
            clean, corrupted, clean_acts, corrupted_acts, {edge}
        )
        with torch.no_grad():
            true_change = metric(clean_logits, patched).item() - baseline
        if abs(true_change) < 1e-9:
            continue
        compared += 1
        assert estimate * true_change > 0, (edge, estimate, true_change)
    assert compared >= 8


def test_scores_cover_both_architectures():
    for factory in (create_quadrants_mono, create_quadrants_poly):
        model = factory()
        clean = _clean_points()
        corrupted = ACDC(model).create_corrupted_data(clean, "flip", seed=1)
        scores = edge_attribution_scores(
            model, clean, corrupted, target_logit_diff(0), n_steps=8
        )
        assert set(scores) == hidden_edges(model)
        assert len(top_k_circuit(scores, 5)) == 5


def test_acdc_records_a_score_for_every_edge_it_kept():
    """
    ACDC's filtration. The relative damage an edge was last measured to cause is
    the threshold at which it would have been removed, so one run yields both a
    circuit and a ranking of every edge it examined.
    """
    model = create_quadrants_mono()
    clean = _clean_points(n=32)
    acdc = ACDC(model, threshold=0.01)
    corrupted = acdc.create_corrupted_data(clean, "flip", seed=1)
    result = acdc.discover_circuit(clean, corrupted, max_iterations=1000)

    assert set(result.edge_scores) <= acdc.active_edges
    assert result.edges <= set(result.edge_scores)
