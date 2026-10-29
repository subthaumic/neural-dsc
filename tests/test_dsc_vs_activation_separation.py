import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_dsc.mlp import MLP


def load_separation_module():
    script_path = Path(__file__).parents[1] / "scripts" / "plot_dsc_vs_activation_separation.py"
    spec = importlib.util.spec_from_file_location(
        "dsc_vs_activation_separation", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_complex(module, edge_filtrations):
    complex_ = module.DirectedSimplicialComplex()
    complex_.complex = {
        0: [((0,), 0.0), ((1,), 0.0)],
        1: [(tuple(simplex), filtration) for simplex, filtration in edge_filtrations],
    }
    return complex_


def test_filtration_integrated_jaccard_uses_1001_points_and_resolves_shifts():
    module = load_separation_module()
    reference = make_complex(module, [((0, 1), 0.20)])
    shifted = make_complex(module, [((0, 1), 0.21)])
    shifted_twice = make_complex(module, [((0, 1), 0.22)])

    assert module.FILTRATION_GRID_POINTS == 1001
    assert module.filtration_integrated_jaccard_distance(reference, reference) == 0.0
    assert module.terminal_jaccard_distance(reference, shifted) == 0.0
    first_shift = module.filtration_integrated_jaccard_distance(reference, shifted)
    second_shift = module.filtration_integrated_jaccard_distance(
        shifted, shifted_twice
    )
    combined_shift = module.filtration_integrated_jaccard_distance(
        reference, shifted_twice
    )
    assert first_shift == pytest.approx(10 / 1001)
    assert module.filtration_integrated_jaccard_distance(
        shifted, reference
    ) == first_shift
    assert combined_shift <= first_shift + second_shift + 1e-12


def test_headline_metrics_normalise_offdiagonal_distances():
    module = load_separation_module()
    jaccard = np.array(
        [
            [0.0, 1.0, 0.5],
            [1.0, 0.0, 0.2],
            [0.5, 0.2, 0.0],
        ]
    )
    cosine = np.array(
        [
            [0.0, 0.5, 0.25],
            [0.5, 0.0, 0.1],
            [0.25, 0.1, 0.0],
        ]
    )
    euclidean = np.array(
        [
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    metrics = module.compute_headline_metrics(jaccard, cosine, euclidean)

    assert metrics["mean_jaccard_distance"] == pytest.approx(0.5666667)
    assert metrics["mean_cosine_distance"] == pytest.approx(0.2833333)
    assert metrics["normalised_mean_jaccard_distance"] == pytest.approx(0.5666667)
    assert metrics["normalised_mean_cosine_distance"] == pytest.approx(0.5666667)
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["dsc_more_separated"] is False


def test_analyze_experiment_writes_fixture_outputs(tmp_path):
    module = load_separation_module()
    results_dir = tmp_path / "toy_results"
    output_dir = tmp_path / "analysis"
    figs_dir = tmp_path / "figs"
    results_dir.mkdir()

    model = MLP([2, 2, 2])
    with torch.no_grad():
        model.hidden_layers[0].weight.copy_(torch.eye(2))
        model.hidden_layers[0].bias.zero_()
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()
    model_path = tmp_path / "toy_model.pth"
    torch.save(model, model_path)

    torch.save(
        [
            torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        ],
        results_dir / "probe_data_onehot.pt",
    )
    class_a = make_complex(module, [((0, 1), 0.25)])
    class_b = make_complex(module, [((1, 0), 0.50)])
    class_a.save(str(results_dir / "dsc_class_0.txt"))
    class_b.save(str(results_dir / "dsc_class_1.txt"))

    config = module.ExperimentConfig(
        name="toy",
        results_dir=results_dir,
        n_classes=2,
        labels=("A", "B"),
        model_kind="generic_mlp",
        model_paths=(model_path,),
        layer_sizes=(2, 2, 2),
    )

    row = module.analyze_experiment(config, output_dir=output_dir, figs_dir=figs_dir)

    assert row["status"] == "ok"
    assert row["mean_jaccard_distance"] == pytest.approx(751 / 1001)
    assert row["mean_terminal_jaccard_distance"] == pytest.approx(1.0)
    assert row["mean_cosine_distance"] == pytest.approx(1.0)
    assert (output_dir / "toy_distances.json").exists()
    assert (figs_dir / "dsc_vs_activation_toy.pdf").exists()
