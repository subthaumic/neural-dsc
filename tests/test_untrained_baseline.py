import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch


def load_script_module(module_name, relative_path):
    repo_root = Path(__file__).parents[1]
    script_dir = repo_root / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_mono_untrained_module():
    return load_script_module(
        "quadrants_mono_untrained",
        "scripts/compute_quadrants_mono_untrained.py",
    )


def load_poly_untrained_module():
    return load_script_module(
        "quadrants_poly_untrained",
        "scripts/compute_quadrants_poly_untrained.py",
    )


def state_dicts_equal(model_a, model_b):
    return all(
        torch.equal(model_a.state_dict()[key], model_b.state_dict()[key])
        for key in model_a.state_dict()
    )


def test_untrained_model_init_is_seed_deterministic():
    module = load_mono_untrained_module()

    model_a = module.build_untrained_model(seed=7)
    model_b = module.build_untrained_model(seed=7)

    assert model_a.layer_sizes == [2, 4, 4, 4]
    assert state_dicts_equal(model_a, model_b)


def test_untrained_model_init_changes_with_seed():
    module = load_poly_untrained_module()

    model_a = module.build_untrained_model(seed=7)
    model_b = module.build_untrained_model(seed=8)

    assert model_a.layer_sizes == [2, 3, 4, 4]
    assert not state_dicts_equal(model_a, model_b)


def test_untrained_model_builder_does_not_use_handcrafted_constructors():
    for module in (load_mono_untrained_module(), load_poly_untrained_module()):
        source = inspect.getsource(module.build_untrained_model)
        assert "create_quadrants" not in source
        module_source = inspect.getsource(module)
        assert "create_quadrants" not in module_source
        assert "select_topk_by_cosine_similarity" not in module_source


def test_untrained_summary_aggregates_seed_rows():
    module = load_mono_untrained_module()
    base_metrics = {
        "dim1_simplices": 1,
        "dim2_simplices": 0,
        "dim3_simplices": 0,
        "median_filtration_nonvertex": 0.1,
        "probe_cosine_max": 0.5,
        "probe_cosine_mean": 0.4,
        "probe_cosine_median": 0.45,
    }
    rows = [
        {
            "architecture": "mono",
            "condition": "trained",
            "seed": "",
            "class_idx": 0,
            "total_simplices": 10,
            "nonvertex_simplices": 6,
            "jaccard_distance_to_trained": 0.0,
            **base_metrics,
        },
        {
            "architecture": "mono",
            "condition": "untrained",
            "seed": 0,
            "class_idx": 0,
            "total_simplices": 5,
            "nonvertex_simplices": 1,
            "jaccard_distance_to_trained": 0.2,
            **base_metrics,
        },
        {
            "architecture": "mono",
            "condition": "untrained",
            "seed": 1,
            "class_idx": 0,
            "total_simplices": 7,
            "nonvertex_simplices": 3,
            "jaccard_distance_to_trained": 0.4,
            **base_metrics,
        },
    ]

    summary = module.aggregate_summary_rows(rows)
    mono_q1 = next(
        row for row in summary if row["architecture"] == "mono" and row["class_idx"] == 0
    )

    assert mono_q1["trained_available"] is True
    assert mono_q1["n_untrained_seeds"] == 2
    assert mono_q1["trained_total_simplices"] == 10
    assert mono_q1["untrained_total_simplices_mean"] == pytest.approx(6.0)
    assert mono_q1["untrained_nonvertex_simplices_mean"] == pytest.approx(2.0)
    assert mono_q1["untrained_jaccard_distance_to_trained_mean"] == pytest.approx(0.3)


def test_split_untrained_entrypoints_bind_expected_architectures():
    mono = load_mono_untrained_module()
    poly = load_poly_untrained_module()

    assert mono.ARCHITECTURE == "mono"
    assert poly.ARCHITECTURE == "poly"
    assert mono.LAYER_SIZES == [2, 4, 4, 4]
    assert poly.LAYER_SIZES == [2, 3, 4, 4]
    assert mono.OUTPUT_DIR.name == "quadrants_mono_untrained"
    assert poly.OUTPUT_DIR.name == "quadrants_poly_untrained"
    assert mono.output_dir_for(1) == mono.OUTPUT_DIR / "seed1"
    assert poly.output_dir_for(2) == poly.OUTPUT_DIR / "seed2"
    assert mono.REFERENCE_RESULT_DIR == "quadrants_mono_trained"
    assert poly.REFERENCE_RESULT_DIR == "quadrants_poly_trained"
    assert mono.REFERENCE_MODEL_PATH == "quadrants_mono_trained.pth"
    assert poly.REFERENCE_MODEL_PATH == "quadrants_poly_trained.pth"
