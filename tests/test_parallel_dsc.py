from pathlib import Path

import torch

from neural_dsc.mlp import MLP
from neural_dsc.utils.parallel_dsc import (
    DSCBuildSpec,
    build_or_load_many,
    resolve_outer_workers,
)


def build_spec(model, probe_data, path, label):
    return DSCBuildSpec(
        label=label,
        model=model,
        probe_data=probe_data,
        output_path=Path(path),
        max_dimension=1,
        dt=0.1,
        max_iterations=2,
        fdr_correction="none",
        significance_level=0.05,
        relevance_margin=0.01,
        relevance_scale="std",
    )


def test_outer_pool_builds_and_resumes_in_input_order(tmp_path):
    torch.manual_seed(7)
    model = MLP([2, 2, 2])
    model.eval()
    probe_data = torch.randn(8, 2)
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    specs = [
        build_spec(model, probe_data, paths[0], "first"),
        build_spec(model, probe_data + 0.1, paths[1], "second"),
    ]

    complexes = build_or_load_many(specs, configured_workers=2)

    assert len(complexes) == 2
    assert all(path.exists() for path in paths)
    mtimes = [path.stat().st_mtime_ns for path in paths]

    cached = build_or_load_many(specs, configured_workers=2)

    assert [path.stat().st_mtime_ns for path in paths] == mtimes
    assert [sc.complex for sc in cached] == [sc.complex for sc in complexes]


def test_outer_worker_auto_mode_leaves_four_cpus(monkeypatch):
    monkeypatch.setattr("neural_dsc.utils.parallel_dsc.mp.cpu_count", lambda: 52)
    assert resolve_outer_workers(None) == 48
    assert resolve_outer_workers(-1) == 48
