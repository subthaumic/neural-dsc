"""
Contexts in parallel, each context parallel inside.

The failure this guards against is silent. `build` wraps its pool creation in a
try/except and, when it fails, prints one line and carries on serially -- so a
nesting that Python refuses does not raise, it just runs at a fraction of the
speed with the right answer at the end. Python does refuse it for one of the
two process pool types: a worker of `mp.Pool` is daemonic and may not start
children, while a worker of `ProcessPoolExecutor` is not and may.
`build_or_load_many` uses the second, which is what makes this legal at all.
"""

import torch

from neural_dsc.models.gen import create_quadrants_mono
from neural_dsc.utils.parallel_dsc import DSCBuildSpec, build_or_load_many

BUILD_ARGS = dict(
    max_dimension=2,
    dt=0.05,
    max_iterations=20,
    fdr_correction="bonferroni",
    significance_level=0.01,
    relevance_margin=0.01,
    relevance_scale="std",
)


def _probes(n=24, seed=0):
    return 6.0 * torch.rand(n, 2, generator=torch.Generator().manual_seed(seed)) - 3.0


def _specs(tmp_path, n_contexts, n_workers, tag):
    model = create_quadrants_mono()
    return [
        DSCBuildSpec(
            label=f"{tag}-{i}",
            model=model,
            probe_data=_probes(seed=i),
            output_path=tmp_path / f"{tag}_{i}.txt",
            n_workers=n_workers,
            **BUILD_ARGS,
        )
        for i in range(n_contexts)
    ]


def _skeleton(sc):
    return {dim: sorted(sc.complex.get(dim, [])) for dim in sc.complex}


def test_a_spec_is_serial_unless_told_otherwise():
    """The quadrant producers pass no worker count and must not change."""
    spec = DSCBuildSpec(
        label="default", model=create_quadrants_mono(), probe_data=_probes(),
        output_path="unused.txt", **BUILD_ARGS,
    )
    assert spec.n_workers == 1


def test_nesting_does_not_fall_back_to_serial(tmp_path, capfd):
    """
    Two lanes, three processes inside each. The inner pools must actually be
    created -- the fallback is a printed line, not an exception, so the only
    evidence either way is in the output.
    """
    built = build_or_load_many(
        _specs(tmp_path, 2, n_workers=3, tag="nested"), configured_workers=2
    )
    captured = capfd.readouterr()
    combined = captured.out + captured.err

    assert "Falling back to sequential processing" not in combined
    assert "Creating worker pool with 3 workers" in combined
    assert "2 lanes x up to 3 processes each (6 total)" in combined
    assert len(built) == 2


def test_nesting_returns_what_serial_building_returns(tmp_path):
    """
    Splitting the loop over predictor sets must not change the complex. The
    correction is applied to a whole dimension at once, so a build that
    reduced per worker rather than per dimension would still return something
    -- just not the same something.
    """
    serial = build_or_load_many(
        _specs(tmp_path / "a", 3, n_workers=1, tag="serial"), configured_workers=1
    )
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    nested = build_or_load_many(
        _specs(tmp_path / "b", 3, n_workers=3, tag="nested"), configured_workers=3
    )

    assert [_skeleton(sc) for sc in serial] == [_skeleton(sc) for sc in nested]


def test_cached_complexes_are_not_rebuilt(tmp_path):
    specs = _specs(tmp_path, 2, n_workers=2, tag="cached")
    build_or_load_many(specs, configured_workers=2)
    again = build_or_load_many(specs, configured_workers=2)
    assert len(again) == 2
    assert all(spec.output_path.exists() for spec in specs)
