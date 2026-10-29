import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from neural_dsc.directed_simplicial_complex import (
    DirectedSimplicialComplex,
    build_dsc_from_cached_pvalues,
    process_simplex,
)
from neural_dsc.utils.fdr_correction import family_cutoff


def load_script_module(name, relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def violating_complex():
    dsc = DirectedSimplicialComplex()
    dsc.complex = {
        0: [((0,), 0.0), ((1,), 0.0), ((2,), 0.0)],
        1: [((0, 1), 0.1), ((0, 2), 0.2), ((1, 2), 0.9)],
        2: [((0, 1, 2), 0.3)],
    }
    return dsc


def test_validate_and_repair_filtration_closure():
    dsc = violating_complex()

    violations = dsc.validate_filtration_closure()
    assert len(violations) == 1
    assert violations[0]["simplex"] == (0, 1, 2)
    assert violations[0]["face"] == (1, 2)

    repair = dsc.repair_filtration_closure()
    assert repair["changed_simplices"] == 1
    assert repair["max_increase"] == pytest.approx(0.6)
    assert dsc.complex[2] == [((0, 1, 2), 0.9)]
    assert dsc.validate_filtration_closure() == []


def test_repair_filtration_closure_leaves_valid_complex_unchanged():
    dsc = DirectedSimplicialComplex()
    dsc.complex = {
        0: [((0,), 0.0), ((1,), 0.0), ((2,), 0.0)],
        1: [((0, 1), 0.1), ((0, 2), 0.2), ((1, 2), 0.3)],
        2: [((0, 1, 2), 0.4)],
    }

    repair = dsc.repair_filtration_closure()

    assert repair["changed_simplices"] == 0
    assert repair["max_increase"] == 0.0
    assert dsc.complex[2] == [((0, 1, 2), 0.4)]


class TinyModel:
    internal_neurons = 3
    layer_sizes = [1, 3, 1]

    def state_dict(self):
        return {}


def fake_pearl_causal_test(predictor_neurons, target_neurons, **kwargs):
    if isinstance(target_neurons, int):
        target_neurons = [target_neurons]
    predictor = tuple(predictor_neurons)
    results = {}
    for target in target_neurons:
        birth = 0.9 if predictor == (1,) and target == 2 else 0.1
        if predictor == (0, 1) and target == 2:
            birth = 0.2
        t_vals = np.array([0.0, birth])
        sig_flags = np.array([False, True])
        p_vals = np.array([1.0, 0.0])
        results[target] = (t_vals, sig_flags, p_vals)
    return results


def test_sequential_build_uses_all_face_filtrations(monkeypatch):
    pearl_module = load_script_module(
        "pearl_module_for_filtration_test",
        "src/neural_dsc/utils/pearl_causality.py",
    )
    monkeypatch.setattr(pearl_module, "pearl_causal_test", fake_pearl_causal_test)
    monkeypatch.setitem(sys.modules, "neural_dsc.utils.pearl_causality", pearl_module)

    dsc = DirectedSimplicialComplex()
    dsc.build(
        model=TinyModel(),
        probe_data=None,
        max_dimension=2,
        n_workers=1,
        fdr_correction="none",
    )

    triangle_filtration = {
        simplex: filtration for simplex, filtration in dsc.complex[2]
    }[(0, 1, 2)]
    assert triangle_filtration == pytest.approx(0.9)
    assert dsc.validate_filtration_closure() == []


def test_process_simplex_worker_path_uses_all_face_filtrations(monkeypatch):
    pearl_module = load_script_module(
        "pearl_module_for_worker_filtration_test",
        "src/neural_dsc/utils/pearl_causality.py",
    )
    monkeypatch.setattr(pearl_module, "pearl_causal_test", fake_pearl_causal_test)
    monkeypatch.setitem(sys.modules, "neural_dsc.utils.pearl_causality", pearl_module)

    import neural_dsc.directed_simplicial_complex as dsc_module

    dsc_module.worker_model = TinyModel()
    dsc_module.worker_probe_data = None
    dsc_module.worker_vertices = [0, 1, 2]
    dsc_module.worker_dt = 0.1
    dsc_module.worker_max_iterations = 10
    dsc_module.worker_significance_level = 0.05
    dsc_module.worker_fdr_correction = "none"
    dsc_module.worker_collect_pvalues = False

    face_filtrations = {
        (0, 1): 0.1,
        (0, 2): 0.2,
        (1, 2): 0.9,
    }
    # process_simplex no longer decides anything: the correction's family spans
    # every simplex tested at the dimension, so it returns raw trajectories and
    # the reduce happens once, in admit_family.
    family, *_ = process_simplex((((0, 1), 0.1), face_filtrations))
    admitted, n = DirectedSimplicialComplex.admit_family(
        family, face_filtrations, alpha=0.05, method="none"
    )

    assert ((0, 1, 2), 0.9) in admitted
    assert n == len(admitted)


ABLATION_POINTS = 11  # index i sits at ablation s = i / 10


def pvalues_for_birth(index):
    """A trajectory that rejects from `index` onward and not before.

    The decision is taken at full ablation and the filtration is the last crossing,
    so a birth has to be a rejection that *persists* to the end of the axis. A
    lone rejection at one index, which is what this fixture used to produce,
    now means the simplex is never admitted at all.
    """
    values = np.ones(ABLATION_POINTS)
    values[index:] = 0.0
    return values


def test_fdr_cached_pvalue_builder_uses_all_face_filtrations():
    pvalues = []
    for predictor in [(0,), (1,), (2,)]:
        for target in {0, 1, 2} - set(predictor):
            birth_index = 9 if predictor == (1,) and target == 2 else 1
            pvalues.append({
                "dimension": 1,
                "predictor": predictor,
                "target": target,
                "all_pvalues": pvalues_for_birth(birth_index),
            })
    for predictor in [
        (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    ]:
        target = next(iter({0, 1, 2} - set(predictor)))
        birth_index = 2 if predictor == (0, 1) and target == 2 else 1
        pvalues.append({
            "dimension": 2,
            "predictor": predictor,
            "target": target,
            "all_pvalues": pvalues_for_birth(birth_index),
        })

    dsc = build_dsc_from_cached_pvalues(
        pvalues,
        model=TinyModel(),
        max_dimension=2,
        dt=0.1,
        significance_level=0.05,
        fdr_method="none",
    )

    triangle_filtration = {
        simplex: filtration for simplex, filtration in dsc.complex[2]
    }[(0, 1, 2)]
    assert triangle_filtration == pytest.approx(0.9)
    assert dsc.validate_filtration_closure() == []



# ---------------------------------------------------------------------------
# The decision rule itself: one test per simplex at full ablation, corrected over
# the dimension's family, dated by the last crossing.
# ---------------------------------------------------------------------------

def _family(*p_value_rows):
    """A dimension's worth of candidates, with all faces already born at 0."""
    grid = np.linspace(0.0, 1.0, len(p_value_rows[0]))
    family = [((0, 1, i + 2), grid, np.asarray(row))
              for i, row in enumerate(p_value_rows)]
    faces = {face: 0.0
             for simplex, _, _ in family
             for face in DirectedSimplicialComplex.codimension_one_faces(simplex)}
    return family, faces


def test_admit_family_requires_rejection_at_full_ablation():
    """A simplex significant early but not at full ablation is not admitted."""
    early_only = [1.0, 0.0, 0.0, 1.0, 1.0]
    admitted, n = DirectedSimplicialComplex.admit_family(
        *_family(early_only), alpha=0.05, method="none"
    )
    assert admitted == [] and n == 0


def test_admit_family_dates_by_last_crossing_not_first():
    """An isolated early rejection does not date the simplex."""
    flickering = [1.0, 0.0, 1.0, 0.0, 0.0]
    admitted, _ = DirectedSimplicialComplex.admit_family(
        *_family(flickering), alpha=0.05, method="none"
    )
    assert len(admitted) == 1
    # grid is [0, .25, .5, .75, 1]; the final run of rejections starts at .75,
    # not at the isolated hit at .25.
    assert admitted[0][1] == pytest.approx(0.75)


def test_admit_family_corrects_across_the_dimension():
    """The correction sees the whole dimension, not one predictor's batch."""
    # Twelve candidates, each rejecting at full ablation with p = 0.01.
    rows = [[1.0, 0.01]] * 12
    uncorrected, _ = DirectedSimplicialComplex.admit_family(
        *_family(*rows), alpha=0.05, method="none"
    )
    bonferroni, _ = DirectedSimplicialComplex.admit_family(
        *_family(*rows), alpha=0.05, method="bonferroni"
    )
    assert len(uncorrected) == 12          # 0.01 < 0.05
    assert bonferroni == []                # 0.01 > 0.05 / 12


def test_admit_family_by_is_never_more_permissive_than_bh():
    rows = [[1.0, p] for p in (0.001, 0.006, 0.02, 0.04, 0.3, 0.8)]
    counts = {}
    for method in ("none", "bh", "by", "bonferroni"):
        admitted, _ = DirectedSimplicialComplex.admit_family(
            *_family(*rows), alpha=0.05, method=method
        )
        counts[method] = len(admitted)
    assert counts["none"] >= counts["bh"] >= counts["by"] >= counts["bonferroni"]


# ---------------------------------------------------------------------------
# The cutoff is a threshold, because it has two jobs: it decides membership at
# s = 1, and it dates the filtration everywhere else.
# ---------------------------------------------------------------------------

def test_family_cutoff_returns_the_critical_value_not_a_rejected_pvalue():
    """level * k / m, not p_(k).

    Both select the same members, but only the threshold means anything away
    from full ablation. Dating a simplex by the ablation at which its trajectory
    first drops below some *other* simplex's final p-value would make its
    filtration depend on an unrelated hypothesis.
    """
    # Six candidates pinned at the signed-rank floor for B = 32, two marginal,
    # the rest nowhere near -- the shape the handcrafted nets actually produce.
    p_values = np.array([2.0 ** -32] * 6 + [1.113e-3] * 2 + [0.5] * 48)
    m = p_values.size

    by = family_cutoff(p_values, alpha=0.01, method="by")
    bh = family_cutoff(p_values, alpha=0.01, method="bh")

    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    assert by == pytest.approx(0.01 / harmonic * 6 / m)
    assert bh == pytest.approx(0.01 * 8 / m)
    # Strictly above everything it rejects, which p_(k) never is.
    assert by > p_values.min()
    assert bh > p_values.min()


@pytest.mark.parametrize("method", ["bh", "by"])
def test_family_cutoff_selects_exactly_the_step_up_rejections(method):
    """Reporting the threshold must not move a single decision.

    k is the largest index with p_(k) <= t_k, and t_k < t_(k+1) < p_(k+1), so
    nothing lies between p_(k) and t_k and the two comparisons agree.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(2, 60))
        p_values = np.sort(rng.random(m) ** 4)
        level = 0.01
        if method == "by":
            level = 0.01 / np.sum(1.0 / np.arange(1, m + 1))
        thresholds = level * np.arange(1, m + 1) / m
        below = np.flatnonzero(p_values <= thresholds)
        expected = set(range(below[-1] + 1)) if below.size else set()

        cutoff = family_cutoff(p_values, alpha=0.01, method=method)
        assert set(np.flatnonzero(p_values <= cutoff)) == expected


def test_cached_builder_families_respect_the_inductive_construction():
    """A dimension's family is what the run posed, not what the cache holds.

    Construction is inductive, so dropping one edge silences every simplex that
    edge would have supported. Those are never tested, and anything reporting on
    the run has to see that: reading the cache grouped by dimension would credit
    the construction with hypotheses it never asked, and would disagree with the
    complex the same call actually built.
    """
    import itertools

    model = FourNeuronModel()
    n = model.internal_neurons
    strong = np.zeros(11)
    weak = np.full(11, 0.02)
    dropped_edge = (2, 3)

    records = []
    for source, target in itertools.permutations(range(n), 2):
        records.append({
            "predictor": (source,), "target": target, "dimension": 1,
            "min_pvalue": 0.0,
            "all_pvalues": weak if (source, target) == dropped_edge else strong,
        })
    for triple in itertools.permutations(range(n), 3):
        records.append({
            "predictor": triple[:2], "target": triple[2], "dimension": 2,
            "min_pvalue": 0.0, "all_pvalues": strong,
        })

    # alpha above the weak edge's p-value, then below it.
    permissive, posed_permissive = build_dsc_from_cached_pvalues(
        records, model, 2, 0.01, 0.05, "none", verbose=False, return_families=True
    )
    strict, posed_strict = build_dsc_from_cached_pvalues(
        records, model, 2, 0.01, 0.001, "none", verbose=False, return_families=True
    )

    assert dropped_edge in {tuple(s) for s, _ in permissive.complex[1]}
    assert dropped_edge not in {tuple(s) for s, _ in strict.complex[1]}

    posed = lambda entry: {tuple(s) for s, _, _ in entry.get(2, {"family": []})["family"]}
    wide, narrow = posed(posed_permissive), posed(posed_strict)
    assert narrow < wide, "dropping a face must silence simplices, not merely reject them"
    # Every silenced triangle is one that needed the dropped edge as a face.
    for simplex in wide - narrow:
        faces = DirectedSimplicialComplex.codimension_one_faces(simplex)
        assert dropped_edge in faces
    # And none of them made it into the complex either.
    assert not (wide - narrow) & {tuple(s) for s, _ in strict.complex.get(2, [])}


class FourNeuronModel:
    """Just enough model for the cached builder: it only reads the neuron count."""

    internal_neurons = 4
    layer_sizes = [1, 4, 1]

    def state_dict(self):
        return {}
