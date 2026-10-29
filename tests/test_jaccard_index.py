import json

import numpy as np
import pytest

from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.io.persistence import load_jaccard_data, save_jaccard_data
from neural_dsc.utils.jaccard_index import (
    JACCARD_FILTRATION_POINTS,
    JACCARD_METHOD_ID,
    compute_jaccard_distance,
    compute_jaccard_index,
    filtration_jaccard_distance,
)


def make_complex(edge_filtrations):
    complex_ = DirectedSimplicialComplex()
    complex_.complex = {
        0: [((0,), 0.0), ((1,), 0.0)],
        1: [(tuple(simplex), filtration) for simplex, filtration in edge_filtrations],
    }
    return complex_


def test_filtration_jaccard_resolves_birth_time_shifts_and_excludes_vertices():
    reference = make_complex([((0, 1), 0.20)])
    shifted = make_complex([((0, 1), 0.21)])

    assert JACCARD_FILTRATION_POINTS == 1001
    assert compute_jaccard_distance(reference, reference) == 0.0
    assert compute_jaccard_distance(reference, shifted) == pytest.approx(10 / 1001)
    assert compute_jaccard_index(reference, shifted) == pytest.approx(991 / 1001)

    vertices_only = make_complex([])
    vertices_only.complex[0].append(((2,), 0.0))
    assert compute_jaccard_distance(make_complex([]), vertices_only) == 0.0


def test_filtration_jaccard_is_a_metric_on_toy_filtrations():
    left = {("a",): 0.20, ("b",): 0.60}
    middle = {("a",): 0.30, ("b",): 0.60}
    right = {("a",): 0.40, ("c",): 0.70}

    left_middle = filtration_jaccard_distance(left, middle)
    middle_right = filtration_jaccard_distance(middle, right)
    left_right = filtration_jaccard_distance(left, right)

    assert filtration_jaccard_distance(left, left) == 0.0
    assert filtration_jaccard_distance(left, right) == pytest.approx(
        filtration_jaccard_distance(right, left)
    )
    assert left_right <= left_middle + middle_right + 1e-12


def test_jaccard_cache_records_method_and_rejects_legacy_data(tmp_path):
    path = tmp_path / "jaccard.json"
    matrix = np.array([[1.0, 0.25], [0.25, 1.0]])
    save_jaccard_data(matrix, {}, str(path))

    payload = json.loads(path.read_text())
    assert payload["method"]["id"] == JACCARD_METHOD_ID
    assert payload["method"]["quantity"] == "distance"
    assert payload["method"]["vertices_included"] is False
    loaded, by_dimension = load_jaccard_data(str(path))
    assert np.array_equal(loaded, matrix)
    assert by_dimension == {}

    payload["method"]["quantity"] = "similarity"
    path.write_text(json.dumps(payload))
    assert load_jaccard_data(str(path)) == (None, None)

    path.write_text(json.dumps({"matrix": matrix.tolist(), "by_dimension": {}}))
    assert load_jaccard_data(str(path)) == (None, None)
