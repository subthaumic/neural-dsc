import pytest
import os
import tempfile
from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex


def _simplex_sets(complex_obj):
    return {
        dim: {simplex for simplex, _ in simplex_list}
        for dim, simplex_list in complex_obj.complex.items()
    }


def _filtration_map(complex_obj):
    return {
        dim: {simplex: filtration for simplex, filtration in simplex_list}
        for dim, simplex_list in complex_obj.complex.items()
    }


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_build_directed_simplicial_complex_basic(model, sample_data):
    """Test basic functionality of the simplicial complex builder."""
    # Use a small batch of data for testing
    probe_data = sample_data["X_test"][:5]
    max_dim = 1

    # Create an instance and build with minimal parameters and small max_dim
    complex = DirectedSimplicialComplex()
    complex.build(
        model=model,
        probe_data=probe_data,
        max_dimension=max_dim,
        significance_level=0.5,  # Make it more likely to find causal relationships
    )

    # Test structure and basic properties
    assert hasattr(complex, "complex"), "Instance should have a complex attribute"
    assert isinstance(complex.complex, dict), "Complex should be a dictionary"
    assert 0 in complex.complex, "Should contain 0-simplices"
    assert len(complex.complex[0]) > 0, "Should have at least one 0-simplex"

    # Should not have simplices of dimension greater than max_dim
    assert max(complex.complex.keys()) <= max_dim, "Should not exceed max dimension"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_build_directed_simplicial_complex_properties(model, sample_data):
    """Test properties of the simplicial complex."""
    # Use a small batch of data
    probe_data = sample_data["X_test"][:5]

    complex = DirectedSimplicialComplex()
    complex.build(
        model=model, probe_data=probe_data, max_dimension=2, significance_level=0.1
    )

    # Check structure of 0-simplices
    for simplex_tuple in complex.complex[0]:
        assert len(simplex_tuple) == 2, "Each entry should be (simplex, filtration)"
        simplex, filtration = simplex_tuple
        assert isinstance(simplex, tuple), "Simplex should be a tuple"
        assert len(simplex) == 1, "0-simplex should contain one vertex"
        assert filtration == 0, "0-simplices should have filtration value 0"

    # Check if any 1-simplices were created
    if 1 in complex.complex and complex.complex[1]:
        # Test if appropriate faces exist
        for simplex_tuple in complex.complex[1]:
            simplex, _ = simplex_tuple
            assert len(simplex) == 2, "1-simplex should have 2 vertices"

            # Both vertices should exist as 0-simplices
            vertices_0d = [s[0][0] for s in complex.complex[0]]
            assert simplex[0] in vertices_0d, "Source vertex should exist as 0-simplex"
            assert simplex[1] in vertices_0d, "Target vertex should exist as 0-simplex"

    # Check if any 2-simplices were created
    if 2 in complex.complex and complex.complex[2]:
        # Test if appropriate faces exist
        for simplex_tuple in complex.complex[2]:
            simplex, _ = simplex_tuple
            assert len(simplex) == 3, "2-simplex should have 3 vertices"

            # Check that all faces exist
            one_simplices = [s[0] for s in complex.complex[1]]
            faces = [
                (simplex[0], simplex[1]),
                (simplex[0], simplex[2]),
                (simplex[1], simplex[2]),
            ]
            for face in faces:
                assert any(
                    all(v1 == v2 for v1, v2 in zip(face, existing))
                    for existing in one_simplices
                ), f"Face {face} should exist"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_empty_complex_with_high_significance(model, sample_data):
    """Test that for very high significance levels, we get an empty complex."""
    probe_data = sample_data["X_test"][:5]

    complex = DirectedSimplicialComplex()
    complex.build(
        model=model,
        probe_data=probe_data,
        max_dimension=2,
        significance_level=1e-9,  # Very high significance threshold
    )

    # 0-simplices should always exist
    assert 0 in complex.complex
    assert len(complex.complex[0]) > 0

    # Expect that no 1-simplices exist with such a high threshold
    assert (
        1 not in complex.complex or not complex.complex[1]
    ), "No 1-simplices should exist, given the high significance level"
    assert (
        2 not in complex.complex or not complex.complex[2]
    ), "If no 1-simplices, there should be no 2-simplices"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_and_load(model, sample_data):
    """Test saving and loading a simplicial complex."""
    probe_data = sample_data["X_test"][:5]

    # Create and build a complex
    complex = DirectedSimplicialComplex()
    complex.build(
        model=model,
        probe_data=probe_data,
        max_dimension=1,
        significance_level=0.5,
    )

    # Test saving and loading text format
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save to temp file
        complex.save(tmp_path, format="text")
        assert os.path.exists(tmp_path), "File should be created"

        # Load back
        loaded_complex = DirectedSimplicialComplex()
        loaded_complex.load(tmp_path, format="text")

        # Check if structure matches
        assert set(complex.complex.keys()) == set(
            loaded_complex.complex.keys()
        ), "Dimensions should match"
        for dim in complex.complex:
            assert len(complex.complex[dim]) == len(
                loaded_complex.complex[dim]
            ), f"Number of {dim}-simplices should match"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Test saving and loading pickle format
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save to temp file
        complex.save(tmp_path, format="pickle")
        assert os.path.exists(tmp_path), "File should be created"

        # Load back
        loaded_complex = DirectedSimplicialComplex()
        loaded_complex.load(tmp_path, format="pickle")

        # Check if structure matches exactly (pickle should preserve everything)
        assert (
            complex.complex == loaded_complex.complex
        ), "Complex dictionaries should be identical"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_collect_pvalues_parallel(model, sample_data):
    """Ensure p-value collection works in the parallel path."""
    probe_data = sample_data["X_test"][:4]
    max_iterations = 10

    complex = DirectedSimplicialComplex()
    build_result = complex.build(
        model=model,
        probe_data=probe_data,
        max_dimension=1,
        max_iterations=max_iterations,
        significance_level=0.5,
        n_workers=2,
        collect_pvalues=True,
    )

    assert isinstance(build_result, tuple), "Expected (complex, pvalues) return"
    _, pvalues = build_result

    n_vertices = model.internal_neurons
    expected_count = n_vertices * (n_vertices - 1)
    assert len(pvalues) == expected_count, "Expected all candidate edges to be evaluated"
    for pv in pvalues:
        assert pv["dimension"] == 1
        assert len(pv["all_pvalues"]) == max_iterations + 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parallel_vs_sequential_consistency(model, sample_data):
    """Parallel and sequential builds should produce the same simplex sets."""
    probe_data = sample_data["X_test"][:4]
    max_iterations = 10

    complex_seq = DirectedSimplicialComplex()
    complex_seq.build(
        model=model,
        probe_data=probe_data,
        max_dimension=1,
        max_iterations=max_iterations,
        significance_level=0.5,
        n_workers=1,
    )

    complex_par = DirectedSimplicialComplex()
    complex_par.build(
        model=model,
        probe_data=probe_data,
        max_dimension=1,
        max_iterations=max_iterations,
        significance_level=0.5,
        n_workers=2,
    )

    assert _simplex_sets(complex_seq) == _simplex_sets(complex_par)

    seq_filtration = _filtration_map(complex_seq)
    par_filtration = _filtration_map(complex_par)
    for dim, simplex_map in seq_filtration.items():
        assert simplex_map.keys() == par_filtration[dim].keys()
        for simplex, filt in simplex_map.items():
            assert abs(filt - par_filtration[dim][simplex]) < 1e-6
