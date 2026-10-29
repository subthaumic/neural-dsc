"""
Unit tests for DSC analysis utilities.

Tests cover:
- compute_simplex_statistics
- analyze_neuron_participation
- analyze_directedness
- get_filtration_distribution
- format_simplex_summary
- print_dsc_summary
- save_dsc_statistics_csv
"""

import pytest
import os
import tempfile
import csv
from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.utils.dsc_analysis import (
    compute_simplex_statistics,
    analyze_neuron_participation,
    analyze_directedness,
    get_filtration_distribution,
    format_simplex_summary,
    print_dsc_summary,
    save_dsc_statistics_csv,
)


@pytest.fixture
def empty_dsc():
    """Create an empty DSC."""
    dsc = DirectedSimplicialComplex()
    dsc.complex = {}
    return dsc


@pytest.fixture
def simple_dsc():
    """Create a simple DSC with known structure for testing."""
    dsc = DirectedSimplicialComplex()
    # Manually construct a simple complex:
    # - 4 vertices (0-simplices): neurons 0, 1, 2, 3
    # - 3 edges (1-simplices): (0,1), (1,2), (0,2)
    # - 1 triangle (2-simplex): (0,1,2)
    dsc.complex = {
        0: [
            ((0,), 0.0),
            ((1,), 0.0),
            ((2,), 0.0),
            ((3,), 0.0),
        ],
        1: [
            ((0, 1), 0.1),
            ((1, 2), 0.2),
            ((0, 2), 0.3),
        ],
        2: [
            ((0, 1, 2), 0.5),
        ],
    }
    return dsc


@pytest.fixture
def multi_dim_dsc():
    """Create a DSC with multiple dimensions and varied filtration values."""
    dsc = DirectedSimplicialComplex()
    dsc.complex = {
        0: [
            ((0,), 0.0),
            ((1,), 0.0),
            ((2,), 0.0),
            ((3,), 0.0),
            ((4,), 0.0),
            ((5,), 0.0),
        ],
        1: [
            ((0, 1), 0.1),
            ((1, 2), 0.15),
            ((2, 3), 0.2),
            ((3, 4), 0.25),
            ((4, 5), 0.3),
            ((0, 2), 0.35),
        ],
        2: [
            ((0, 1, 2), 0.4),
            ((2, 3, 4), 0.5),
        ],
        3: [
            ((0, 1, 2, 3), 0.8),
        ],
    }
    return dsc


class TestComputeSimplexStatistics:
    """Tests for compute_simplex_statistics function."""

    def test_empty_dsc(self, empty_dsc):
        """Test with empty DSC."""
        stats = compute_simplex_statistics(empty_dsc)
        assert stats['total_simplices'] == 0
        assert stats['max_dimension'] == 0
        assert stats['total_vertices'] == 0
        assert stats['counts_by_dimension'] == {}

    def test_simple_dsc(self, simple_dsc):
        """Test with simple DSC."""
        stats = compute_simplex_statistics(simple_dsc)

        assert stats['total_simplices'] == 8  # 4 + 3 + 1
        assert stats['max_dimension'] == 2
        assert stats['total_vertices'] == 4  # neurons 0, 1, 2, 3
        assert stats['counts_by_dimension'] == {0: 4, 1: 3, 2: 1}

    def test_filtration_stats(self, simple_dsc):
        """Test filtration statistics computation."""
        stats = compute_simplex_statistics(simple_dsc)

        # Check dimension 1 filtration stats
        assert 1 in stats['filtration_stats_by_dimension']
        filt_stats = stats['filtration_stats_by_dimension'][1]
        assert filt_stats['min'] == 0.1
        assert filt_stats['max'] == 0.3
        assert filt_stats['count'] == 3
        # Median of [0.1, 0.2, 0.3] is 0.2
        assert filt_stats['median'] == 0.2

    def test_multi_dimension(self, multi_dim_dsc):
        """Test with multi-dimensional DSC."""
        stats = compute_simplex_statistics(multi_dim_dsc)

        assert stats['max_dimension'] == 3
        assert stats['counts_by_dimension'] == {0: 6, 1: 6, 2: 2, 3: 1}
        assert stats['total_simplices'] == 15


class TestAnalyzeNeuronParticipation:
    """Tests for analyze_neuron_participation function."""

    def test_simple_participation(self, simple_dsc):
        """Test neuron participation with simple DSC."""
        layer_sizes = [2, 2]  # Two layers of 2 neurons each
        participation = analyze_neuron_participation(simple_dsc, layer_sizes)

        # Neuron 0 appears in: (0,), (0,1), (0,2), (0,1,2) = 4 times
        # Neuron 1 appears in: (1,), (0,1), (1,2), (0,1,2) = 4 times
        # Neuron 2 appears in: (2,), (1,2), (0,2), (0,1,2) = 4 times
        # Neuron 3 appears in: (3,) = 1 time
        assert participation['neuron_counts'][0] == 4
        assert participation['neuron_counts'][1] == 4
        assert participation['neuron_counts'][2] == 4
        assert participation['neuron_counts'][3] == 1

    def test_neuron_ranking(self, simple_dsc):
        """Test neuron ranking is sorted correctly."""
        layer_sizes = [2, 2]
        participation = analyze_neuron_participation(simple_dsc, layer_sizes)

        ranking = participation['neuron_ranking']
        # Should be sorted by count (descending)
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_layer_participation(self, simple_dsc):
        """Test layer-wise participation breakdown."""
        layer_sizes = [2, 2]  # Layer 0: neurons 0,1; Layer 1: neurons 2,3
        participation = analyze_neuron_participation(simple_dsc, layer_sizes)

        # Layer 0 (neurons 0, 1): 4 + 4 = 8
        # Layer 1 (neurons 2, 3): 4 + 1 = 5
        assert participation['layer_participation'][0] == 8
        assert participation['layer_participation'][1] == 5

    def test_top_neurons(self, simple_dsc):
        """Test top_neurons returns correct number."""
        layer_sizes = [2, 2]
        participation = analyze_neuron_participation(simple_dsc, layer_sizes)

        assert len(participation['top_neurons']) <= 10


class TestAnalyzeDirectedness:
    """Tests for analyze_directedness function."""

    def test_simple_directedness(self, simple_dsc):
        """Test directedness analysis."""
        directedness = analyze_directedness(simple_dsc)

        # Should have vertex sets for edges and triangles
        assert 'vertex_set_stats' in directedness
        assert 'asymmetry_examples' in directedness
        assert 'fully_directed_count' in directedness
        assert 'partially_directed_count' in directedness

    def test_asymmetry_detection(self):
        """Test detection of asymmetric edges."""
        dsc = DirectedSimplicialComplex()
        # Create asymmetric edges: (0,1) exists but not (1,0)
        dsc.complex = {
            0: [((0,), 0.0), ((1,), 0.0)],
            1: [((0, 1), 0.1)],  # Only one direction
        }

        directedness = analyze_directedness(dsc)

        # Should detect asymmetry
        assert len(directedness['asymmetry_examples']) > 0
        assert directedness['partially_directed_count'] > 0

    def test_symmetric_edges(self):
        """Test with symmetric (fully directed) edges."""
        dsc = DirectedSimplicialComplex()
        # Both directions exist
        dsc.complex = {
            0: [((0,), 0.0), ((1,), 0.0)],
            1: [((0, 1), 0.1), ((1, 0), 0.2)],
        }

        directedness = analyze_directedness(dsc)

        # No asymmetry for this vertex set
        assert directedness['fully_directed_count'] == 1


class TestGetFiltrationDistribution:
    """Tests for get_filtration_distribution function."""

    def test_empty_dsc(self, empty_dsc):
        """Test with empty DSC."""
        dist = get_filtration_distribution(empty_dsc)

        assert dist['all_filtrations'] == []
        assert dist['quartiles']['median'] == 0

    def test_simple_distribution(self, simple_dsc):
        """Test filtration distribution with simple DSC."""
        dist = get_filtration_distribution(simple_dsc)

        # Should have all filtration values
        assert len(dist['all_filtrations']) == 8  # 4 + 3 + 1 simplices

        # Check by_dimension
        assert 0 in dist['by_dimension']
        assert 1 in dist['by_dimension']
        assert 2 in dist['by_dimension']

        # Dimension 0 should all be 0.0
        assert all(f == 0.0 for f in dist['by_dimension'][0])

    def test_quartiles(self, multi_dim_dsc):
        """Test quartile computation."""
        dist = get_filtration_distribution(multi_dim_dsc)

        quartiles = dist['quartiles']
        assert 'q25' in quartiles
        assert 'median' in quartiles
        assert 'q75' in quartiles
        assert 'min' in quartiles
        assert 'max' in quartiles

        # Check ordering
        assert quartiles['min'] <= quartiles['q25']
        assert quartiles['q25'] <= quartiles['median']
        assert quartiles['median'] <= quartiles['q75']
        assert quartiles['q75'] <= quartiles['max']

    def test_histogram(self, multi_dim_dsc):
        """Test histogram computation."""
        dist = get_filtration_distribution(multi_dim_dsc)

        assert 'histogram' in dist
        assert 'counts' in dist['histogram']
        assert 'bins' in dist['histogram']
        assert len(dist['histogram']['bins']) == len(dist['histogram']['counts']) + 1


class TestFormatSimplexSummary:
    """Tests for format_simplex_summary function."""

    def test_summary_format(self, simple_dsc):
        """Test summary is a properly formatted string."""
        summary = format_simplex_summary(simple_dsc, "Test DSC")

        assert isinstance(summary, str)
        assert "Test DSC" in summary
        assert "Total simplices" in summary
        assert "Dimension" in summary

    def test_empty_dsc_summary(self, empty_dsc):
        """Test summary for empty DSC."""
        summary = format_simplex_summary(empty_dsc, "Empty DSC")

        assert isinstance(summary, str)
        assert "Empty DSC" in summary

    def test_top_k_parameter(self, multi_dim_dsc):
        """Test top_k parameter limits output."""
        summary = format_simplex_summary(multi_dim_dsc, "Test", top_k=2)

        # Should still produce valid output
        assert isinstance(summary, str)


class TestPrintDscSummary:
    """Tests for print_dsc_summary function."""

    def test_prints_without_error(self, simple_dsc, capsys):
        """Test that print_dsc_summary runs without error."""
        print_dsc_summary(simple_dsc, "Test DSC")

        captured = capsys.readouterr()
        assert "Test DSC" in captured.out
        assert "Total simplices" in captured.out

    def test_empty_dsc_print(self, empty_dsc, capsys):
        """Test printing empty DSC."""
        print_dsc_summary(empty_dsc, "Empty")

        captured = capsys.readouterr()
        assert "Empty" in captured.out


class TestSaveDscStatisticsCsv:
    """Tests for save_dsc_statistics_csv function."""

    def test_save_single_dsc(self, simple_dsc):
        """Test saving statistics for single DSC."""
        stats = compute_simplex_statistics(simple_dsc)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            save_dsc_statistics_csv([stats], filepath, labels=["Test DSC"])

            # Verify file exists and has content
            assert os.path.exists(filepath)

            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]['label'] == "Test DSC"
            assert int(rows[0]['total_simplices']) == 8
        finally:
            os.unlink(filepath)

    def test_save_multiple_dscs(self, simple_dsc, multi_dim_dsc):
        """Test saving statistics for multiple DSCs."""
        stats1 = compute_simplex_statistics(simple_dsc)
        stats2 = compute_simplex_statistics(multi_dim_dsc)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            save_dsc_statistics_csv([stats1, stats2], filepath,
                                    labels=["Simple", "Multi-dim"])

            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]['label'] == "Simple"
            assert rows[1]['label'] == "Multi-dim"
        finally:
            os.unlink(filepath)

    def test_empty_stats_list(self):
        """Test with empty stats list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            save_dsc_statistics_csv([], filepath)
            # Should not create file or should create empty file
            # The function returns early, so file shouldn't have CSV content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_csv_has_correct_columns(self, simple_dsc):
        """Test CSV has expected columns."""
        stats = compute_simplex_statistics(simple_dsc)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            save_dsc_statistics_csv([stats], filepath, labels=["Test"])

            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

            # Check required columns
            assert 'label' in headers
            assert 'total_simplices' in headers
            assert 'max_dimension' in headers
            assert 'total_vertices' in headers

            # Check dimension-specific columns exist
            assert any('dim0_count' in h for h in headers)
            assert any('dim1_count' in h for h in headers)
        finally:
            os.unlink(filepath)

    def test_auto_labels(self, simple_dsc):
        """Test automatic label generation when labels not provided."""
        stats = compute_simplex_statistics(simple_dsc)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            save_dsc_statistics_csv([stats], filepath)  # No labels

            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]['label'] == "DSC_0"  # Auto-generated
        finally:
            os.unlink(filepath)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_analysis_pipeline(self, multi_dim_dsc):
        """Test running full analysis pipeline."""
        # Compute statistics
        stats = compute_simplex_statistics(multi_dim_dsc)

        # Analyze participation
        layer_sizes = [3, 3]  # 6 neurons total
        participation = analyze_neuron_participation(multi_dim_dsc, layer_sizes)

        # Analyze directedness
        directedness = analyze_directedness(multi_dim_dsc)

        # Get filtration distribution
        filtration = get_filtration_distribution(multi_dim_dsc)

        # Format summary
        summary = format_simplex_summary(multi_dim_dsc, "Integration Test")

        # All should work together
        assert stats['total_simplices'] > 0
        assert len(participation['neuron_counts']) > 0
        assert 'vertex_set_stats' in directedness
        assert len(filtration['all_filtrations']) > 0
        assert len(summary) > 0

    def test_save_and_verify(self, simple_dsc, multi_dim_dsc):
        """Test saving multiple DSCs and verifying content."""
        stats = [
            compute_simplex_statistics(simple_dsc),
            compute_simplex_statistics(multi_dim_dsc),
        ]
        labels = ["Simple", "Complex"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            save_dsc_statistics_csv(stats, filepath, labels=labels)

            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Verify data integrity
            assert int(rows[0]['total_simplices']) == 8   # simple_dsc
            assert int(rows[1]['total_simplices']) == 15  # multi_dim_dsc
            assert int(rows[0]['max_dimension']) == 2
            assert int(rows[1]['max_dimension']) == 3
        finally:
            os.unlink(filepath)
