import numpy as np
import pytest
from neural_dsc.utils.pearl_causality import (
    pearl_causal_test,
    relevance_epsilon,
    relevance_test_pvalue,
)


class TestPearlCausality:
    def test_pearl_causal_test_single_neurons(self, model, sample_data):
        # Basic functionality test (batch, single predictor)
        probe_data = sample_data["X_test"][:5]
        predictor_neurons = 0
        target_neuron = 5

        t_values, significance_flags, p_values = pearl_causal_test(
            predictor_neurons, target_neuron, model, probe_data
        )

        assert isinstance(t_values, np.ndarray)
        assert isinstance(significance_flags, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert len(t_values) == len(significance_flags) == len(p_values)
        assert significance_flags.dtype == bool
        assert p_values.dtype == float

    def test_pearl_causal_test_batch_predictors(self, model, sample_data):
        # Advanced functionality test (batch, multiple predictors)
        probe_data = sample_data["X_test"][:5]
        predictor_neurons = [0, 1]
        target_neuron = 5

        t_values, significance_flags, p_values = pearl_causal_test(
            predictor_neurons, target_neuron, model, probe_data
        )

        assert isinstance(t_values, np.ndarray)
        assert isinstance(significance_flags, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert len(t_values) == len(significance_flags) == len(p_values)
        assert significance_flags.dtype == bool
        assert p_values.dtype == float

    def test_pearl_causal_test_batch_targets(self, model, sample_data):
        """Test the batch target functionality of pearl_causal_test."""
        probe_data = sample_data["X_test"][:5]
        # Test with a list of predictors and a list of target neurons
        predictor_neurons = 0
        target_neurons = [3, 4, 5]

        results = pearl_causal_test(
            predictor_neurons, target_neurons, model, probe_data
        )

        # Verify results is a dictionary with the right keys
        assert isinstance(results, dict)
        assert set(results.keys()) == set(target_neurons)

        # Verify each value in the dictionary has the right format
        for target, (t_vals, sig_flags, p_vals) in results.items():
            assert isinstance(t_vals, np.ndarray)
            assert isinstance(sig_flags, np.ndarray)
            assert isinstance(p_vals, np.ndarray)
            assert len(t_vals) == len(sig_flags) == len(p_vals)
            assert sig_flags.dtype == bool
            assert p_vals.dtype == float

    def test_pearl_causal_test_batch_neurons(self, model, sample_data):
        """Test the batch target functionality of pearl_causal_test."""
        probe_data = sample_data["X_test"][:5]
        # Test with a list of predictors and a list of target neurons
        predictor_neurons = [0, 1]
        target_neurons = [3, 4, 5]

        results = pearl_causal_test(
            predictor_neurons, target_neurons, model, probe_data
        )

        # Verify results is a dictionary with the right keys
        assert isinstance(results, dict)
        assert set(results.keys()) == set(target_neurons)

        # Verify each value in the dictionary has the right format
        for target, (t_vals, sig_flags, p_vals) in results.items():
            assert isinstance(t_vals, np.ndarray)
            assert isinstance(sig_flags, np.ndarray)
            assert isinstance(p_vals, np.ndarray)
            assert len(t_vals) == len(sig_flags) == len(p_vals)
            assert sig_flags.dtype == bool
            assert p_vals.dtype == float

    def test_pearl_causal_test_single_sample(self, model, sample_data):
        # Advanced functionality test (single sample, single predictor)
        probe_data = sample_data["X_test"][0]
        predictor_neurons = 0
        target_neuron = 5

        t_values, significance_flags, p_values = pearl_causal_test(
            predictor_neurons, target_neuron, model, probe_data
        )

        assert isinstance(t_values, np.ndarray)
        assert isinstance(significance_flags, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert len(t_values) == len(significance_flags) == len(p_values)
        assert significance_flags.dtype == bool
        assert p_values.dtype == float

    def test_pearl_causal_test_custom_params(self, model, sample_data):
        # Test with custom parameters
        probe_data = sample_data["X_test"][:5]
        predictor_neurons = [2, 3]
        target_neuron = 5

        t_values, significance_flags, p_values = pearl_causal_test(
            predictor_neurons,
            target_neuron,
            model,
            probe_data,
            max_iterations=1,
            significance_level=0.01,
        )

        assert isinstance(t_values, np.ndarray)
        assert isinstance(significance_flags, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert len(t_values) == len(significance_flags) == len(p_values)
        # Check if time parameter was respected (approximately)
        assert np.max(t_values) <= 5

    def test_pearl_causal_test_single_vs_batch_neurons(self, model, sample_data):
        # Test that passing single int vs a list with one int gives consistent results.
        probe_data = sample_data["X_test"][:5]
        single_predictor = 0
        list_predictor = [0]
        single_target = 5
        list_target = [5]

        # Test with int target
        t_values1, significance_flags1, p_values1 = pearl_causal_test(
            single_predictor, single_target, model, probe_data
        )

        # Test with list containing single int
        results = pearl_causal_test(list_predictor, list_target, model, probe_data)
        t_values2, significance_flags2, p_values2 = results[5]

        # Results should be identical
        np.testing.assert_array_equal(t_values1, t_values2)
        np.testing.assert_array_equal(significance_flags1, significance_flags2)
        np.testing.assert_array_equal(p_values1, p_values2)

    def test_pearl_causal_test_empty_targets(self, model, sample_data):
        """Test with empty target list."""
        probe_data = sample_data["X_test"][:5]
        predictor_neurons = [0]

        # Empty target list should return empty dictionary
        results = pearl_causal_test(predictor_neurons, [], model, probe_data)
        assert isinstance(results, dict)
        assert len(results) == 0


class TestRelevanceScale:
    """
    The relevance margin must stay class-conditional: one global margin,
    multiplied by each target neuron's own SD over *this* class's probe set.
    """

    def _acts(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        # neuron 0 wide, neuron 1 narrow, neuron 2 silent for this class
        return np.stack([
            rng.normal(3.0, 2.0, n),
            rng.normal(0.5, 0.25, n),
            np.zeros(n),
        ], axis=1)

    def test_epsilon_is_margin_times_own_class_sd(self):
        acts = self._acts()
        margin = 0.25
        for q in (0, 1):
            eps = relevance_epsilon(acts[:, q], acts, margin, "std")
            assert eps == pytest.approx(margin * acts[:, q].std())

    def test_epsilon_tracks_the_conditional_distribution(self):
        # Same neuron, two different class-conditional spreads -> two epsilons.
        margin = 0.25
        wide, narrow = self._acts(seed=1), self._acts(seed=2)
        narrow[:, 0] *= 0.1
        eps_wide = relevance_epsilon(wide[:, 0], wide, margin, "std")
        eps_narrow = relevance_epsilon(narrow[:, 0], narrow, margin, "std")
        assert eps_narrow < eps_wide

    def test_silent_neuron_gets_a_positive_class_conditional_floor(self):
        # A neuron silent for this class must not fall back to the point null,
        # and its floor must come from within-class dispersion.
        acts = self._acts()
        eps = relevance_epsilon(acts[:, 2], acts, 0.25, "std")
        assert eps > 0
        assert eps == pytest.approx(0.25 * 0.05 * np.std(acts, axis=0).mean())

    def test_zero_margin_disables_relevance_testing(self):
        acts = self._acts()
        assert relevance_epsilon(acts[:, 0], acts, 0.0, "std") == 0.0

    def test_relevance_test_ignores_tiny_effects_at_any_sample_size(self):
        # The property the interval null exists to provide: a consistent but
        # negligible effect must not become significant as B grows.
        rng = np.random.default_rng(3)
        for n in (30, 100, 400, 2000):
            delta = 0.002 + 0.001 * rng.standard_normal(n)
            assert relevance_test_pvalue(delta, 0.0) < 1e-6      # point null fires
            assert relevance_test_pvalue(delta, 0.05) > 0.5      # relevance does not

    def test_relevance_test_still_detects_real_effects(self):
        rng = np.random.default_rng(4)
        delta = 0.5 + 0.1 * rng.standard_normal(100)
        assert relevance_test_pvalue(delta, 0.05) < 1e-6
        assert relevance_test_pvalue(-delta, 0.05) < 1e-6        # two-sided

    def test_unknown_scale_mode_rejected(self):
        acts = self._acts()
        with pytest.raises(ValueError):
            relevance_epsilon(acts[:, 0], acts, 0.25, "not_a_mode")
