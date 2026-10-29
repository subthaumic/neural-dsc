import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neural_dsc.mlp import ABLATION_TOLERANCE, MLP
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
import tempfile
import os


class TestMLP:
    def test_init(self):
        model = MLP([4, 3, 3, 2])
        assert len(model.hidden_layers) == 2
        assert model.hidden_layers[0].in_features == 4
        assert model.hidden_layers[0].out_features == 3
        assert model.hidden_layers[1].in_features == 3
        assert model.hidden_layers[1].out_features == 3
        assert model.output_layer.in_features == 3
        assert model.output_layer.out_features == 2

    def test_forward(self, model, sample_data):
        x = sample_data["X_test"][:5]
        output = model(x)
        assert output.shape == (5, 2)

        output, activations = model(x, return_activations=True)
        assert output.shape == (5, 2)
        assert 0 in activations and 1 in activations
        assert activations[0].shape == (5, 3)
        assert activations[1].shape == (5, 3)

    def test_fit_and_evaluate(self, model, sample_data):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Verify fit runs without errors
        model.fit(sample_data["train_loader"], criterion, optimizer, num_epochs=2)

        # Verify evaluate runs and returns accuracy
        accuracy = model.evaluate(sample_data["test_loader"])
        assert 0 <= accuracy[0] <= 1

    def test_get_flat_idx(self, model):
        # Test valid indices
        assert model.get_flat_idx((0, 0)) == 0
        assert model.get_flat_idx((0, 2)) == 2
        assert model.get_flat_idx((1, 0)) == 3
        assert model.get_flat_idx((1, 2)) == 5

        # Test out of bounds
        with pytest.raises(ValueError):
            model.get_flat_idx((2, 0))
        with pytest.raises(ValueError):
            model.get_flat_idx((0, 3))

    def test_get_unflat_idx(self, model):
        # Test valid indices
        assert model.get_unflat_idx(0) == (0, 0)
        assert model.get_unflat_idx(2) == (0, 2)
        assert model.get_unflat_idx(3) == (1, 0)
        assert model.get_unflat_idx(5) == (1, 2)

        # Test out of bounds
        with pytest.raises(ValueError):
            model.get_unflat_idx(6)

    def test_get_activations(self, model, sample_data):
        # Single input
        x = sample_data["X_test"][0]
        activations = model.get_activations(x)
        assert activations.shape == (1, 6)  # batch_size=1, total_neurons=6
        assert not activations.requires_grad

        # Batch input
        x_batch = torch.randn(5, 4)
        activations_batch = model.get_activations(x_batch)
        assert activations_batch.shape == (5, 6)  # batch_size=5, total_neurons=6
        assert not activations_batch.requires_grad

        # Test with detach=False
        activations_with_grad = model.get_activations(x, detach=False)
        assert activations_with_grad.shape == (1, 6)

    def test_ablation_trajectory(self, model, sample_data):
        x = sample_data["X_test"][0]
        n_steps = 40

        # Single neuron
        activations, report = model.ablation_trajectory(x, 0, n_steps=n_steps)
        assert activations.shape[0] == n_steps + 1  # ablation grid, s = 0 .. 1
        assert activations.shape[1] == 1            # batch size
        assert activations.shape[2] == 6            # total neurons
        assert set(report) >= {"ablated", "residual_fraction", "baseline"}

        # Multiple neurons
        activations, _ = model.ablation_trajectory(x, [0, 1], n_steps=n_steps)
        assert activations.shape[1] == 1
        assert activations.shape[2] == 6

        # Batch input
        x_batch = sample_data["X_test"][:3]
        activations, report = model.ablation_trajectory(x_batch, 0, n_steps=n_steps)
        assert activations.shape[1] == 3
        assert report["ablated"].shape == (3,)

    def test_invert(self, model, sample_data):
        # Create a target output
        x = sample_data["X_test"][0]
        y = model(x)

        # Test inversion
        y_inv = model.invert(y)
        assert y_inv.shape == (1, 4)

        # Check that the inverted input produces output close to target
        output = model(y_inv)
        assert torch.allclose(output, y.unsqueeze(0), atol=1)

    def test_generate_batch_from_target(self, model, sample_data):
        # Use model's output from a sample as the target
        x = sample_data["X_test"][0]
        target = model(x)

        # Generate batch
        batch = model.generate_batch_from_target(target, batch_size=5)
        assert batch.shape == (5, 4)

        # Check that outputs are close to target
        outputs = model(batch)
        assert outputs.shape == (5, 2)

        # Average distance should be small
        mean_distance = torch.mean(torch.norm(outputs - target, dim=1))
        assert mean_distance < 0.5  # Reasonable threshold

    def test_save_and_load(self, model, sample_data):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmpfile:
            model_path = tmpfile.name

        try:
            # Save the model
            model.save(model_path)
            assert os.path.exists(model_path)

            # Load the model
            loaded_model = torch.load(model_path, weights_only=False)
            assert isinstance(loaded_model, MLP)

            # Check that layer sizes are the same
            assert loaded_model.layer_sizes == model.layer_sizes

            # Check that state dicts are the same
            assert all(
                torch.equal(loaded_model.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict()
            )

            # Check forward pass consistency
            x_test = sample_data["X_test"][:5]
            original_output = model(x_test)
            loaded_output = loaded_model(x_test)
            assert torch.equal(original_output, loaded_output)

        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_copy(self, model, sample_data):
        copied_model = model.copy()

        # Check if it's a different object
        assert copied_model is not model

        # Check if it's an instance of MLP
        assert isinstance(copied_model, MLP)

        # Check layer sizes
        assert copied_model.layer_sizes == model.layer_sizes

        # Check state dicts are identical
        assert all(
            torch.equal(copied_model.state_dict()[key], model.state_dict()[key])
            for key in model.state_dict()
        )

        # Check that changes to one model don't affect the other
        # e.g., change a parameter in the copied model
        with torch.no_grad():
            if list(copied_model.parameters()):
                list(copied_model.parameters())[0][0, 0] = 999.9
                assert not torch.equal(
                    list(copied_model.parameters())[0], list(model.parameters())[0]
                )

        # Check forward pass consistency initially
        x_test = sample_data["X_test"][:5]
        original_output = model(x_test)
        copied_output_before_change = copied_model(
            x_test
        )  # This will be different if params changed

        # Re-verify original model's output hasn't changed due to copied_model modification
        original_output_after_copy_change = model(x_test)
        assert torch.equal(original_output, original_output_after_copy_change)


class TestActivationJacobian:
    """`activation_jacobian` must reproduce autograd exactly on ReLU nets."""

    @staticmethod
    def _autograd_jacobian(model, x):
        rows = []
        for i in range(x.shape[0]):
            rows.append(
                torch.autograd.functional.jacobian(
                    lambda s: model.get_activations(s.unsqueeze(0), detach=False).squeeze(0),
                    x[i],
                )
            )
        return torch.stack(rows)

    @pytest.mark.parametrize("hidden", [[3, 3], [5, 7, 6, 4, 3], [4]])
    def test_matches_autograd(self, hidden):
        torch.manual_seed(0)
        model = MLP([2] + hidden + [4])
        model.eval()
        x = torch.randn(16, 2)
        analytic = model.activation_jacobian(x)
        assert analytic.shape == (16, sum(hidden), 2)
        assert torch.allclose(analytic, self._autograd_jacobian(model, x), atol=1e-6)

    @pytest.mark.parametrize("factory", [create_quadrants_mono, create_quadrants_poly])
    def test_matches_autograd_bitwise_on_handcrafted_nets(self, factory):
        """
        On the handcrafted quadrant nets the two agree bit for bit.

        That is a property of those particular weights (0 and +/-1, exactly
        representable), not of the architecture: with random weights the chained
        matmuls associate differently than autograd's reverse accumulation and
        pick up float32 rounding of order 1e-8 -- a reduction-order artefact,
        not a difference in what is computed.
        """
        model = factory()
        model.eval()
        torch.manual_seed(0)
        x = torch.rand(16, 2) * 4 - 2
        assert torch.equal(model.activation_jacobian(x), self._autograd_jacobian(model, x))

    def test_relu_boundary_uses_subgradient_zero(self):
        """At a pre-activation of exactly 0, ReLU' is taken to be 0, as autograd does."""
        model = MLP([2, 3, 4])
        model.eval()
        with torch.no_grad():
            model.hidden_layers[0].weight.zero_()
            model.hidden_layers[0].bias.zero_()
        x = torch.randn(5, 2)
        assert torch.all(model.activation_jacobian(x) == 0)


class TestAblationTrajectory:
    """The one way trajectories are constructed, and the properties it must have.

    `ablation_trajectory` follows the minimum-norm descent that drives the
    predictor set to zero and reports it against the ablation

        s = 1 - sum_S a_i(x(t)) / sum_S a_i(x(0)),

    resampled onto a uniform grid in s. Everything the construction relies on
    follows from that parameterisation rather than from the integrator, and each
    of those consequences is checked here.
    """

    @staticmethod
    def _ablation_totals(activations, idxs):
        """sum_S a_i at each grid point, which is what the ablation is defined from."""
        return activations[:, :, idxs].clamp(min=0).sum(-1)

    @staticmethod
    def _ablation_deviation(model, x, idxs, n_steps, dt):
        """Worst departure from sum_S a_i(grid point k) = (1 - k/n_steps) * baseline.

        Measured only on probes that both have something to remove and reached
        the counterfactual: a probe whose predictors start at zero is frozen and
        has no ablation to be uniform in, and one the descent could not finish never
        got to s = 1.
        """
        activations, report = model.ablation_trajectory(x, idxs, n_steps=n_steps, dt=dt)
        totals = activations[:, :, idxs].clamp(min=0).sum(-1)
        baseline = totals[0]
        usable = report["ablated"] & (baseline > 1e-3)
        if not bool(usable.any()):
            return None
        ablation_grid = torch.linspace(0.0, 1.0, n_steps + 1).unsqueeze(1)
        return float((totals - (1.0 - ablation_grid) * baseline)[:, usable].abs().max())

    @pytest.mark.parametrize("idxs", [[0], [3], [0, 1], [0, 2]])
    def test_ablation_grid_is_uniform_up_to_the_integrator_step(self, idxs):
        """Grid point k must sit at s = k / n_steps, to the integrator's accuracy.

        This is the property the whole filtration rests on: the axis is the
        ablation, so index k says how much of the predictor set has been removed,
        not how much time has passed. It is not exact, and the reason is worth
        recording -- between two integrator states the predictor sum is linear
        in *time*, so resampling onto the ablation interpolates across a slight
        curve. The residual is therefore first order in dt and vanishes as the
        path is followed more finely, which is what is asserted here: a
        fourfold finer step cuts the deviation by roughly four. A resampling
        bug would leave a deviation that does not shrink.
        """
        torch.manual_seed(1)
        model = MLP([2, 4, 4, 4])
        model.eval()
        x = torch.rand(12, 2) * 4 - 2

        coarse = self._ablation_deviation(model, x, idxs, n_steps=40, dt=0.02)
        fine = self._ablation_deviation(model, x, idxs, n_steps=40, dt=0.005)
        if coarse is None or fine is None:
            pytest.skip("no probe both had something to remove and reached s = 1")
        # The slack keeps the single-predictor cases, which are exact to float32
        # and so cannot shrink any further, from failing a ratio test.
        assert fine < coarse / 3.0 + 1e-6
        assert fine < 5e-3

    def test_report_is_honest_about_an_infeasible_ablation(self):
        """More predictors than input dimensions cannot generally be zeroed.

        The descent lives in a 2-d input space, so three independent predictors
        over-determine it. The point is not that this fails -- it must -- but
        that the report says so instead of presenting the stopping point as the
        counterfactual.
        """
        torch.manual_seed(1)
        model = MLP([2, 4, 4, 4])
        model.eval()
        x = torch.rand(12, 2) * 4 - 2
        _, report = model.ablation_trajectory(x, [1, 4, 5], n_steps=20)
        stopped_short = ~report["ablated"]
        assert bool(stopped_short.any())
        # A probe reported as not ablated has residual left, and vice versa.
        assert (report["residual_fraction"][stopped_short] > ABLATION_TOLERANCE).all()

    def test_ablation_grid_refines_without_moving(self):
        """A coarse grid is exactly a subsample of a fine one.

        Membership is decided at s = 1 and every grid contains s = 1, so the
        construction must not depend on `n_steps` at all; the filtration values
        may only be located more precisely. That holds because the trajectory is
        resampled onto the ablation, so refining the grid adds points between the
        existing ones rather than moving them.
        """
        torch.manual_seed(1)
        model = MLP([2, 4, 4, 4])
        model.eval()
        x = torch.rand(10, 2) * 4 - 2
        coarse, _ = model.ablation_trajectory(x, [0, 1], n_steps=20)
        fine, _ = model.ablation_trajectory(x, [0, 1], n_steps=40)
        assert torch.equal(coarse, fine[::2])

    @pytest.mark.parametrize("dt", [0.02, 0.005])
    def test_endpoint_does_not_depend_on_the_integrator_step(self, dt):
        """dt sets accuracy, not destination.

        The counterfactual is where the predictors are zero; dt only decides how
        accurately the path there is followed. This is the property the exact
        kink landing bought: overshooting by up to dt used to switch on the
        anti-parallel partner of a predictor, giving that partner an activation
        proportional to dt and manufacturing an edge from the step size.
        """
        torch.manual_seed(1)
        model = MLP([2, 4, 4, 4])
        model.eval()
        x = torch.rand(10, 2) * 4 - 2
        reference, _ = model.ablation_trajectory(x, [0, 1], n_steps=20, dt=0.01)
        other, _ = model.ablation_trajectory(x, [0, 1], n_steps=20, dt=dt)
        assert (reference[-1] - other[-1]).abs().max() < 1e-4

    def test_samples_are_independent(self):
        """Batching must not couple probes: one probe's trajectory is unchanged
        by which other probes share the batch."""
        torch.manual_seed(7)
        model = MLP([2, 4, 4, 4])
        model.eval()
        x = torch.rand(10, 2) * 4 - 2
        full, _ = model.ablation_trajectory(x.clone(), [0, 1], n_steps=20)
        for i in (0, 4, 9):
            alone, _ = model.ablation_trajectory(x[i : i + 1].clone(), [0, 1], n_steps=20)
            assert torch.allclose(full[:, i : i + 1], alone, atol=1e-6)

    def test_predictor_inactive_from_the_start(self):
        """A probe whose predictors are already off is frozen, and already done.

        There is nothing to remove, so the trajectory is constant and the
        counterfactual is the baseline -- which the report must call ablated
        rather than stalled.
        """
        model = MLP([2, 3, 4])
        model.eval()
        with torch.no_grad():
            model.hidden_layers[0].weight.zero_()
            model.hidden_layers[0].bias.fill_(-1.0)
        x = torch.rand(8, 2)
        trajectory, report = model.ablation_trajectory(x, [0, 1], n_steps=10)
        assert torch.all(trajectory == trajectory[0])
        assert bool(report["ablated"].all())
        assert not bool(report["stalled"].any())

    def test_relu_boundary_input(self):
        """The origin puts every unit exactly on its kink, which must resolve."""
        torch.manual_seed(5)
        model = MLP([2, 4, 4, 4])
        model.eval()
        trajectory, _ = model.ablation_trajectory(torch.zeros(4, 2), [0, 1], n_steps=10)
        assert torch.isfinite(trajectory).all()

    def test_single_sample_batch(self):
        torch.manual_seed(4)
        model = MLP([2, 4, 4, 4])
        model.eval()
        trajectory, report = model.ablation_trajectory(
            torch.rand(1, 2) * 4 - 2, [0, 1], n_steps=20
        )
        assert trajectory.shape[1] == 1
        assert report["ablated"].shape == (1,)
