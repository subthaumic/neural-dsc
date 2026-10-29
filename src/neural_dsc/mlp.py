import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Relative activation left on a predictor below which the counterfactual counts
# as ablated. See `MLP.ablation_trajectory` for how the value was chosen.
ABLATION_TOLERANCE = 1e-5


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model.
    """

    def __init__(self, layer_sizes):
        """
        Initialize a multi-layer perceptron model.

        Args:
            layer_sizes (list): List of integers representing the sizes of each layer.
                               First element is the input size, last is the output size,
                               and all elements in between are hidden layer sizes.
        """
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output layer sizes must be provided")

        self.hidden_layers = nn.ModuleList()
        self.activation = nn.ReLU()

        # Extract input_size, hidden_sizes, and output_size
        self.layer_sizes = layer_sizes

        self.input_size = layer_sizes[0]
        self.hidden_sizes = layer_sizes[1:-1]
        self.output_size = layer_sizes[-1]

        self.total_neurons = sum(layer_sizes)
        self.internal_neurons = sum(self.hidden_sizes)

        if self.hidden_sizes:
            # First hidden layer connects input to first hidden layer
            self.hidden_layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
            # Additional hidden layers connect between hidden layers
            for i in range(1, len(self.hidden_sizes)):
                self.hidden_layers.append(
                    nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i])
                )
            last_hidden_size = self.hidden_sizes[-1]
        else:
            # If no hidden layers, connect input directly to output
            last_hidden_size = self.input_size

        # Linear output layer (logits)
        self.output_layer = nn.Linear(last_hidden_size, self.output_size)

    def get_flat_idx(self, unflat_idx):
        """
        Translates a neuron index specified by (layer_idx, neuron_idx)
        to its corresponding flattened index in the concatenated activation vector.
        """
        layer_idx, neuron_idx = unflat_idx
        # List of neuron counts per hidden layer.
        out_features = [layer.out_features for layer in self.hidden_layers]
        if layer_idx < 0 or layer_idx >= len(out_features):
            raise ValueError(
                f"Layer index {layer_idx} is out of range."
                + f"Available layers: 0 to {len(out_features)-1}."
            )
        if neuron_idx < 0 or neuron_idx >= out_features[layer_idx]:
            raise ValueError(
                f"Neuron index {neuron_idx} is out of bounds for layer {layer_idx}"
                + f"with size {out_features[layer_idx]}."
            )
        return sum(out_features[:layer_idx]) + neuron_idx # type: ignore

    def get_unflat_idx(self, flat_idx):
        """
        Translates a flattened index from the concatenated activation vector
        back to a tuple (layer_idx, neuron_idx).
        """
        out_features = [layer.out_features for layer in self.hidden_layers]
        running_total = 0
        for i, count in enumerate(out_features):
            if flat_idx < running_total + count: # type: ignore
                return (i, flat_idx - running_total)
            running_total += count # type: ignore
        raise ValueError(
            f"Flat index {flat_idx} is out of range. Total neurons: {running_total}."
        )

    def forward(self, x, return_activations=False, apply_softmax=False):
        activations = {}
        for idx, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            activations[idx] = x
        out = self.output_layer(x)

        # Apply softmax if requested (for getting probabilities)
        if apply_softmax:
            out = F.softmax(out, dim=1)

        if return_activations:
            return out, activations
        else:
            return out

    def fit(
        self,
        train_loader,
        criterion=None,
        optimizer=None,
        num_epochs=150,
        print_every=10,
    ):
        """
        Trains the model using the provided DataLoader, loss function, and optimizer.
        """
        print("Training...")
        # Default to CrossEntropyLoss if no criterion is provided
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Default to Adam optimizer if none is provided
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            if (epoch + 1) % print_every == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    def evaluate(self, test_loader, criterion=None):
        """
        Evaluates the model on the test DataLoader and prints the accuracy.

        Args:
            test_loader: DataLoader for the test dataset
            criterion: Optional loss function (defaults to CrossEntropyLoss if None)

        Returns:
            tuple: (accuracy, loss) as floats
        """
        self.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        # Use CrossEntropyLoss by default
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_loader:
                # Get logits
                outputs = self(inputs)

                # Calculate loss
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = running_loss / len(test_loader)

        print(f"Test Accuracy: {accuracy:.2%}, Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def get_activations(self, x, detach=True):
        # Ensure x is batched.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Run forward pass to obtain activations dictionary.
        self.eval()
        _, acts = self.forward(x, return_activations=True)
        # Flatten each activation tensor (preserving batch dimension)
        # and sort keys for consistency.
        act_list = [acts[key].flatten(1) for key in sorted(acts.keys())] # type: ignore
        h = torch.cat(act_list, dim=1)
        return h.detach() if detach else h

    def activation_jacobian(self, x):
        """
        Jacobian of every internal activation with respect to the input.

        For a ReLU MLP this is exact in closed form, so there is no need for
        autograd: with a_l = relu(W_l a_{l-1} + b_l) and D_l the diagonal 0/1
        mask of active units,

            d a_l / d x = D_l W_l (d a_{l-1} / d x),     d a_0 / d x = I.

        Computing it directly is ~700x faster than calling
        `torch.autograd.functional.jacobian` per sample, and agrees with it
        bitwise on the handcrafted quadrant nets.

        Args:
            x (torch.Tensor): batched input, shape (B, input_dim).

        Returns:
            torch.Tensor of shape (B, internal_neurons, input_dim), with rows in
            the same order as `get_activations`.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        # d a_0 / d x = identity, broadcast over the batch.
        jac = torch.eye(self.input_size, dtype=x.dtype, device=x.device)
        jac = jac.unsqueeze(0).expand(batch_size, -1, -1)

        acts = x
        per_layer = []
        for layer in self.hidden_layers:
            pre = layer(acts)
            acts = self.activation(pre)
            # ReLU'(z) is 0 at z = 0, matching autograd's subgradient choice.
            mask = (pre > 0).to(x.dtype).unsqueeze(-1)
            jac = mask * torch.matmul(layer.weight, jac)
            per_layer.append(jac)

        if not per_layer:
            return torch.zeros(batch_size, 0, self.input_size,
                               dtype=x.dtype, device=x.device)
        return torch.cat(per_layer, dim=1)

    # ------------------------------------------------------------------
    # Counterfactual ablation as a tracked continuation.
    #
    # The trajectory is parameterised by the FRACTION of activation removed,
    # s in [0, 1], not by an integration time. At parameter s every predictor of
    # every probe is held at (1 - s) of its own baseline, so s = 1 is exactly
    # "all predictors off" and the endpoint is reached by construction rather
    # than by convergence check.
    #
    # Three things follow, and all three fix defects of a time-parameterised
    # flow:
    #
    #   * Filtration values become scale free. In activation units they ran to
    #     ~6 on the handcrafted nets and ~80 on the trained ones, so they were
    #     not comparable across architectures -- which is what the manuscript
    #     compares.
    #   * The counterfactual is homogeneous across probes. Under a common time,
    #     a probe with a small baseline was fully ablated and frozen while a
    #     large one was barely touched, and the paired test mixed the two and
    #     called it one intervention. At a common s they are all ablated by the
    #     same fraction.
    #   * Predictors reach zero together, so the active set never restages
    #     mid-trajectory, and a predictor's own gate stays open for all s < 1.
    #
    # What remains hard is upstream: a dead ReLU above the predictor has zero
    # derivative, so the Jacobian cannot see the direction that would revive it,
    # and steepest descent chatters at the kink instead of crossing it. On
    # trained weights that stalled 52 of 128 probes at 72% of baseline, with a
    # fully ablating input 1.0 units away, and it is not a local minimum --
    # descent exists at every radius, so a line search does not help (enforcing
    # monotone decrease made it strictly worse, 0 of 128). The direction used
    # here is therefore the min-norm element of the Goldstein epsilon-
    # subdifferential: over every linear piece within epsilon of the iterate,
    # the smallest step that meets the tracking target on the true piece and
    # does not undershoot it on any nearby piece.
    # ------------------------------------------------------------------

    def _locate_neuron(self, index):
        """Map a global internal-neuron index to (hidden layer, unit)."""
        offset = 0
        for layer_idx, size in enumerate(self.hidden_sizes):
            if index < offset + size:
                return layer_idx, index - offset
            offset += size
        raise IndexError(f"neuron {index} is not an internal neuron")

    def _forward_trace(self, x):
        """Pre-activations per hidden layer, alongside the activations."""
        pre, acts = [], []
        current = x
        for layer in self.hidden_layers:
            z = layer(current)
            current = self.activation(z)
            pre.append(z)
            acts.append(current)
        return pre, acts

    def _predictor_rows(self, gates, targets, upto, dtype, device, batch_shape):
        """d z / d x for each predictor, under an explicit set of gates.

        Only layers strictly below a predictor's own layer can affect its
        pre-activation gradient, so the recursion runs to `upto` and stops.
        Accumulates forward as `Jz_l = W_l Ja_{l-1}`, `Ja_l = D_l Jz_l`, keeping
        only the rows the predictors need: shape (*batch_shape, m, input_dim).
        """
        jac = torch.eye(self.input_size, dtype=dtype, device=device)
        jac = jac.expand(*batch_shape, self.input_size, self.input_size)
        rows = [None] * len(targets)
        for layer_idx in range(upto + 1):
            jz = torch.matmul(self.hidden_layers[layer_idx].weight, jac)
            for slot, (target_layer, unit) in enumerate(targets):
                if target_layer == layer_idx:
                    rows[slot] = jz[..., unit, :]
            if layer_idx < upto:
                jac = gates[layer_idx].unsqueeze(-1) * jz
        return torch.stack(rows, dim=-2)

    @staticmethod
    def _least_norm_step(grads, delta):
        """Min-norm dx with `grads @ dx = delta`, batched.

        Returns (dx, infeasibility), the second being ||grads @ dx - delta|| as
        a fraction of ||delta||. It is zero whenever the requested rates lie in
        the row space of the Jacobian and grows only when they do not, i.e.
        when the predictors' gradients cannot be satisfied at once. `pinv` will
        happily return the least-squares compromise in that case, so the caller
        has to be told rather than left to trust a step that never happened.
        """
        gram = torch.matmul(grads, grads.transpose(-1, -2))
        sol = torch.matmul(torch.linalg.pinv(gram), delta.unsqueeze(-1))
        dx = torch.matmul(grads.transpose(-1, -2), sol).squeeze(-1)
        gap = torch.matmul(grads, dx.unsqueeze(-1)).squeeze(-1) - delta
        scale = torch.linalg.norm(delta, dim=-1).clamp(min=1e-12)
        return dx, torch.linalg.norm(gap, dim=-1) / scale

    def _kink_direction(self, x, idxs, want, active, tol=1e-9, max_ambiguous=3):
        """Descent direction resolved to a genuine adjacent linear piece.

        A ReLU unit sitting exactly on its kink has no gradient, and the
        `z > 0` convention scores it off, which yields a row belonging to no
        adjacent piece: on the handcrafted monosemantic net, ablating n0 lands
        x on 0, and the row for n4 comes out (0, 1) where the two neighbouring
        pieces give (1, 1) and (1.25, 1). Descending it walks the input along
        y to the origin instead of into the counterfactual set.

        The fix is to pick a piece rather than average over one. For each gate
        assignment of the units at a kink, solve for the direction that piece
        prescribes, then keep the assignment only if the direction actually
        enters it -- the unit's pre-activation must move the way its assumed
        gate says. Among the assignments that survive, the smallest step wins,
        which is the min-norm element of the Clarke subdifferential restricted
        to the pieces the trajectory can reach. Assignments that no motion
        realises, such as two anti-parallel units on at once, fail the test and
        drop out on their own.
        """
        targets = [self._locate_neuron(int(i)) for i in idxs]
        upto = max(layer for layer, _ in targets)
        pre, _ = self._forward_trace(x)
        base_gates = [(z > 0).to(x.dtype) for z in pre]
        batch = x.shape[0]

        if upto == 0:
            return None
        z_up = torch.cat(pre[:upto], dim=1)
        kink = z_up.abs() <= tol
        n_amb = min(max_ambiguous, int(kink.sum(dim=1).max()))
        if n_amb == 0:
            return None

        # The units at a kink, per sample; slots beyond a sample's count repeat
        # the first and are neutralised by `valid`.
        order = torch.argsort((~kink).to(x.dtype), dim=1, stable=True)[:, :n_amb]
        valid = kink.gather(1, order)

        # Pre-activation gradients of those units, taken on the base gates: a
        # unit's own row depends only on gates strictly below it.
        rows_up = self._predictor_rows(
            base_gates, [self._locate_neuron(int(i)) for i in range(z_up.shape[1])],
            upto - 1 if upto > 1 else 0, x.dtype, x.device, (batch,))
        g_amb = rows_up.gather(
            1, order.unsqueeze(-1).expand(-1, -1, rows_up.shape[-1]))

        best_dx = None
        best_rate = None
        best_norm = torch.full((batch,), float("inf"), dtype=x.dtype, device=x.device)
        bounds = torch.tensor([0] + list(self.hidden_sizes), device=x.device).cumsum(0)
        for pattern in range(1 << n_amb):
            bits = torch.tensor([(pattern >> b) & 1 for b in range(n_amb)],
                                dtype=x.dtype, device=x.device)
            gates = [g.clone() for g in base_gates]
            for slot in range(n_amb):
                for layer_idx in range(upto):
                    lo, hi = int(bounds[layer_idx]), int(bounds[layer_idx + 1])
                    sel = valid[:, slot] & (order[:, slot] >= lo) & (order[:, slot] < hi)
                    if not bool(sel.any()):
                        continue
                    rid = torch.nonzero(sel).flatten()
                    gates[layer_idx][rid, order[rid, slot] - lo] = bits[slot]
            rows = self._predictor_rows(gates, targets, upto, x.dtype, x.device,
                                        (batch,))
            rows = rows * active.unsqueeze(-1).to(rows.dtype)
            dx, _ = self._least_norm_step(rows, want * active.to(want.dtype))
            # Does the step enter the piece it assumed?
            move = torch.matmul(g_amb, dx.unsqueeze(-1)).squeeze(-1)
            wants_on = bits.unsqueeze(0) > 0.5
            agree = torch.where(wants_on, move > 0, move < 0) | ~valid
            ok = agree.all(dim=1)
            norm = torch.linalg.norm(dx, dim=-1)
            take = ok & (norm < best_norm)
            if best_dx is None:
                best_dx = torch.zeros_like(dx)
                best_rate = torch.zeros_like(want)
            # The rate has to be read off the piece that produced the step. The
            # base rows belong to the degenerate assignment and would report a
            # rate the trajectory is not travelling at, which then mis-sizes the
            # clip.
            piece_rate = -torch.matmul(rows, dx.unsqueeze(-1)).squeeze(-1)
            best_dx = torch.where(take.unsqueeze(-1), dx, best_dx)
            best_rate = torch.where(take.unsqueeze(-1), piece_rate, best_rate)
            best_norm = torch.where(take, norm, best_norm)
        return best_dx, torch.isfinite(best_norm), best_rate

    def ablation_trajectory(self, x, neuron_idxs, n_steps=100, dt=0.01,
                            max_internal=None, kink_tol=1e-9, max_ambiguous=3,
                            stall_patience=50):
        """Activations along the counterfactual ablation, indexed by ablation.

        The path is the minimum-norm descent in *input* space that drives the
        predictor set to zero: at each step the still-active predictors are
        reduced together at equal rate, and a predictor that reaches zero drops
        out of the objective. Two properties of how the path is terminated and
        read are what make it usable as a filtration.

        First, the step is shortened so that the binding predictor lands exactly
        on its kink rather than one step past it. Overshooting by up to `dt`
        switches on the partner of a predictor with an anti-parallel gradient,
        and the partner's activation then scales linearly with `dt` -- 0.0102,
        0.0050, 0.0026, 0.0014, 0.00068 at dt = 0.02 down to 0.00125 on the
        handcrafted monosemantic net. That is enough to clear the relevance
        margin and manufacture an edge recording the integrator's step size and
        nothing about the network.

        Second, the filtration value is the fraction of the predictor set's
        baseline activation removed so far,

            s = 1 - sum_S a_i(x(t)) / sum_S a_i(x(0)),

        rather than elapsed time. Elapsed time is a property of the integrator
        and of the local conditioning, and differs from probe to probe; the ablation
        is a property of the intervention. The trajectory is resampled onto a
        uniform grid in s, so index k corresponds to s = k / n_steps for every
        probe and s = 1 is the fully ablated counterfactual. Probes are then
        aligned by ablation rather than by clock, and the axis no longer depends on
        the step count or on a horizon chosen in advance.

        The aggregate is used rather than a per-predictor fraction deliberately.
        Constraining each predictor to (1 - s) of its own baseline over-determines
        the step once the set is larger than the input dimension, and on a network
        with zero biases -- which is positively homogeneous -- it forces the
        radial ray, along which every neuron in the network is scaled by (1 - s)
        together. That counterfactual is a uniform dimming and cannot express any
        causal structure at all. Reading the aggregate off the unconstrained
        descent path avoids this: the path is chosen by the descent, and s only
        measures how far along it the ablation has got.

        Returns (activations, report) with activations of shape
        (n_steps + 1, B, internal_neurons).
        """
        single = not hasattr(neuron_idxs, "__iter__") or isinstance(neuron_idxs, int)
        idxs = [neuron_idxs] if single else list(neuron_idxs)

        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.clone().detach()
        acts = self.get_activations(x)
        baseline = acts[:, idxs]
        batch, dim = x.shape[0], acts.shape[1]
        total0 = baseline.clamp(min=0).sum(1)
        alive = total0 > 0

        if max_internal is None:
            # Each active predictor falls at rate 1/sqrt(m) per unit time, so a
            # well-conditioned ablation finishes within the summed baseline. The
            # rate degrades where the Jacobian is nearly singular and the descent
            # crawls, so the budget carries a wide margin and `truncated` reports
            # any probe that still needs more.
            max_internal = int(16.0 * float(total0.max()) / dt) + 64

        traj, ablation_grid = [acts], [torch.zeros(batch, dtype=acts.dtype, device=x.device)]
        running = alive.clone()
        stalled = torch.zeros(batch, dtype=torch.bool, device=x.device)
        # A probe can chatter across a kink, taking steps that cancel and leaving
        # the ablation where it was. Without a guard it burns the whole budget and
        # drags every other probe's wall clock with it, so a probe that stops
        # advancing is frozen and reported rather than retried indefinitely.
        idle = torch.zeros(batch, dtype=torch.long, device=x.device)
        last_ablation = ablation_grid[0].clone()

        for _ in range(max_internal):
            active = acts[:, idxs] > 0
            running = running & active.any(dim=1)
            if not bool(running.any()):
                break

            jac = self.activation_jacobian(x)[:, idxs, :]
            jac = jac * active.unsqueeze(-1).to(jac.dtype)
            jac_norm = torch.linalg.norm(jac.flatten(1), dim=1)
            dead = torch.isclose(jac_norm, torch.zeros_like(jac_norm))
            # A probe whose Jacobian vanishes while predictors are still positive
            # cannot be ablated by any input move, and has to be reported.
            stalled |= running & dead
            running = running & ~dead
            if not bool(running.any()):
                break

            m = active.sum(dim=1).clamp(min=1).to(jac.dtype)
            # Equal-rate descent on the still-active predictors, but taken as the
            # minimum-norm subgradient over the gate patterns within `epsilon` of
            # a kink. Without this, a unit sitting exactly at its kink is scored
            # gate-off, and the resulting row is the gradient of no adjacent
            # piece at all: on the handcrafted monosemantic net the direction for
            # n4 at x = 0 comes out (0, 1) where the two pieces give (1, 1) and
            # (1.25, 1), and descending it walks the input to the origin instead
            # of into the counterfactual set.
            want = -active.to(jac.dtype) / torch.sqrt(m).unsqueeze(1)
            jjt = torch.matmul(jac, jac.transpose(1, 2))
            sol = torch.matmul(torch.linalg.pinv(jjt), want.unsqueeze(-1))
            step_dir = torch.matmul(jac.transpose(1, 2), sol).squeeze(-1)

            # Step clipping lands units exactly on their kinks, so the units whose
            # gate is genuinely undetermined are found by an exact test rather
            # than a neighbourhood one, and there are usually none. Only those
            # probes pay for the piece-consistency search.
            pre, _ = self._forward_trace(x)
            at_kink = (torch.cat(pre, dim=1).abs() <= kink_tol) & running.unsqueeze(1)
            sub = torch.nonzero(at_kink.any(dim=1)).flatten()
            if len(sub):
                fixed = self._kink_direction(x[sub], idxs, want[sub], active[sub],
                                             tol=kink_tol,
                                             max_ambiguous=max_ambiguous)
                if fixed is not None:
                    fix, ok, fix_rate = fixed
                    step_dir = step_dir.clone()
                    step_dir[sub] = torch.where(ok.unsqueeze(-1), fix, step_dir[sub])

            # Longest step that leaves every active predictor at or above zero,
            # measured against the rate the linearisation actually delivers.
            rate = -torch.matmul(jac, step_dir.unsqueeze(-1)).squeeze(-1)
            if len(sub) and fixed is not None:
                rate[sub] = torch.where(ok.unsqueeze(-1), fix_rate, rate[sub])
            room = torch.where(active & (rate > 1e-12),
                               acts[:, idxs] / rate.clamp(min=1e-12),
                               torch.full_like(rate, float("inf")))
            h = room.amin(dim=1).clamp(max=dt)
            h = torch.where(running, h, torch.zeros_like(h))

            x = x + h.unsqueeze(1) * step_dir
            acts = self.get_activations(x)
            traj.append(acts)
            ablation_now = 1.0 - acts[:, idxs].clamp(min=0).sum(1) / total0.clamp(min=1e-12)
            ablation_grid.append(ablation_now)

            moved = (ablation_now - last_ablation) > 1e-9
            idle = torch.where(moved | ~running, torch.zeros_like(idle), idle + 1)
            last_ablation = torch.where(moved, ablation_now, last_ablation)
            frozen = running & (idle >= stall_patience)
            stalled |= frozen
            running = running & ~frozen

        # Honest statement of what is left, rather than whichever branch the loop
        # happened to exit through: a probe is truncated if the budget ran out
        # with predictors still above zero.
        truncated = (acts[:, idxs] > 0).any(dim=1) & alive
        path = torch.stack(traj, dim=0)                       # (N+1, B, dim)
        # A probe with no activation to remove is ablated at s = 0; give it a
        # constant ablation so the resampling carries its baseline forward.
        ablation = torch.stack(ablation_grid, dim=0)
        ablation = torch.where(alive.unsqueeze(0), ablation, torch.ones_like(ablation))
        ablation[0] = torch.where(alive, torch.zeros_like(ablation[0]), torch.ones_like(ablation[0]))
        ablation = torch.cummax(ablation, dim=0).values               # monotone in s

        grid = torch.linspace(0.0, 1.0, n_steps + 1, dtype=acts.dtype,
                              device=x.device)
        if path.shape[0] == 1:
            # Nothing moved: every predictor was already zero at baseline, so the
            # counterfactual equals the factual at every ablation.
            out = path.expand(n_steps + 1, batch, dim).clone()
            return out, {
                "stalled": stalled, "truncated": truncated,
                "residual_fraction": torch.zeros(batch, dtype=acts.dtype,
                                                 device=x.device),
                "ablated": torch.ones(batch, dtype=torch.bool, device=x.device),
                "baseline": baseline, "internal_steps": 0,
            }
        d_t = ablation.transpose(0, 1).contiguous()               # (B, N+1)
        hi = torch.searchsorted(d_t, grid.expand(batch, -1).contiguous())
        hi = hi.clamp(1, path.shape[0] - 1)
        lo = hi - 1
        d_lo, d_hi = d_t.gather(1, lo), d_t.gather(1, hi)
        w = ((grid.unsqueeze(0) - d_lo) / (d_hi - d_lo).clamp(min=1e-12)).clamp(0, 1)
        p_t = path.permute(1, 0, 2)                           # (B, N+1, dim)
        a_lo = p_t.gather(1, lo.unsqueeze(-1).expand(-1, -1, dim))
        a_hi = p_t.gather(1, hi.unsqueeze(-1).expand(-1, -1, dim))
        out = (a_lo + w.unsqueeze(-1) * (a_hi - a_lo)).permute(1, 0, 2)

        residual = out[-1][:, idxs]
        scale = baseline.clamp(min=1e-12)
        fraction = torch.where(baseline > 0, residual / scale,
                               torch.zeros_like(residual))
        done = (fraction <= ABLATION_TOLERANCE).all(dim=1)
        report = {
            # Both flags mean "stopped with something left to remove". A probe
            # whose last predictor decays asymptotically trips the stagnation
            # guard at a residual of order 1e-30, which is stopping at the answer
            # rather than short of it, and reporting it would bury the cases that
            # genuinely failed.
            "stalled": stalled & ~done,
            "truncated": truncated & ~done,
            "residual_fraction": fraction.amax(dim=1),
            "ablated": done,
            "baseline": baseline,
            "internal_steps": path.shape[0] - 1,
        }
        return out, report

    def invert(self, target):
        """
        Approximate an input that produces the given target vector.
        Inverts the network layer-by-layer.
        """
        self.eval()
        # Ensure target is a tensor with batch dim.
        target = target.clone().detach()
        if target.dim() == 1:
            target = target.unsqueeze(0)

        current = target
        # Invert output layer.
        W = self.output_layer.weight.data
        b = self.output_layer.bias.data
        current = torch.pinverse(W) @ (current.T - b.unsqueeze(1))
        current = current.T

        # Invert hidden layers in reverse order.
        for hidden_layer in reversed(self.hidden_layers):
            current = torch.clamp(current, min=0)
            W = hidden_layer.weight.data
            b = hidden_layer.bias.data
            current = torch.pinverse(W) @ (current.T - b.unsqueeze(1)) # type: ignore
            current = current.T
        return current

    def generate_batch_from_target(
        self, target, batch_size=32, num_iterations=500, lr=0.01, noise_std=0.01
    ):
        """
        Generates a batch of inputs yielding outputs near target.
        Combines pseudo-inverse initialization with gradient-descent refinement.
        """
        self.eval()
        # Prepare target tensor.
        target = target.clone().detach()
        if target.dim() == 1:
            target = target.unsqueeze(0)

        # Step 1: Initial guess via inversion.
        x_init = self.invert(target)

        # Step 2: Refine with gradient descent.
        x_refined = x_init.clone().detach().requires_grad_(True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam([x_refined], lr=lr)
        for _ in range(num_iterations):
            optimizer.zero_grad()
            out = self.forward(x_refined)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        # Step 3: Generate batch with noise.
        batch = x_refined.detach().repeat(batch_size, 1)
        noise = torch.randn_like(batch) * noise_std
        return batch + noise

    def save(self, filepath):
        """
        Save the model parameters to disk.

        Args:
            filepath (str): Path to save the model to.
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        torch.save(self, filepath)
        print(f"Model saved to {filepath}")

    def copy(self):
        """
        Creates a copy of the current MLP instance with the same architecture and state.

        Returns:
            MLP: A new MLP instance with the same parameters.
        """
        # Create a new instance with the same layer sizes
        copied_model = MLP(layer_sizes=self.layer_sizes)
        # Load the state dictionary into the new instance
        copied_model.load_state_dict(self.state_dict())
        return copied_model
