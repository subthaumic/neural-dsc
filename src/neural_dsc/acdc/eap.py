"""
Edge attribution patching (EAP) and its integrated-gradients variant (EAP-IG).

EAP replaces ACDC's iterative search with a first-order Taylor estimate of what
patching an edge would do, so every edge is scored in one forward-backward pass
instead of one forward pass per edge per iteration
(Syed, Rager & Conmy, arXiv:2310.10348). For an edge i -> j between adjacent
hidden layers of an MLP, patching it moves the target's pre-activation by

    delta_j = W[j, i] * (h'_i - h_i),

with h the post-ReLU source activation on the clean run and h' on the
corrupted one, so the effect on a metric m is estimated by

    score(i -> j) = sum_batch  W[j, i] * (h'_i - h_i) * dm/dz_j.

EAP-IG (Hanna, Liu & Variengien, arXiv:2403.17806) keeps the same activation
difference but averages the gradient along the straight path from the corrupted
input to the clean one rather than reading it only at the clean end. The
distinction matters exactly where the endpoint gradient is unrepresentative,
which on these networks is everywhere: the quadrant classifiers are ReLU nets
whose units are hard off over half the input domain, so a unit that is silent
at the clean point contributes zero gradient there no matter how much it would
have mattered along the way.

The choice of metric is not free, and one combination is degenerate. ACDC scores
against the KL divergence from the clean output, which is minimised *at* the
clean output -- its gradient there is identically zero, so single-step EAP on
that metric returns a score of exactly zero for every edge. This is a property
of the metric rather than a bug, and `n_steps > 1` recovers it, which is one of
the reasons EAP-IG exists. `tests/test_eap.py` pins both halves. Use
`target_logit_diff` for single-step EAP, or `kl_to_clean` for both if the point
is to show the difference.

Unlike ACDC, neither method returns a circuit: both return a real-valued score
per edge, which is a filtration on the weight graph and can be compared against
a DSC's filtration rank for rank rather than thresholded into a set first.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .graph import Edge, hidden_edges


def kl_to_clean(clean_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    KL(clean || current), averaged over the batch. ACDC's objective.

    Zero and stationary at `logits == clean_logits`, so a gradient read there
    carries no information; see the module docstring.
    """
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(clean_logits, dim=-1),
        reduction="batchmean",
    )


def target_logit_diff(target_class: int) -> Callable:
    """
    Metric factory: target-class logit minus the mean of the others.

    A task metric in the sense EAP assumes -- it measures whether the network
    still does the thing, rather than whether its output has moved -- so its
    gradient at the clean run is informative. Sign convention: larger is better,
    so a positive attribution score means patching the edge *helps*, and the
    important edges are the ones with large negative score. Rank by magnitude.
    """

    def metric(clean_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        n = logits.shape[-1]
        target = logits[:, target_class]
        others = (logits.sum(dim=-1) - target) / (n - 1)
        return (target - others).mean()

    return metric


def _forward_keeping_preactivations(model, x):
    """Forward pass returning the hidden pre-activations as graph nodes."""
    preactivations = []
    current = x
    for layer in model.hidden_layers:
        z = layer(current)
        preactivations.append(z)
        current = torch.relu(z)
    return model.output_layer(current), preactivations


def _post_relu_activations(model, x):
    with torch.no_grad():
        _, activations = model(x, return_activations=True)
    return {i: a.detach() for i, a in activations.items()}


def edge_attribution_scores(
    model,
    clean_data: torch.Tensor,
    corrupted_data: torch.Tensor,
    metric: Optional[Callable] = None,
    n_steps: int = 1,
) -> Dict[Edge, float]:
    """
    Score every hidden-to-hidden edge by attribution patching.

    Args:
        model: MLP instance.
        clean_data: probe points.
        corrupted_data: their counterfactuals, aligned row for row.
        metric: callable (clean_logits, logits) -> scalar tensor. Defaults to
            `kl_to_clean`, which is ACDC's objective and which needs
            `n_steps > 1` to be informative.
        n_steps: gradient samples along the corrupted-to-clean input path.
            1 is EAP; anything larger is EAP-IG. Midpoints are used, so the
            clean endpoint -- where a saturated unit reports no gradient -- is
            never the only sample.

    Returns:
        Signed score per edge, keyed as `((layer, i), (layer + 1, j))`. Rank by
        magnitude; the sign says which way the metric would move.
    """
    if metric is None:
        metric = kl_to_clean
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

    model.eval()
    with torch.no_grad():
        clean_logits = model(clean_data).detach()

    clean_activations = _post_relu_activations(model, clean_data)
    corrupted_activations = _post_relu_activations(model, corrupted_data)

    # Averaged gradient of the metric with respect to each hidden
    # pre-activation, along the straight input path from corrupted to clean.
    summed = None
    for step in range(n_steps):
        if n_steps == 1:
            # The clean input itself, not `corrupted + 1.0 * (clean - corrupted)`:
            # the two differ in the last bits, which is enough to turn the
            # exactly-zero KL gradient at the clean point into rounding noise.
            x = clean_data.detach().clone()
        else:
            alpha = (step + 0.5) / n_steps
            x = (corrupted_data + alpha * (clean_data - corrupted_data)).detach()
        x.requires_grad_(True)
        logits, preactivations = _forward_keeping_preactivations(model, x)
        grads = torch.autograd.grad(metric(clean_logits, logits), preactivations)
        if summed is None:
            summed = [g.detach().clone() for g in grads]
        else:
            for total, g in zip(summed, grads):
                total += g.detach()
    mean_grad = [total / n_steps for total in summed]

    scores = {}
    for edge in hidden_edges(model):
        (from_layer, i), (to_layer, j) = edge
        # from_layer is 1-based over hidden layers, so hidden index from_layer-1.
        source_delta = (
            corrupted_activations[from_layer - 1][:, i]
            - clean_activations[from_layer - 1][:, i]
        )
        weight = model.hidden_layers[to_layer - 1].weight[j, i].detach()
        scores[edge] = float((weight * source_delta * mean_grad[to_layer - 1][:, j]).sum())
    return scores


def top_k_circuit(scores: Dict[Edge, float], k: int) -> set:
    """The k highest-magnitude edges, as a circuit comparable to ACDC's output."""
    ranked = sorted(scores, key=lambda e: abs(scores[e]), reverse=True)
    return set(ranked[:k])
