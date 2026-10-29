"""
The edge set the circuit-discovery baselines search, and its translation into
the index space a DSC speaks.

Two vocabularies meet here. ACDC and EAP address a neuron as
`(layer, neuron)`, where layer 0 is the input, layer 1 the first hidden layer,
and the last layer the output; their edges are the individual weights between
adjacent layers. A `DirectedSimplicialComplex` addresses a neuron by a single
flat index over the *internal* neurons only -- `range(model.internal_neurons)`,
hidden layers concatenated in order -- and its 1-simplices are ordered pairs of
those, unconstrained by adjacency or by direction of flow.

Both baselines are restricted to hidden-to-hidden edges, matching the DSC's
vertex set. That restriction is the substance of the comparison rather than an
implementation detail: it means the baselines can only ever propose edges of
one of the three kinds `relation_kind` distinguishes, so the two-thirds of a
DSC's 1-skeleton that is not forward-adjacent lies outside their range by
construction and cannot be scored as disagreement.
"""

from typing import Dict, List, Set, Tuple

Neuron = Tuple[int, int]
Edge = Tuple[Neuron, Neuron]


def hidden_layer_offsets(model) -> Dict[int, int]:
    """
    Flat-index offset of each hidden layer, keyed by ACDC layer number.

    Hidden layer h occupies flat indices `offset[h + 1] ...` in the DSC's
    enumeration, since ACDC counts the input as layer 0.
    """
    offsets = {}
    running = 0
    for h, size in enumerate(model.hidden_sizes):
        offsets[h + 1] = running
        running += size
    return offsets


def flat_index(model, layer: int, neuron: int) -> int:
    """
    Flat DSC index of an ACDC `(layer, neuron)` address.

    Derived from `model.hidden_sizes` rather than hardcoded: the monosemantic
    and polysemantic quadrant nets have different first hidden widths (4 and 3),
    so a fixed offset silently mislabels every neuron of the second layer on one
    of them.

    Raises:
        ValueError: for the input or output layer, which have no flat index --
            they are not DSC vertices.
    """
    offsets = hidden_layer_offsets(model)
    if layer not in offsets:
        raise ValueError(
            f"layer {layer} is not a hidden layer of this model "
            f"(hidden layers are {sorted(offsets)}); it has no DSC vertex index"
        )
    size = model.hidden_sizes[layer - 1]
    if not 0 <= neuron < size:
        raise ValueError(f"neuron {neuron} out of range for layer {layer} of width {size}")
    return offsets[layer] + neuron


def hidden_edges(model) -> Set[Edge]:
    """
    Every weight between adjacent hidden layers, as `((l, i), (l+1, j))`.

    Input and output layers are excluded so that the searched graph spans
    exactly the DSC's vertex set.
    """
    edges = set()
    for layer in range(1, len(model.hidden_sizes)):
        for i in range(model.hidden_sizes[layer - 1]):
            for j in range(model.hidden_sizes[layer]):
                edges.add(((layer, i), (layer + 1, j)))
    return edges


def to_flat_edges(model, edges) -> Set[Tuple[int, int]]:
    """Translate `(layer, neuron)` edges into ordered flat-index pairs."""
    return {
        (flat_index(model, *source), flat_index(model, *target))
        for source, target in edges
    }


def layer_of_flat(model) -> List[int]:
    """Hidden-layer number (ACDC convention) of each flat index."""
    layers = []
    for h, size in enumerate(model.hidden_sizes):
        layers.extend([h + 1] * size)
    return layers


def relation_kind(model, edge: Tuple[int, int]) -> str:
    """
    Classify a flat-index ordered pair by how the two neurons sit in the network.

    Returns one of:
        "forward_adjacent" -- source in layer l, target in layer l+1. The only
            kind a weight-graph method can propose.
        "within_layer" -- both in the same layer, with no weight between them.
        "other" -- backwards, or forward across a layer gap.

    A DSC edge is a claim about a counterfactual in *input* space, so it is not
    confined to the weight graph: suppressing a deep neuron requires moving the
    input, which perturbs every shallower neuron that depends on it. The
    backwards and within-layer edges are that effect, and they are statements no
    baseline here is able to make either way.
    """
    layers = layer_of_flat(model)
    source, target = edge
    ls, lt = layers[source], layers[target]
    if lt == ls + 1:
        return "forward_adjacent"
    if lt == ls:
        return "within_layer"
    return "other"
