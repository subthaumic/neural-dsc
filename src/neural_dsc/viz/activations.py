"""Per-node mean values for a context, in the form the DSC panels want.

Kept apart from `neural_dsc.viz.complex` because this is the only part of the drawing
path that needs a model and a tensor: the complex renderer takes numbers and
knows nothing about where they came from.
"""

from typing import NamedTuple

import matplotlib as mpl
import torch


class PanelValues(NamedTuple):
    """Mean value of every node in one panel: input, hidden and output."""

    inputs: list
    hidden: dict
    outputs: list

    def every_value(self):
        return list(self.inputs) + list(self.hidden.values()) + list(self.outputs)

    def fill_kwargs(self):
        """The `draw_dsc` keywords that colour a panel by these values."""
        return {"node_values": self.hidden, "input_values": self.inputs,
                "output_values": self.outputs}


def probe_node_values(model, points):
    """Mean input, hidden activation and output over one context's probe set.

    Taken from the probes the complex was built from, not from a fresh draw, so
    the number colouring a node belongs to the same context as the simplices
    drawn around it.
    """
    with torch.no_grad():
        return PanelValues(
            inputs=points.mean(dim=0).tolist(),
            hidden=dict(enumerate(
                model.get_activations(points).mean(dim=0).tolist())),
            outputs=model(points).mean(dim=0).tolist(),
        )


def panel_values(model, probe_sets):
    """`probe_node_values` for each context, or None when there are no probes."""
    if not probe_sets:
        return None
    model.eval()
    return [probe_node_values(model, points) for points in probe_sets]


def shared_node_norm(panels):
    """One scale across a figure's panels, so their node fills are comparable.

    Symmetric about zero, matching `neural_dsc.viz.network`: coolwarm is a diverging
    ramp whose midpoint carries the meaning, and ReLU activations are
    non-negative, so zero has to land on the white centre. Scaling 0..max
    instead would put a dead neuron at full blue and a mid one at invisible
    white, which inverts the thing the fill is for. Input coordinates do go
    negative, and on this ramp they read as blue, which is what they are.
    """
    if not panels:
        return None
    values = [value for panel in panels for value in panel.every_value()]
    if not values:
        return None
    magnitude = max(abs(value) for value in values) or 1.0
    return mpl.colors.Normalize(vmin=-magnitude, vmax=magnitude)


def fill_kwargs(panels, index):
    """Node-fill keywords for one panel, or none at all if there are none."""
    return panels[index].fill_kwargs() if panels else {}
