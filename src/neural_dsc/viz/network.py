"""Architecture and activation plots on the shared network geometry."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from neural_dsc.viz.complex import (
    BOUNDARY_EDGE,
    BOUNDARY_FACE,
    CAPTION_OFFSET,
    HIDDEN_EDGE,
    HIDDEN_FACE,
    NEURON_RADIUS,
    format_network_axes,
    neuron_layout,
)

# The bias marker is drawn at 0.64 of the neuron's area, so its half-width is
# sqrt(0.64) = 0.8 of the neuron radius.
BIAS_MARKER_SCALE = 0.8
BIAS_CAPTION_GAP = 0.12


def _node_size_for_data_radius(ax, radius):
    """Return a networkx marker area matching a radius in data coordinates."""
    ax.figure.canvas.draw()
    centre = ax.transData.transform((0.0, 0.0))
    rim = ax.transData.transform((radius, 0.0))
    radius_points = np.linalg.norm(rim - centre) * 72.0 / ax.figure.dpi
    return (2.0 * radius_points) ** 2


def _normalization(values, symmetric):
    if not values:
        return mcolors.Normalize(vmin=-1.0 if symmetric else 0.0, vmax=1.0)
    if symmetric:
        magnitude = max(abs(value) for value in values)
        magnitude = magnitude if magnitude > 0 else 1.0
        return mcolors.Normalize(vmin=-magnitude, vmax=magnitude)
    lower, upper = min(values), max(values)
    if lower == upper:
        padding = max(0.1, abs(lower) * 0.05)
        lower, upper = lower - padding, upper + padding
    return mcolors.Normalize(vmin=lower, vmax=upper)


def plot_neural_network(
    model,
    input=None,
    plot_parameters=True,
    title=None,
    save_path=None,
    show_plot=True,
    ax=None,
    **kwargs,
):
    """Visualize an MLP, optionally colouring neurons by mean activation.

    Architecture, activation, and DSC plots all use the same neuron layout.
    Layer spacing, neuron size, vertical centring, and layer captions therefore
    mean the same thing in every figure.
    """
    layout = neuron_layout(model.layer_sizes)
    graph = nx.DiGraph()
    activations = None
    output_values = None
    input_values = None
    if input is not None:
        with torch.no_grad():
            output_values = torch.mean(model(input), dim=0)
            activations = torch.mean(model.get_activations(input), dim=0)
            input_values = torch.mean(input, dim=0)

    num_layers = len(model.layer_sizes)
    neuron_nodes = []
    boundary_nodes = set()
    bias_nodes = []
    bias_offset = 0.40 * layout.row_gap

    for layer_idx, layer_size in enumerate(model.layer_sizes):
        is_input = layer_idx == 0
        is_output = layer_idx == num_layers - 1
        for neuron_idx in range(layer_size):
            if is_input:
                node_id = f"input_{neuron_idx}"
                point = layout.inputs[neuron_idx]
            elif is_output:
                node_id = f"output_{neuron_idx}"
                point = layout.outputs[neuron_idx]
            else:
                node_id = model.get_flat_idx((layer_idx - 1, neuron_idx))
                point = layout.hidden[node_id]

            graph.add_node(
                node_id,
                pos=point,
                layer=layer_idx,
                neuron_idx=neuron_idx,
                is_bias=False,
            )
            neuron_nodes.append(node_id)
            if is_input or is_output:
                boundary_nodes.add(node_id)

            if is_input and input_values is not None:
                graph.nodes[node_id]["activation"] = input_values[neuron_idx].item()
            elif is_output and output_values is not None:
                graph.nodes[node_id]["activation"] = output_values[neuron_idx].item()
            elif activations is not None:
                graph.nodes[node_id]["activation"] = activations[node_id].item()

            if not plot_parameters or is_input:
                continue
            if is_output and hasattr(model, "output_layer"):
                bias = model.output_layer.bias.data[neuron_idx].item()
            elif not is_output:
                bias = model.hidden_layers[layer_idx - 1].bias.data[neuron_idx].item()
            else:
                continue
            bias_node_id = f"bias_{node_id}"
            graph.add_node(
                bias_node_id,
                pos=(point[0], point[1] + bias_offset),
                bias=bias,
                is_bias=True,
                layer=layer_idx,
            )
            bias_nodes.append(bias_node_id)
            graph.add_edge(bias_node_id, node_id, weight=bias, is_bias=True)

    if plot_parameters:
        for layer_idx in range(num_layers - 1):
            source_size = model.layer_sizes[layer_idx]
            target_size = model.layer_sizes[layer_idx + 1]
            if layer_idx == 0:
                weight_matrix = model.hidden_layers[0].weight.data
            elif layer_idx < num_layers - 2:
                weight_matrix = model.hidden_layers[layer_idx].weight.data
            elif hasattr(model, "output_layer"):
                weight_matrix = model.output_layer.weight.data
            else:
                weight_matrix = model.hidden_layers[-1].weight.data

            for source_idx in range(source_size):
                source_id = (
                    f"input_{source_idx}"
                    if layer_idx == 0
                    else model.get_flat_idx((layer_idx - 1, source_idx))
                )
                for target_idx in range(target_size):
                    target_id = (
                        f"output_{target_idx}"
                        if layer_idx == num_layers - 2
                        else model.get_flat_idx((layer_idx, target_idx))
                    )
                    graph.add_edge(
                        source_id,
                        target_id,
                        weight=weight_matrix[target_idx, source_idx].item(),
                        is_bias=False,
                    )

    activation_values = [
        graph.nodes[node]["activation"]
        for node in neuron_nodes
        if "activation" in graph.nodes[node]
    ]
    layer_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if not data.get("is_bias", False)
    ]
    bias_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("is_bias", False)
    ]
    weights = [graph.edges[edge]["weight"] for edge in layer_edges]
    biases = [graph.nodes[node]["bias"] for node in bias_nodes]

    # Parameter plots retain a single signed scale for weights, biases, and any
    # activations, as in the existing figures. Activation-only diagnostics use
    # their requested sequential map and their own observed range.
    if plot_parameters:
        cmap = plt.get_cmap(kwargs.get("parameter_cmap", "coolwarm"))
        norm = _normalization(activation_values + weights + biases, symmetric=True)
    else:
        cmap = plt.get_cmap(kwargs.get("activation_cmap", "plasma"))
        norm = _normalization(activation_values, symmetric=False)

    created_here = ax is None
    if created_here:
        panel_height = kwargs.get("panel_height", 6.5)
        figure_width = max(8.0, panel_height * layout.aspect + 1.15)
        _, ax = plt.subplots(figsize=(figure_width, panel_height))

    if not kwargs.get("hide_colorbar", False):
        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_map.set_array([])
        colourbar = ax.figure.colorbar(scalar_map, ax=ax, shrink=0.80, pad=0.02)
        colourbar.set_label(
            kwargs.get("colorbar_label", ""),
            rotation=270,
            labelpad=15,
            fontsize=kwargs.get("colorbar_labelsize", 12),
        )
        colourbar.ax.tick_params(labelsize=kwargs.get("colorbar_ticksize", 10))

    # Every top-row neuron carries a bias marker above it, so the captions have
    # to clear the markers rather than the neurons.
    bias_reach = bias_offset + BIAS_MARKER_SCALE * NEURON_RADIUS * layout.row_gap
    format_network_axes(
        ax,
        layout,
        title=title,
        layer_labels=kwargs.get("layer_labels", True),
        title_fontsize=kwargs.get("title_fontsize", 17),
        title_inside=False,
        frame=kwargs.get("frame", True),
        caption_lift=max(
            0.0, bias_reach + BIAS_CAPTION_GAP - CAPTION_OFFSET),
    )
    radius = NEURON_RADIUS * layout.row_gap
    node_size = _node_size_for_data_radius(ax, radius)
    pos = nx.get_node_attributes(graph, "pos")

    if layer_edges:
        max_weight = max(abs(weight) for weight in weights)
        max_weight = max_weight if max_weight > 0 else 1.0
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=layer_edges,
            edge_color=[cmap(norm(weight)) for weight in weights],
            width=[0.8 + 2.5 * abs(weight) / max_weight for weight in weights],
            arrows=True,
            arrowsize=14,
            node_size=node_size,
            ax=ax,
        )
    if bias_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=bias_edges,
            edge_color=[cmap(norm(graph.edges[edge]["weight"])) for edge in bias_edges],
            width=1.5,
            arrows=True,
            arrowsize=12,
            node_size=node_size,
            ax=ax,
        )

    neuron_colours = []
    neuron_edges = []
    for node in neuron_nodes:
        if "activation" in graph.nodes[node]:
            neuron_colours.append(cmap(norm(graph.nodes[node]["activation"])))
        elif node in boundary_nodes:
            neuron_colours.append(BOUNDARY_FACE)
        else:
            neuron_colours.append(HIDDEN_FACE)
        neuron_edges.append(BOUNDARY_EDGE if node in boundary_nodes else HIDDEN_EDGE)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=neuron_nodes,
        node_color=neuron_colours,
        edgecolors=neuron_edges,
        linewidths=1.0,
        node_size=node_size,
        node_shape="o",
        ax=ax,
    )
    if bias_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=bias_nodes,
            node_color=[cmap(norm(graph.nodes[node]["bias"])) for node in bias_nodes],
            edgecolors="#777777",
            linewidths=0.8,
            node_size=node_size * 0.64,
            node_shape="s",
            ax=ax,
        )

    if save_path:
        ax.figure.savefig(save_path, bbox_inches="tight")
        print(f"  Saved neural network visualization to {save_path}")
    if show_plot:
        plt.show()
    elif created_here and save_path is not None:
        plt.close(ax.figure)
    return ax
