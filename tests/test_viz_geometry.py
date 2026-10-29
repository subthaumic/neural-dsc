import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle

from neural_dsc.mlp import MLP
from neural_dsc.viz.complex import (BOUNDARY_GAP, BOUNDARY_RADIUS_SCALE, HIDDEN_GAP,
                            NEURON_RADIUS, RAD_RECIPROCAL, arc_points,
                            NODE_LABEL_OFFSET, draw_dsc, neuron_layout,
                            two_simplex_boundary)
from neural_dsc.viz.network import plot_neural_network


LAYER_SIZES = [2, 4, 4, 4]


def test_boundary_layers_sit_closer_in_than_the_hidden_layers():
    """Two-tier spacing, not one step everywhere.

    Every simplex lives between the hidden columns. A uniform step was tried and
    squeezes the whole complex into a third of the panel while the two boundary
    bands hold the rest empty, so the gaps are deliberately unequal.
    """
    layout = neuron_layout(LAYER_SIZES)

    assert np.diff(layout.columns) == pytest.approx(
        [BOUNDARY_GAP, HIDDEN_GAP, BOUNDARY_GAP])
    assert BOUNDARY_GAP < HIDDEN_GAP


def test_a_two_simplex_closes_along_its_own_face_not_the_reciprocal_edge():
    """The (a, c) side of (a, b, c) is edge (a, c) reversed, never edge (c, a).

    When the reciprocal edge is present it is a different drawn arc, and closing
    along it bounds the patch with three edges that are not this simplex's
    faces -- a triangle a -> b -> c -> a that no ordered 2-simplex has.
    """
    positions = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 2.0)}
    radii = {(0, 1): 0.035, (1, 2): 0.035,
             (0, 2): RAD_RECIPROCAL, (2, 0): RAD_RECIPROCAL}

    closing = two_simplex_boundary((0, 1, 2), positions, radii)[-48:]
    face = arc_points(positions[0], positions[2], RAD_RECIPROCAL)[::-1]
    reciprocal = arc_points(positions[2], positions[0], RAD_RECIPROCAL)

    assert closing == pytest.approx(face)
    assert closing != pytest.approx(reciprocal)


def test_the_two_orderings_of_a_predictor_pair_get_different_patches():
    """Otherwise drawing both is pointless -- they would sit exactly on top."""
    positions = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 2.0)}
    radii = {(0, 1): RAD_RECIPROCAL, (1, 0): RAD_RECIPROCAL,
             (0, 2): 0.035, (1, 2): 0.035}

    forward = two_simplex_boundary((0, 1, 2), positions, radii)
    backward = two_simplex_boundary((1, 0, 2), positions, radii)

    assert forward.shape == backward.shape
    assert not np.allclose(forward, backward)


def test_dsc_sizes_neurons_by_role_and_places_layer_captions_above():
    layout = neuron_layout(LAYER_SIZES)
    fig, ax = plt.subplots()
    draw_dsc(
        ax,
        {},
        LAYER_SIZES,
        matplotlib.colors.Normalize(0, 1),
        matplotlib.colormaps["viridis"],
        layout=layout,
    )

    hidden_radius = NEURON_RADIUS * layout.row_gap
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    assert len(circles) == sum(LAYER_SIZES)
    # Hidden neurons carry the simplices; the boundary columns frame them.
    assert {circle.radius for circle in circles} == {
        hidden_radius, hidden_radius * BOUNDARY_RADIUS_SCALE}

    captions = [text for text in ax.texts if text.get_text() in layout.labels]
    assert [text.get_text() for text in captions] == layout.labels
    highest_neuron = max(circle.center[1] + circle.radius for circle in circles)
    assert all(text.get_position()[1] > highest_neuron for text in captions)
    plt.close(fig)


def test_architecture_plot_uses_the_same_positions_and_captions():
    model = MLP(layer_sizes=LAYER_SIZES)
    layout = neuron_layout(LAYER_SIZES)
    fig, ax = plt.subplots()
    plot_neural_network(
        model,
        plot_parameters=False,
        hide_colorbar=True,
        show_plot=False,
        ax=ax,
    )

    collection = max(ax.collections, key=lambda item: len(item.get_offsets()))
    actual = {tuple(point) for point in collection.get_offsets()}
    expected = set(layout.inputs + list(layout.hidden.values()) + layout.outputs)
    assert actual == expected

    captions = [text for text in ax.texts if text.get_text() in layout.labels]
    highest_neuron = max(y for _, y in expected) + NEURON_RADIUS * layout.row_gap
    assert [text.get_text() for text in captions] == layout.labels
    assert all(text.get_position()[1] > highest_neuron for text in captions)
    plt.close(fig)


def test_hidden_neurons_are_numbered_one_indexed_like_the_manuscript():
    """Nodes show 1..8; everything stored stays 0..7.

    The appendix names the monosemantic units $a_1 \\dots n_8$, so a reader
    matching a simplex to the table needs the panel to agree. Same split as the
    class labels: Q1..Q4 on the page, 0..3 on disk.
    """
    layout = neuron_layout(LAYER_SIZES)
    fig, ax = plt.subplots()
    draw_dsc(
        ax,
        {},
        LAYER_SIZES,
        matplotlib.colors.Normalize(0, 1),
        matplotlib.colormaps["viridis"],
        layout=layout,
    )

    hidden = {(round(x, 6), round(y, 6)) for x, y in layout.hidden.values()}
    numbers = {
        text.get_text() for text in ax.texts
        if (round(text.get_position()[0], 6),
            round(text.get_position()[1], 6)) in hidden
    }
    assert numbers == {str(i + NODE_LABEL_OFFSET) for i in layout.hidden}
    assert numbers == {"1", "2", "3", "4", "5", "6", "7", "8"}
    plt.close(fig)


def test_layer_captions_clear_the_bias_markers_in_the_parameter_view():
    """The caption band is measured from the neurons; biases sit above them.

    Moving the captions above the columns put them straight onto the bias
    markers the parameter view hangs over every top-row neuron. The lift is what
    keeps the two apart, so pin the ordering rather than the constant.
    """
    from neural_dsc.viz.complex import CAPTION_OFFSET, format_network_axes

    layout = neuron_layout(LAYER_SIZES)
    top = layout.ylim[1] - 1.16
    bias_reach = top + 0.40 * layout.row_gap + 0.8 * 0.15 * layout.row_gap

    fig, ax = plt.subplots()
    format_network_axes(ax, layout, title_inside=False, caption_lift=max(
        0.0, 0.40 * layout.row_gap + 0.8 * 0.15 * layout.row_gap + 0.12
        - CAPTION_OFFSET))
    captions = [text for text in ax.texts if text.get_text() in layout.labels]

    assert captions
    assert all(text.get_position()[1] > bias_reach for text in captions)
    assert ax.get_ylim()[1] > max(t.get_position()[1] for t in captions)
    plt.close(fig)
