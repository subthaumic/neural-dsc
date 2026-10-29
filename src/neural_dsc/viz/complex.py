"""The visual language shared by network and simplicial-complex figures.

One language, used by every DSC figure in the paper. A complex is drawn on the
network it was built from, because a simplex is a statement about particular
neurons and an abstract vertex set is a worse picture of one. The input and
output layers are drawn too, paler in a DSC because they carry no simplices,
but at the same size and spacing as every other layer: half a network is not a
network, and changing its geometry changes what the picture says.

Individual neurons are unlabelled. Which layer a neuron is in is already said
by where it sits, and saying it twice only adds a legend.

Colour is filtration value throughout -- edges, 2-simplex patches, and the
predictor arrows inside them -- so one colourbar reads the whole picture.
"""

from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.path import Path
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)


# Edge curvature. Reciprocal pairs bow far enough apart to be told apart; a
# lone edge bows just enough not to look like a ruler.
RAD_RECIPROCAL = 0.14
RAD_SINGLE = 0.035

# Curvature of the dashed predictor arrow inside a 2-simplex.
RAD_PREDICTOR = 0.06

# One geometry describes the underlying network in every renderer.
#
# Rows sit one unit apart. The hidden columns sit further apart than that,
# because every simplex lives between them; the boundary columns sit closer in,
# and smaller, because they carry none. Equal steps everywhere were tried and
# are worse: at a uniform 1.80 the whole complex is squeezed into a third of the
# panel while the two boundary bands hold the rest empty, and full-size boundary
# neurons compete with the hidden block instead of framing it.
#
# The wish behind the uniform step is still a real one -- see the note in
# scripts/README.md on distributing these panels more evenly. Collapsing the two
# gaps into one is not the way to get there.
HIDDEN_GAP = 1.80
BOUNDARY_GAP = 0.95

NEURON_RADIUS = 0.15
BOUNDARY_RADIUS_SCALE = 0.66
HIDDEN_FACE = "#dfe6ec"
BOUNDARY_FACE = "#f0f0f0"
HIDDEN_EDGE = "#333333"
BOUNDARY_EDGE = "#999999"

# Panel margins, in row units. Layer captions always run along the foot; putting
# them above in some renderers and below in others was a particularly subtle way
# to make identical architectures look unrelated. DSC panel names go inside at
# top left so a multi-panel grid retains an unambiguous title for each cell.
SIDE_MARGIN = 0.26
BOTTOM_MARGIN = 0.30
CAPTION_OFFSET = 0.34
TITLE_OFFSET = 0.78
TOP_MARGIN_TITLED = 1.16
TOP_MARGIN_CAPTIONED = 0.72
TOP_MARGIN_BARE = 0.30

# A light rounded frame around each panel. In a grid it is what finally makes
# "these nodes belong to this result" a statement the eye gets for free, rather
# than one it has to infer from spacing.
# Each hidden neuron carries its number. Simplices are read off the complex as
# index tuples, so without this a reader cannot tell which node a simplex is
# about. Small and inside the node, because it is a name for the node and not a
# quantity to be compared across them.
#
# Displayed one-indexed, matching the manuscript's $a_1 \dots n_8$. Everything
# under the hood -- flat indices, simplex tuples, filenames, the saved complexes
# -- stays zero-indexed, exactly as Q1..Q4 name the classes that are stored as
# 0..3. This offset is the only place the two conventions meet.
NODE_LABEL_SIZE = 6.5
NODE_LABEL_OFFSET = 1

PANEL_FRAME = "#c3cad2"
PANEL_ROUNDING = 0.22
TITLE_INSET = 0.20

PATCH_ALPHA = 0.18

# Node fill when a panel is asked to show mean activation. The same ramp the
# dedicated activation figures use, so a node reads the same way in both. Not
# viridis, which is already spent on filtration value in the very same panel.
ACTIVATION_CMAP = "coolwarm"


def patch_alpha(count):
    """Fill opacity for `count` 2-simplices in one panel.

    A handcrafted complex has two to four of them and gets the full value. A
    trained one has seventy, and at full value they composite to an opaque slab
    that says only "many", so the total ink is held roughly fixed instead. Where
    the ink concentrates still shows which neurons the higher-order structure
    sits on, which is the thing the fill is for.
    """
    return max(0.03, min(PATCH_ALPHA, 0.9 / max(1, count)))


def predictor_style(count):
    """Opacity and width of the dashed predictor arrows, thinned when crowded."""
    return (max(0.25, min(0.9, 6.0 / max(1, count))),
            max(0.7, min(1.5, 8.0 / max(1, count))))

# Rows normally sit one unit apart. A very tall network gets its rows squeezed
# rather than allowing one column to dwarf the rest of the figure.
MAX_COLUMN_HEIGHT = 8.0

Layout = namedtuple(
    "Layout", "hidden inputs outputs columns labels row_gap xlim ylim aspect")


def as_complex_dict(sc):
    """Accept a DirectedSimplicialComplex or the plain dimension -> list dict."""
    return sc.complex if hasattr(sc, "complex") else sc


def neuron_layout(layer_sizes):
    """Positions for every neuron, hidden ones keyed by flat activation index.

    The flat index runs layer-major over the hidden layers, which is the
    convention `MLP.get_unflat_idx` inverts, so a simplex read off a complex
    indexes straight into `Layout.hidden`.

    Every column is centred on the same midline, so a layer with three neurons
    sits symmetrically beside one with four.
    """
    layer_sizes = list(layer_sizes)
    hidden_sizes = layer_sizes[1:-1]
    row_gap = min(1.0, MAX_COLUMN_HEIGHT / max(1, max(layer_sizes) - 1))
    midline = 1.5

    def column(x, size):
        top = midline + row_gap * (size - 1) / 2.0
        return [(x, top - row_gap * index) for index in range(size)]

    hidden_x = [BOUNDARY_GAP + HIDDEN_GAP * depth
                for depth in range(len(hidden_sizes))]
    columns = [0.0] + hidden_x + [(hidden_x[-1] if hidden_x else 0.0)
                                  + BOUNDARY_GAP]
    hidden, flat = {}, 0
    for x, size in zip(hidden_x, hidden_sizes):
        for point in column(x, size):
            hidden[flat] = point
            flat += 1

    inputs = column(columns[0], layer_sizes[0])
    outputs = column(columns[-1], layer_sizes[-1])
    labels = ["In"] + [f"HL{depth + 1}" for depth in range(len(hidden_sizes))] \
        + ["Out"]

    every_y = [y for _, y in inputs + outputs + list(hidden.values())]
    xlim = (min(columns) - SIDE_MARGIN, max(columns) + SIDE_MARGIN)
    ylim = (min(every_y) - BOTTOM_MARGIN, max(every_y) + TOP_MARGIN_TITLED)
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    return Layout(hidden, inputs, outputs, columns, labels, row_gap,
                  xlim, ylim, aspect)


def label_colour(face):
    """Black or white, whichever the fill can actually be read against."""
    red, green, blue = mpl.colors.to_rgb(face)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#ffffff" if luminance < 0.55 else "#333333"


def panel_frame(ax):
    """Round off the panel's own bounds. Call once the limits are final."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={PANEL_ROUNDING}",
        facecolor="none", edgecolor=PANEL_FRAME, linewidth=1.0,
        zorder=-1, clip_on=False,
    ))


def format_network_axes(ax, layout, title=None, layer_labels=True,
                        title_fontsize=11, title_inside=True, frame=True,
                        caption_lift=0.0):
    """Apply the shared bounds, aspect, layer captions, and optional title.

    Layer captions are deliberately owned by this helper rather than by either
    renderer. This keeps their wording and their position above the network
    identical in architecture, activation, and DSC figures -- they head their
    columns, the way a column heading does.

    `caption_lift` is for a renderer that draws its own furniture above the top
    row of neurons -- the parameter view hangs a bias marker over every one of
    them. Without the lift the caption lands on those markers, since the caption
    band is measured from the neurons, not from whatever is stacked on them.
    """
    top = layout.ylim[1] - TOP_MARGIN_TITLED
    if layer_labels:
        for x, label in zip(layout.columns, layout.labels):
            ax.text(x, top + CAPTION_OFFSET + caption_lift, label,
                    ha="center", va="bottom", fontsize=8, color="#666666")
    if title is not None:
        if title_inside:
            ax.text(layout.xlim[0] + TITLE_INSET,
                    top + TITLE_OFFSET + caption_lift, title,
                    ha="left", va="bottom", fontsize=title_fontsize,
                    color="#222222")
        else:
            ax.set_title(title, fontsize=title_fontsize, pad=8)

    if title is not None and title_inside:
        upper = layout.ylim[1] + caption_lift
    elif layer_labels:
        upper = top + TOP_MARGIN_CAPTIONED + caption_lift
    else:
        upper = top + max(TOP_MARGIN_BARE, caption_lift)
    ax.set_xlim(*layout.xlim)
    ax.set_ylim(layout.ylim[0], upper)
    ax.set_aspect("equal")
    ax.axis("off")
    if frame:
        panel_frame(ax)
    return ax


def arc_control(start, end, rad):
    """The control point matplotlib's ``arc3`` uses, so a fill can follow it.

    Reimplemented rather than read back off the patch because the fills have to
    be built before the arrows are drawn, and because a 2-simplex traverses one
    of its sides backwards -- the arc from u to v at rad r is the arc from v to
    u at -r, which is only usable if the construction is explicit.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    return (start + end) / 2.0 + rad * np.array([delta[1], -delta[0]])


def arc_points(start, end, rad, samples=48):
    """Quadratic Bezier from start to end, the curve ``arc3`` actually draws."""
    control = arc_control(start, end, rad)
    t = np.linspace(0.0, 1.0, samples).reshape(-1, 1)
    start, end = np.asarray(start, float), np.asarray(end, float)
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end


def trimmed_arc(start, end, rad, gap_start, gap_end, samples=64):
    """The arc from start to end, clipped clear of a disc at each end.

    The gaps are in data units, so an arrow stops at the rim of its neuron
    whatever size the figure is drawn at. Doing this with FancyArrowPatch's
    `shrinkA`/`shrinkB` instead would fix the gap in points, which silently
    stops matching the neurons the moment a panel is resized.
    """
    points = arc_points(start, end, rad, samples)
    clear_start = np.linalg.norm(
        points - np.asarray(start, float), axis=1) >= gap_start
    clear_end = np.linalg.norm(
        points - np.asarray(end, float), axis=1) >= gap_end
    keep = clear_start & clear_end
    return points[keep] if keep.sum() >= 2 else None


def arc_arrow(points, **kwargs):
    """A FancyArrowPatch along an already-computed polyline."""
    return FancyArrowPatch(path=Path(points), **kwargs)


def edge_filtrations(complex_dict):
    """Filtration value of each drawn edge, keyed by the ordered pair."""
    values = {}
    for simplex, filtration in complex_dict.get(1, []):
        if len(simplex) != 2:
            continue
        edge = (int(simplex[0]), int(simplex[1]))
        values[edge] = min(float(filtration), values.get(edge, np.inf))
    return values


def edge_radii(complex_dict):
    """The rad each drawn edge gets, keyed by the ordered pair."""
    edges = set(edge_filtrations(complex_dict))
    return {
        edge: (RAD_RECIPROCAL if (edge[1], edge[0]) in edges else RAD_SINGLE)
        for edge in edges
    }


def traversal_rad(radii, u, v):
    """The rad that traces the drawn u--v edge in the direction u to v.

    A 2-simplex is bounded by its own faces and by nothing else, so the side
    that ``(a, b, c)`` shares with the reciprocal ordering is the *directed*
    edge each one actually contains. Following it exactly is what lets the two
    orderings be told apart: they differ precisely in which arc of the
    reciprocal pair they rest on.
    """
    if (u, v) in radii:
        return radii[(u, v)]
    if (v, u) in radii:
        return -radii[(v, u)]
    return 0.0


def directed_two_simplices(complex_dict):
    """Every ordered 2-simplex, kept ordered. No collapsing of orientations."""
    values = {}
    for simplex, filtration in complex_dict.get(2, []):
        if len(simplex) != 3:
            continue
        key = tuple(int(vertex) for vertex in simplex)
        values[key] = min(float(filtration), values.get(key, np.inf))
    return sorted(values.items())


def two_simplex_boundary(simplex, positions, radii):
    """The closed path bounding an ordered 2-simplex, along its own faces.

    The faces of ``(a, b, c)`` are ``(a, b)``, ``(b, c)`` and ``(a, c)``, so the
    walk goes a -> b -> c and then back along ``(a, c)`` reversed. The patch
    therefore ends exactly where the drawn arrows do instead of cutting straight
    chords across them.

    That last side is asked for as ``(a, c)`` and negated, never as ``(c, a)``.
    When the complex holds the reciprocal edge too -- which it usually does --
    ``(c, a)`` is a different drawn arc, and closing along it bounds the region
    with three edges that are not this simplex's faces: it shows a triangle
    a -> b -> c -> a that no ordered 2-simplex has.
    """
    a, b, c = simplex
    return np.vstack([
        arc_points(positions[a], positions[b], traversal_rad(radii, a, b)),
        arc_points(positions[b], positions[c], traversal_rad(radii, b, c)),
        arc_points(positions[c], positions[a], -traversal_rad(radii, a, c)),
    ])


def draw_dsc(ax, sc, layer_sizes, norm, cmap, title=None, layout=None,
             layer_labels=True, title_fontsize=11, node_values=None,
             input_values=None, output_values=None, node_norm=None,
             node_cmap=ACTIVATION_CMAP, frame=True, node_labels=True):
    """Draw one complex on its network. Returns the layout it used.

    `node_values` optionally fills the hidden neurons by some per-neuron
    quantity -- mean activation on the context's probe set, say -- keyed by flat
    index. `input_values` and `output_values` do the same for the boundary
    columns, in column order. Pass a shared `node_norm` across a grid, or panels
    cannot be compared with each other.

    Colouring all three from one ramp is a deliberate simplification: mean input
    coordinate, mean hidden activation and mean output share a scale here but
    not a unit, so the fills say which units are driven hard relative to one
    another and nothing finer.
    """
    complex_dict = as_complex_dict(sc)
    layout = layout if layout is not None else neuron_layout(layer_sizes)
    positions = layout.hidden
    radii = edge_radii(complex_dict)
    filtrations = edge_filtrations(complex_dict)
    radius = NEURON_RADIUS * layout.row_gap

    # Higher-order structure goes behind the 1-skeleton. Both orderings of a
    # predictor pair are drawn; where they overlap the fill simply doubles, and
    # the lens between the two reciprocal arcs is the visible difference.
    two_simplices = directed_two_simplices(complex_dict)
    fill_alpha = patch_alpha(len(two_simplices))
    arrow_alpha, arrow_width = predictor_style(len(two_simplices))
    for simplex, filtration in two_simplices:
        if not all(vertex in positions for vertex in simplex):
            continue
        first, second, target = simplex
        colour = cmap(norm(filtration))
        ax.add_patch(Polygon(
            two_simplex_boundary(simplex, positions, radii), closed=True,
            facecolor=colour, edgecolor=colour, linewidth=1.0,
            alpha=fill_alpha, zorder=0,
        ))

        # The dashed arrow runs from the predictor edge to the target, keeping
        # the direction an undirected fill would discard, and starts on the
        # curve rather than on the chord so it visibly leaves its own edge.
        edge_midpoint = arc_points(
            positions[first], positions[second],
            traversal_rad(radii, first, second), samples=3)[1]
        spine = trimmed_arc(edge_midpoint, positions[target], RAD_PREDICTOR,
                            0.0, radius)
        if spine is not None:
            ax.add_patch(arc_arrow(
                spine, arrowstyle="-|>", mutation_scale=11, linestyle="--",
                linewidth=arrow_width, color=colour, alpha=arrow_alpha,
                zorder=1,
            ))

    for edge, rad in sorted(radii.items()):
        if not all(vertex in positions for vertex in edge):
            continue
        spine = trimmed_arc(positions[edge[0]], positions[edge[1]], rad,
                            radius, radius)
        if spine is None:
            continue
        ax.add_patch(arc_arrow(
            spine, arrowstyle="-|>", mutation_scale=12, linewidth=1.7,
            color=cmap(norm(filtrations[edge])), zorder=2,
        ))

    fills = mpl.colormaps[node_cmap] if node_values is not None else None

    def shade(value):
        return fills(node_norm(value) if node_norm is not None else value)

    boundary = [(layout.inputs, input_values), (layout.outputs, output_values)]
    for points, values in boundary:
        for index, point in enumerate(points):
            face = BOUNDARY_FACE
            if fills is not None and values is not None and index < len(values):
                face = shade(values[index])
            ax.add_patch(Circle(point, radius=radius * BOUNDARY_RADIUS_SCALE,
                                facecolor=face, edgecolor=BOUNDARY_EDGE,
                                linewidth=1.0, zorder=3))
    for index, point in positions.items():
        if fills is not None and index in node_values:
            face = shade(node_values[index])
        else:
            face = HIDDEN_FACE
        ax.add_patch(Circle(point, radius=radius, facecolor=face,
                            edgecolor=HIDDEN_EDGE, linewidth=1.0, zorder=3))
        if node_labels:
            ax.text(point[0], point[1], str(index + NODE_LABEL_OFFSET),
                    ha="center", va="center", fontsize=NODE_LABEL_SIZE,
                    color=label_colour(face), zorder=4)

    format_network_axes(
        ax, layout, title=title, layer_labels=layer_labels,
        title_fontsize=title_fontsize, title_inside=True, frame=frame,
    )
    return layout


class SimplexKey:
    """Sentinel handle for one simplex entry in the shared key."""

    def __init__(self, dimension):
        self.dimension = dimension


class SimplexHandler(HandlerBase):
    """Draw each swatch as the thing itself, not as a description of it.

    Both entries are arrows that terminate in a head at their endpoint, because
    that is what a simplex looks like in these panels -- an arrow *ending* at
    its target, not a line with an arrow marker somewhere along it. The
    2-simplex adds the patch its dashed predictor arrow sits inside, since the
    patch and the arrow are one element and not two.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        middle = y0 + 0.5 * height
        artists = []
        if orig_handle.dimension >= 2:
            artists.append(Rectangle(
                (x0, y0), width, height, facecolor="#888888",
                edgecolor="#555555", alpha=PATCH_ALPHA, linewidth=1.0))
            tail, tip, style, linewidth = (
                x0 + 0.14 * width, x0 + 0.86 * width, "--", 1.4)
        else:
            tail, tip, style, linewidth = x0, x0 + width, "-", 1.7
        artists.append(FancyArrowPatch(
            (tail, middle), (tip, middle), arrowstyle="-|>", mutation_scale=10,
            linestyle=style, linewidth=linewidth, color="#555555",
            shrinkA=0, shrinkB=0,
        ))
        for artist in artists:
            artist.set_transform(trans)
        return artists


def dsc_legend_handles():
    """Handles, labels and handler map for the shared key.

    Two entries, because the picture has two things in it.
    """
    return (
        [SimplexKey(1), SimplexKey(2)],
        ["1-simplex", "2-simplex"],
        {SimplexKey: SimplexHandler()},
    )


def panel_figsize(layout, columns=1, rows=1, panel_height=4.1,
                  furniture_height=0.95, side_width=0.0):
    """A figure whose panel grid has the same aspect as the panels themselves.

    Every panel is drawn at equal aspect, so a grid of any other shape
    letterboxes each one: the axes gets a rectangle, the data box is fitted
    inside it, and the leftover becomes gutter. Sizing the figure from the
    layout removes that rather than hand-tuning a figsize until it looks close.

    `furniture_height` pays for the suptitle and the key beneath the panels.
    Leave them unpaid and the axes lose that height instead; equal aspect then
    takes the width away to match, and the missing width comes back as a gutter
    between the columns -- the same bug one level up.
    """
    return (columns * panel_height * layout.aspect + side_width,
            rows * panel_height + furniture_height)


def add_dsc_key(fig, axes, norm, cmap, shrink=0.34, aspect=38, gap=0.012,
                label="Filtration value $s^*$"):
    """The legend, and the colourbar horizontally beneath it, under the panels.

    Order matters and matplotlib will not give it for free: constrained_layout
    reserves the bottom band for the colourbar, and a figure legend anchored
    below the figure would land under it. So the colourbar goes first with
    enough pad to open a gap, the layout is settled and then frozen, and the
    legend is anchored off the colourbar's measured position -- which is why
    this reads bottom-up rather than top-down.

    Anything added to the figure after this call is placed against a frozen
    layout, so add the suptitle first.
    """
    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colourbar = fig.colorbar(
        scalar_map, ax=axes, location="bottom", orientation="horizontal",
        shrink=shrink, aspect=aspect, pad=0.10,
    )
    colourbar.set_label(label, fontsize=9)
    colourbar.ax.tick_params(labelsize=7, length=2.5, width=0.7)
    colourbar.outline.set_linewidth(0.6)

    fig.canvas.draw()
    box = colourbar.ax.get_position()
    fig.set_layout_engine("none")
    handles, labels, handler_map = dsc_legend_handles()
    fig.legend(
        handles=handles, labels=labels, handler_map=handler_map,
        loc="lower center", ncol=2, frameon=False, fontsize=9,
        handlelength=2.8, handleheight=1.4, columnspacing=2.6,
        bbox_to_anchor=(0.5, box.y1 + gap),
    )
    return colourbar


def plot_simplicial_complex(
    sc, model, title=None, save_path=None, show_plot=True, ax=None, **kwargs
):
    """Draw a filtered directed simplicial complex on the network it came from.

    Args:
        sc: a DirectedSimplicialComplex, or the dimension -> [(simplex,
            filtration), ...] dictionary.
        model: MLP instance, read for `layer_sizes`.
        title: title of the panel.
        save_path: where to save, if anywhere.
        ax: optional axes to draw on.
        values: optional `PanelValues` colouring the nodes by mean activation.
    Returns:
        The matplotlib axes with the plot.
    """
    complex_dict = as_complex_dict(sc)
    layout = neuron_layout(model.layer_sizes)
    cmap = mpl.colormaps[kwargs.get("cmap", "viridis")]
    filtrations = [f for _, f in edge_filtrations(complex_dict).items()]
    filtrations += [f for _, f in directed_two_simplices(complex_dict)]
    norm = mpl.colors.Normalize(
        vmin=kwargs.get("vmin", 0),
        vmax=kwargs.get("vmax", max(filtrations) if filtrations else 1.0),
    )

    values = kwargs.get("values")
    created_here = ax is None
    if created_here:
        fig, ax = plt.subplots(
            figsize=panel_figsize(layout, panel_height=4.3),
            constrained_layout=True,
        )
    draw_dsc(ax, complex_dict, model.layer_sizes, norm, cmap, title=title,
             layout=layout, layer_labels=kwargs.get("layer_labels", True),
             title_fontsize=kwargs.get("title_fontsize", 12),
             frame=kwargs.get("frame", True),
             node_norm=kwargs.get("node_norm"),
             **(values.fill_kwargs() if values is not None else {}))

    if created_here and filtrations and not kwargs.get("hide_colorbar", False):
        add_dsc_key(ax.figure, ax, norm, cmap,
                    shrink=kwargs.get("colorbar_shrink", 0.52))

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    elif created_here and save_path is not None:
        # Draw-and-write is the only thing most callers want, and leaving the
        # figure open makes a loop over classes hit matplotlib's open-figure
        # warning and hold every one of them in memory.
        plt.close(ax.figure)
    return ax
