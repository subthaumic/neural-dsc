"""Filtration-aware Jaccard comparisons for directed simplicial complexes."""

from collections.abc import Hashable, Mapping, Sequence

import numpy as np


JACCARD_FILTRATION_INTERVALS = 1000
JACCARD_FILTRATION_POINTS = JACCARD_FILTRATION_INTERVALS + 1
JACCARD_FILTRATION_GRID = np.linspace(0.0, 1.0, JACCARD_FILTRATION_POINTS)
JACCARD_FILTRATION_TOLERANCE = 1e-12
JACCARD_METHOD_ID = "filtration_integrated_1001_nonvertex_v1"


def _pointwise_jaccard_distance(left: set, right: set) -> float:
    """Jaccard distance, with two empty sets treated as identical."""
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def filtration_jaccard_distance(
    left: Mapping[Hashable, float],
    right: Mapping[Hashable, float],
    filtration_grid: Sequence[float] | np.ndarray = JACCARD_FILTRATION_GRID,
) -> float:
    """Average sublevel-set Jaccard distance over a filtration grid.

    ``left`` and ``right`` map simplex identities to filtration values. The
    arithmetic mean over the default 1001-point grid is the discrete integral
    over the normalised filtration domain [0, 1]. Empty sublevel sets have
    distance zero. The cumulative-count implementation is equivalent to
    constructing every sublevel set explicitly but avoids doing so 1001 times.
    """
    grid = np.asarray(filtration_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("filtration_grid must be a non-empty one-dimensional array")
    if np.any(np.diff(grid) < 0):
        raise ValueError("filtration_grid must be sorted")

    union_births = np.zeros(grid.size + 1, dtype=np.int64)
    intersection_births = np.zeros(grid.size + 1, dtype=np.int64)
    missing = float("inf")

    for simplex in left.keys() | right.keys():
        left_birth = float(left.get(simplex, missing))
        right_birth = float(right.get(simplex, missing))

        union_index = int(
            np.searchsorted(
                grid,
                min(left_birth, right_birth) - JACCARD_FILTRATION_TOLERANCE,
                side="left",
            )
        )
        if union_index < grid.size:
            union_births[union_index] += 1

        if np.isfinite(left_birth) and np.isfinite(right_birth):
            intersection_index = int(
                np.searchsorted(
                    grid,
                    max(left_birth, right_birth)
                    - JACCARD_FILTRATION_TOLERANCE,
                    side="left",
                )
            )
            if intersection_index < grid.size:
                intersection_births[intersection_index] += 1

    union_counts = np.cumsum(union_births[:-1])
    intersection_counts = np.cumsum(intersection_births[:-1])
    distances = np.zeros(grid.size, dtype=float)
    nonempty = union_counts > 0
    distances[nonempty] = 1.0 - (
        intersection_counts[nonempty] / union_counts[nonempty]
    )
    return float(np.mean(distances))


def _filtration_map(complex_obj, dim=None):
    """Map non-vertex simplex identities to filtration values."""
    dimensions = (dim,) if dim is not None else complex_obj.complex.keys()
    return {
        tuple(simplex): float(filtration)
        for dimension in dimensions
        if dimension > 0
        for simplex, filtration in complex_obj.complex.get(dimension, [])
    }


def compute_jaccard_distance(
    complex1,
    complex2,
    by_dimension=False,
    filtration_threshold=None,
    filtration_grid=JACCARD_FILTRATION_GRID,
):
    """Compute filtration-aware Jaccard distance between two DSCs.

    With no threshold, this averages the Jaccard distance between non-vertex
    sublevel sets at 1001 equally spaced filtration values in [0, 1]. Supplying
    ``filtration_threshold`` instead evaluates a single sublevel set. Vertices
    are always excluded because they are fixed and shared by construction.
    """

    def distance(dim=None):
        left = _filtration_map(complex1, dim)
        right = _filtration_map(complex2, dim)
        if filtration_threshold is None:
            return filtration_jaccard_distance(left, right, filtration_grid)
        threshold = float(filtration_threshold) + JACCARD_FILTRATION_TOLERANCE
        left_set = {simplex for simplex, value in left.items() if value <= threshold}
        right_set = {
            simplex for simplex, value in right.items() if value <= threshold
        }
        return _pointwise_jaccard_distance(left_set, right_set)

    if not by_dimension:
        return distance()

    dimensions = sorted((set(complex1.complex) | set(complex2.complex)) - {0})
    return {dimension: distance(dimension) for dimension in dimensions}


def compute_jaccard_index(
    complex1,
    complex2,
    by_dimension=False,
    filtration_threshold=None,
    filtration_grid=JACCARD_FILTRATION_GRID,
):
    """Compute Jaccard similarity as one minus the corresponding distance.

    This compatibility wrapper retains the historical function name. Unlike
    the former terminal simplex-set comparison, its default is filtration-aware
    and excludes vertices.
    """
    distance = compute_jaccard_distance(
        complex1,
        complex2,
        by_dimension=by_dimension,
        filtration_threshold=filtration_threshold,
        filtration_grid=filtration_grid,
    )
    if by_dimension:
        return {dimension: 1.0 - value for dimension, value in distance.items()}
    return 1.0 - distance
