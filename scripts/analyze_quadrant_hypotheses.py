"""Descriptive checks for the topological-signature hypotheses in the article."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_ROOT / "quadrant_hypothesis_checks.json"
ARCHITECTURES = (
    "quadrants_mono_handcrafted",
    "quadrants_poly_handcrafted",
)
ADJACENT_PAIRS = ((0, 1), (0, 3), (1, 2), (2, 3))
FILTRATION_INTERVALS = 1000


def load_complex(path: Path) -> dict[int, dict[tuple[int, ...], float]]:
    complex_: defaultdict[int, dict[tuple[int, ...], float]] = defaultdict(dict)
    dimension = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if match := re.match(r"## Dimension (\d+)", line):
            dimension = int(match.group(1))
            continue
        if dimension is None:
            continue
        if match := re.match(r"\(([0-9,]+)\),\s+([0-9.-]+)", line):
            simplex = tuple(int(value) for value in match.group(1).split(",") if value)
            complex_[dimension][simplex] = float(match.group(2))
    return dict(complex_)


def is_directed_face(face: tuple[int, ...], simplex: tuple[int, ...]) -> bool:
    vertices = iter(simplex)
    return all(any(candidate == vertex for candidate in vertices) for vertex in face)


def maximal_nonvertex_simplices(
    complex_: dict[int, dict[tuple[int, ...], float]],
) -> list[tuple[int, ...]]:
    simplices = [
        simplex
        for dimension, dimension_simplices in complex_.items()
        if dimension >= 1
        for simplex in dimension_simplices
    ]
    return [
        simplex
        for simplex in simplices
        if not any(
            len(other) > len(simplex) and is_directed_face(simplex, other)
            for other in simplices
        )
    ]


def filtration_union(
    first: dict[int, dict[tuple[int, ...], float]],
    second: dict[int, dict[tuple[int, ...], float]],
) -> dict[int, dict[tuple[int, ...], float]]:
    union: defaultdict[int, dict[tuple[int, ...], float]] = defaultdict(dict)
    for complex_ in (first, second):
        for dimension, simplices in complex_.items():
            for simplex, filtration in simplices.items():
                union[dimension][simplex] = min(
                    filtration,
                    union[dimension].get(simplex, float("inf")),
                )
    return dict(union)


def sublevel_nonvertices(
    complex_: dict[int, dict[tuple[int, ...], float]],
    filtration: float,
) -> set[tuple[int, ...]]:
    return {
        simplex
        for dimension, simplices in complex_.items()
        if dimension >= 1
        for simplex, birth in simplices.items()
        if birth <= filtration + 1e-12
    }


def filtration_jaccard_distance(
    first: dict[int, dict[tuple[int, ...], float]],
    second: dict[int, dict[tuple[int, ...], float]],
) -> float:
    distances = []
    for step in range(FILTRATION_INTERVALS + 1):
        filtration = step / FILTRATION_INTERVALS
        left = sublevel_nonvertices(first, filtration)
        right = sublevel_nonvertices(second, filtration)
        distances.append(0.0 if not left and not right else 1 - len(left & right) / len(left | right))
    return sum(distances) / len(distances)


def terminal_jaccard_distance(
    first: dict[int, dict[tuple[int, ...], float]],
    second: dict[int, dict[tuple[int, ...], float]],
) -> float:
    left = sublevel_nonvertices(first, 1.0)
    right = sublevel_nonvertices(second, 1.0)
    return 0.0 if not left and not right else 1 - len(left & right) / len(left | right)


def analyze_architecture(name: str) -> dict[str, object]:
    result_dir = RESULTS_ROOT / name
    onehot = [load_complex(result_dir / f"dsc_class_{index}.txt") for index in range(4)]

    incidence: defaultdict[int, int] = defaultdict(int)
    context_incidence: dict[str, dict[str, int]] = {}
    contexts_per_neuron: defaultdict[int, set[int]] = defaultdict(set)
    for context, complex_ in enumerate(onehot):
        counts: defaultdict[int, int] = defaultdict(int)
        maximal = maximal_nonvertex_simplices(complex_)
        for simplex in maximal:
            for neuron in set(simplex):
                counts[neuron] += 1
                incidence[neuron] += 1
                contexts_per_neuron[neuron].add(context)
        context_incidence[f"Q{context + 1}"] = {
            f"n{neuron + 1}": count for neuron, count in sorted(counts.items())
        }

    union_checks = {}
    for first, second in ADJACENT_PAIRS:
        twohot = load_complex(result_dir / f"dsc_twohot_{first}{second}.txt")
        union = filtration_union(onehot[first], onehot[second])
        union_checks[f"Q{first + 1}+Q{second + 1}"] = {
            "integrated_jaccard_distance": filtration_jaccard_distance(twohot, union),
            "terminal_jaccard_distance": terminal_jaccard_distance(twohot, union),
        }

    return {
        "maximal_simplex_incidence_by_context": context_incidence,
        "maximal_simplex_incidence_total": {
            f"n{neuron + 1}": count for neuron, count in sorted(incidence.items())
        },
        "contexts_with_maximal_simplex_incidence": {
            f"n{neuron + 1}": len(contexts)
            for neuron, contexts in sorted(contexts_per_neuron.items())
        },
        "twohot_vs_filtration_union": union_checks,
    }


def main() -> None:
    results = {
        "method": {
            "maximal_simplex": "not an order-preserving face of a higher-dimensional directed simplex",
            "union_filtration": "minimum birth value across the two one-hot complexes",
            "jaccard": "mean non-vertex sublevel-set distance on the 1001-point filtration grid",
        },
        "architectures": {name: analyze_architecture(name) for name in ARCHITECTURES},
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
