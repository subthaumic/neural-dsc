"""
Compare class separation in DSC space against activation-vector space.

This fixed-config post-processing script consumes existing experiment outputs
from the canonical quadrant producers. It restricts each comparison to
the one-hot class DSCs, integrates Jaccard distances between their sublevel
simplex sets over the filtration, computes cosine/Euclidean distances between
class-mean internal activations over the same saved probe sets, and writes
summary tables and matrix figures.

USAGE:
  python scripts/plot_dsc_vs_activation_separation.py

To change run parameters, edit the CONFIGURATION block below.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from neural_dsc.utils.progress import RunTimer
from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.io.persistence import load_probe_data
from neural_dsc.mlp import MLP
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.utils.jaccard_index import (
    JACCARD_FILTRATION_GRID,
    JACCARD_FILTRATION_INTERVALS,
    JACCARD_FILTRATION_POINTS,
    compute_jaccard_distance,
)


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
FIGS_DIR = REPO_ROOT / "figs" / "dsc_vs_activation"

# The quadrant experiments use K=1000 ablation intervals, hence K+1=1001
# filtration points including both endpoints. The requested discrete integral is
# the arithmetic mean over these points, so every sublevel-set distance has
# weight 1 / 1001.
FILTRATION_GRID_INTERVALS = JACCARD_FILTRATION_INTERVALS
FILTRATION_GRID_POINTS = JACCARD_FILTRATION_POINTS
FILTRATION_GRID = JACCARD_FILTRATION_GRID


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    results_dir: Path
    n_classes: int
    labels: tuple[str, ...]
    model_kind: str
    model_paths: tuple[Path, ...]
    layer_sizes: tuple[int, ...] | None = None
    # Figure title, named as in the article: "monosemantic" and "polysemantic"
    # only for the handcrafted networks, layer widths for the trained ones.
    title: str | None = None


# CONFIGURATION
EXPERIMENTS = (
    ExperimentConfig(
        name="quadrants_mono_handcrafted",
        results_dir=RESULTS_DIR / "quadrants_mono_handcrafted",
        n_classes=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        model_kind="quadrants_mono",
        model_paths=(MODELS_DIR / "quadrants_mono_handcrafted.pth",),
        title="Monosemantic",
    ),
    ExperimentConfig(
        name="quadrants_poly_handcrafted",
        results_dir=RESULTS_DIR / "quadrants_poly_handcrafted",
        n_classes=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        model_kind="quadrants_poly",
        model_paths=(MODELS_DIR / "quadrants_poly_handcrafted.pth",),
        title="Polysemantic",
    ),
    ExperimentConfig(
        name="quadrants_mono_trained",
        results_dir=RESULTS_DIR / "quadrants_mono_trained",
        n_classes=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        model_kind="quadrants_mono",
        model_paths=(MODELS_DIR / "quadrants_mono_trained.pth",),
        title="Trained [2,4,4,4]",
    ),
    ExperimentConfig(
        name="quadrants_poly_trained",
        results_dir=RESULTS_DIR / "quadrants_poly_trained",
        n_classes=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        model_kind="quadrants_poly",
        model_paths=(MODELS_DIR / "quadrants_poly_trained.pth",),
        title="Trained [2,3,4,4]",
    ),
)

OUTPUT_DIR = RESULTS_DIR / "dsc_vs_activation"


def load_model(config: ExperimentConfig) -> MLP:
    if config.model_kind == "quadrants_mono":
        model = create_quadrants_mono()
        model_path = first_existing_path(config.model_paths)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    if config.model_kind == "quadrants_poly":
        model = create_quadrants_poly()
        model_path = first_existing_path(config.model_paths)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    if config.model_kind == "generic_mlp":
        errors = []
        for model_path in config.model_paths:
            if not model_path.exists():
                continue
            try:
                loaded = torch.load(model_path, weights_only=False)
            except Exception as exc:
                errors.append(f"{model_path}: {exc}")
                continue
            if isinstance(loaded, MLP):
                loaded.eval()
                return loaded
            if isinstance(loaded, dict):
                if config.layer_sizes is None:
                    raise ValueError(
                        f"Missing layer sizes for state dict model {config.name}"
                    )
                model = MLP(layer_sizes=list(config.layer_sizes))
                model.load_state_dict(loaded)
                model.eval()
                return model
            errors.append(f"{model_path}: unsupported type {type(loaded)}")

        message = (
            f"No usable model checkpoint found for {config.name}: "
            + ", ".join(str(path) for path in config.model_paths)
        )
        if errors:
            message += " (" + "; ".join(errors) + ")"
        raise FileNotFoundError(message)

    raise ValueError(f"Unknown model kind: {config.model_kind}")


def repo_relative(path: Path) -> str:
    """The path relative to the repository, so that no local path is stored."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def first_existing_path(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_class_probe_data(config: ExperimentConfig) -> list[torch.Tensor]:
    path = config.results_dir / "probe_data_onehot.pt"
    probe_data = load_probe_data(str(path))
    if probe_data is None:
        raise FileNotFoundError(f"Missing one-hot probe data: {path}")
    if len(probe_data) < config.n_classes:
        raise ValueError(
            f"{path} has {len(probe_data)} probe sets, expected at least {config.n_classes}"
        )
    return probe_data[: config.n_classes]


def load_complex(path: Path) -> DirectedSimplicialComplex:
    sc = DirectedSimplicialComplex()
    sc.load(str(path))
    return sc


def load_class_complexes(
    config: ExperimentConfig,
) -> list[DirectedSimplicialComplex]:
    complexes = []
    for class_idx in range(config.n_classes):
        path = config.results_dir / f"dsc_class_{class_idx}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {config.name} class DSC: {path}")
        complexes.append(load_complex(path))
    return complexes


def nonvertex_filtration_map(
    complex_: DirectedSimplicialComplex,
) -> dict[tuple[int, ...], float]:
    """Map every non-vertex simplex to its filtration value."""
    return {
        tuple(simplex): float(filtration)
        for dimension, simplices in complex_.complex.items()
        if dimension > 0
        for simplex, filtration in simplices
    }


def jaccard_distance(left: set[tuple[int, ...]], right: set[tuple[int, ...]]) -> float:
    """Jaccard distance, with identical empty sets at distance zero."""
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def terminal_jaccard_distance(
    left: DirectedSimplicialComplex,
    right: DirectedSimplicialComplex,
) -> float:
    """The previous analysis: Jaccard distance between completed simplex sets."""
    return jaccard_distance(
        set(nonvertex_filtration_map(left)),
        set(nonvertex_filtration_map(right)),
    )


def filtration_integrated_jaccard_distance(
    left: DirectedSimplicialComplex,
    right: DirectedSimplicialComplex,
    filtration_grid: np.ndarray = FILTRATION_GRID,
) -> float:
    """Average sublevel-set Jaccard distance over the filtration grid.

    Vertices remain excluded, matching the previous analysis. Before either
    complex contains a non-vertex simplex, both sublevel sets are empty and
    therefore identical. Because the saved quadrant filtration values lie on
    the 0.001 grid, this discrete average distinguishes every recorded value.
    """
    return float(
        compute_jaccard_distance(left, right, filtration_grid=filtration_grid)
    )


def pairwise_dsc_distances(
    complexes: list[DirectedSimplicialComplex],
) -> tuple[np.ndarray, np.ndarray]:
    """Return filtration-integrated and previous terminal Jaccard matrices."""
    n_complexes = len(complexes)
    integrated = np.zeros((n_complexes, n_complexes), dtype=float)
    terminal = np.zeros((n_complexes, n_complexes), dtype=float)
    for i in range(n_complexes):
        for j in range(i + 1, n_complexes):
            integrated_ij = filtration_integrated_jaccard_distance(
                complexes[i], complexes[j]
            )
            terminal_ij = terminal_jaccard_distance(complexes[i], complexes[j])
            integrated[i, j] = integrated[j, i] = integrated_ij
            terminal[i, j] = terminal[j, i] = terminal_ij
    return integrated, terminal


def class_mean_activations(
    model: MLP,
    probe_data: list[torch.Tensor],
) -> np.ndarray:
    means = []
    for class_idx, points in enumerate(probe_data):
        if points.numel() == 0:
            raise ValueError(f"Class {class_idx} has no probe points")
        with torch.no_grad():
            activations = model.get_activations(points)
        means.append(torch.mean(activations, dim=0).detach().cpu().numpy())
    return np.vstack(means)


def pairwise_cosine_distance(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    denom = np.outer(norms, norms)
    similarity = np.zeros((vectors.shape[0], vectors.shape[0]), dtype=float)
    np.divide(vectors @ vectors.T, denom, out=similarity, where=denom > 0)
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def pairwise_euclidean_distance(vectors: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - vectors[None, :, :]
    distance = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distance, 0.0)
    return distance


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got {matrix.shape}")
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def normalise_by_max_offdiag(matrix: np.ndarray) -> np.ndarray:
    values = off_diagonal_values(matrix)
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 0:
        return matrix.copy()
    return matrix / max_value


def matrix_distribution(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    values = off_diagonal_values(matrix)
    if values.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def spearman_distance_correlation(
    left_distance: np.ndarray,
    right_distance: np.ndarray,
) -> tuple[float, float]:
    left_values = off_diagonal_values(left_distance)
    right_values = off_diagonal_values(right_distance)
    if np.std(left_values) == 0 or np.std(right_values) == 0:
        return math.nan, math.nan
    result = spearmanr(left_values, right_values)
    return float(result.statistic), float(result.pvalue)


def compute_headline_metrics(
    jaccard_distance: np.ndarray,
    cosine_distance: np.ndarray,
    euclidean_distance: np.ndarray,
) -> dict[str, Any]:
    jaccard_norm = normalise_by_max_offdiag(jaccard_distance)
    cosine_norm = normalise_by_max_offdiag(cosine_distance)
    euclidean_norm = normalise_by_max_offdiag(euclidean_distance)

    jaccard_values = off_diagonal_values(jaccard_distance)
    cosine_values = off_diagonal_values(cosine_distance)
    rho, pvalue = spearman_distance_correlation(
        jaccard_distance, cosine_distance
    )

    metrics: dict[str, Any] = {
        "mean_jaccard_distance": float(np.mean(jaccard_values)),
        "mean_cosine_distance": float(np.mean(cosine_values)),
        "mean_euclidean_distance": float(np.mean(off_diagonal_values(euclidean_distance))),
        "normalised_mean_jaccard_distance": float(np.mean(off_diagonal_values(jaccard_norm))),
        "normalised_mean_cosine_distance": float(np.mean(off_diagonal_values(cosine_norm))),
        "normalised_mean_euclidean_distance": float(
            np.mean(off_diagonal_values(euclidean_norm))
        ),
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
    }
    metrics["dsc_more_separated"] = (
        metrics["normalised_mean_jaccard_distance"]
        > metrics["normalised_mean_cosine_distance"]
    )
    metrics.update(matrix_distribution("jaccard_distance", jaccard_distance))
    metrics.update(matrix_distribution("cosine_distance", cosine_distance))
    metrics.update(matrix_distribution("euclidean_distance", euclidean_distance))
    return metrics


def plot_distance_matrices(
    config: ExperimentConfig,
    jaccard_distance: np.ndarray,
    cosine_distance: np.ndarray,
    euclidean_distance: np.ndarray,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    matrices = (
        ("Jaccard distance", jaccard_distance),
        ("Cosine distance", cosine_distance),
        ("Euclidean distance", euclidean_distance),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (title, matrix) in zip(axes, matrices):
        vmax = float(np.max(matrix)) if np.max(matrix) > 0 else 1.0
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(config.n_classes))
        ax.set_yticks(np.arange(config.n_classes))
        ax.set_xticklabels(config.labels, rotation=45, ha="right")
        ax.set_yticklabels(config.labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    fig.suptitle(config.title or config.name)
    fig.savefig(save_path)
    plt.close(fig)


def serialise_float(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value


def write_experiment_json(
    config: ExperimentConfig,
    jaccard_distance: np.ndarray,
    terminal_jaccard_distance: np.ndarray,
    cosine_distance: np.ndarray,
    euclidean_distance: np.ndarray,
    metrics: dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config.name}_distances.json"
    data = {
        "experiment": config.name,
        "source_results_dir": repo_relative(config.results_dir),
        "n_classes": config.n_classes,
        "labels": list(config.labels),
        "jaccard_method": {
            "name": "mean sublevel-set Jaccard distance over filtration",
            "grid_intervals": FILTRATION_GRID_INTERVALS,
            "grid_points": FILTRATION_GRID_POINTS,
            "summand_weight": 1.0 / FILTRATION_GRID_POINTS,
            "vertices_included": False,
            "empty_empty_distance": 0.0,
        },
        "metrics": {key: serialise_float(value) for key, value in metrics.items()},
        "matrices": {
            "jaccard_distance": jaccard_distance.tolist(),
            "terminal_jaccard_distance": terminal_jaccard_distance.tolist(),
            "cosine_distance": cosine_distance.tolist(),
            "euclidean_distance": euclidean_distance.tolist(),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def summary_row_for_error(config: ExperimentConfig, error: Exception) -> dict[str, Any]:
    row = base_summary_row(config)
    row["status"] = "skipped"
    row["error"] = str(error)
    return row


def base_summary_row(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "status": "ok",
        "source_results_dir": repo_relative(config.results_dir),
        "n_classes": config.n_classes,
        "mean_jaccard_distance": math.nan,
        "mean_cosine_distance": math.nan,
        "mean_euclidean_distance": math.nan,
        "normalised_mean_jaccard_distance": math.nan,
        "normalised_mean_cosine_distance": math.nan,
        "normalised_mean_euclidean_distance": math.nan,
        "mean_terminal_jaccard_distance": math.nan,
        "normalised_mean_terminal_jaccard_distance": math.nan,
        "mean_jaccard_distance_change": math.nan,
        "normalised_mean_jaccard_distance_change": math.nan,
        "terminal_spearman_rho": math.nan,
        "terminal_spearman_pvalue": math.nan,
        "spearman_rho": math.nan,
        "spearman_pvalue": math.nan,
        "dsc_more_separated": "",
        "error": "",
    }


def analyze_experiment(
    config: ExperimentConfig,
    output_dir: Path = OUTPUT_DIR,
    figs_dir: Path = FIGS_DIR,
) -> dict[str, Any]:
    probe_data = load_class_probe_data(config)
    complexes = load_class_complexes(config)
    jaccard_distance, terminal_distance = pairwise_dsc_distances(complexes)
    model = load_model(config)
    class_means = class_mean_activations(model, probe_data)
    cosine_distance = pairwise_cosine_distance(class_means)
    euclidean_distance = pairwise_euclidean_distance(class_means)

    metrics = compute_headline_metrics(
        jaccard_distance,
        cosine_distance,
        euclidean_distance,
    )
    terminal_normalised = normalise_by_max_offdiag(terminal_distance)
    metrics["mean_terminal_jaccard_distance"] = float(
        np.mean(off_diagonal_values(terminal_distance))
    )
    metrics["normalised_mean_terminal_jaccard_distance"] = float(
        np.mean(off_diagonal_values(terminal_normalised))
    )
    metrics["mean_jaccard_distance_change"] = (
        metrics["mean_jaccard_distance"]
        - metrics["mean_terminal_jaccard_distance"]
    )
    metrics["normalised_mean_jaccard_distance_change"] = (
        metrics["normalised_mean_jaccard_distance"]
        - metrics["normalised_mean_terminal_jaccard_distance"]
    )
    terminal_rho, terminal_pvalue = spearman_distance_correlation(
        terminal_distance, cosine_distance
    )
    metrics["terminal_spearman_rho"] = terminal_rho
    metrics["terminal_spearman_pvalue"] = terminal_pvalue

    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    # Drawn straight into figs/. It used to be drawn into results/ and copied
    # here, which left the same picture in two places and one of them wrong.
    figure_path = figs_dir / f"dsc_vs_activation_{config.name}.pdf"
    plot_distance_matrices(
        config,
        jaccard_distance,
        cosine_distance,
        euclidean_distance,
        figure_path,
    )

    write_experiment_json(
        config,
        jaccard_distance,
        terminal_distance,
        cosine_distance,
        euclidean_distance,
        metrics,
        output_dir,
    )

    row = base_summary_row(config)
    row.update(metrics)
    return row


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(base_summary_row(EXPERIMENTS[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {key: serialise_float(row.get(key, "")) for key in fieldnames}
            writer.writerow(clean_row)


def main() -> None:
    rows = []
    for config in EXPERIMENTS:
        print("=" * 80)
        print(f"Analyzing {config.name}")
        print("=" * 80)
        try:
            row = analyze_experiment(config)
            print(
                f"  mean integrated Jaccard={row['mean_jaccard_distance']:.4f} "
                f"(terminal={row['mean_terminal_jaccard_distance']:.4f}), "
                f"mean cosine={row['mean_cosine_distance']:.4f}, "
                f"Spearman rho={row['spearman_rho']:.4f} "
                f"(terminal={row['terminal_spearman_rho']:.4f}), "
                f"dsc_more_separated={row['dsc_more_separated']}"
            )
        except Exception as exc:
            row = summary_row_for_error(config, exc)
            print(f"  Skipped: {exc}")
        rows.append(row)

    summary_path = OUTPUT_DIR / "summary.csv"
    write_summary(rows, summary_path)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    with RunTimer("DSC vs activation separation"):
        main()
