"""
Untrained poly-capacity quadrant DSC baseline.

This fixed-config script runs the one-hot DSC construction workflow for
PyTorch-default random initialisations of MLP([2, 3, 4, 4]) across seeds
0, 1, and 2. The weights are seeded and saved for reproducibility, and no
training or handcrafted quadrant constructor is used. Probe points are copied
from the corresponding trained poly-capacity run so the trained and untrained
DSCs are compared on the same inputs.

USAGE:
  python scripts/compute_quadrants_poly_untrained.py

To change run parameters, edit the CONFIGURATION block below.

All twelve (seed, class) constructions run through one outer process pool. Each
individual DSC build is serial.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
import os

# MUST run before numpy/scipy/torch are imported, here and in every spawned
# child. OpenBLAS reads its thread count when the shared library loads, so
# setting it later -- as `pin_worker_threads` does inside the pool initializer
# -- is too late for BLAS. Without this, each of N worker processes spins up one
# BLAS thread per core: on a 52-core box with 48 workers that is ~2500 threads
# against a typical RLIMIT_NPROC of 4096, and OpenBLAS starts failing
# pthread_create. setdefault, so an explicit value from the environment wins.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.utils.progress import RunTimer
from neural_dsc.utils.parallel_dsc import (
    DSCBuildSpec,
    build_or_load_many,
    resolve_outer_workers,
)
from neural_dsc.utils.probe_selection import softmax_probabilities
from neural_dsc.utils.trained_quadrants import load_trained_quadrant_model
from neural_dsc.io.persistence import (
    check_config_fingerprint,
    load_probe_data,
    save_probe_data,
)
from neural_dsc.mlp import MLP
from neural_dsc.utils.dsc_analysis import (
    compute_simplex_statistics,
    create_analysis_summary,
    save_dsc_statistics_csv,
)
from neural_dsc.utils.jaccard_index import compute_jaccard_distance
from neural_dsc.viz.data import get_onehot_markers


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"


# CONFIGURATION
SEEDS = (0, 1, 2)
N_PROBE_POINTS = 128  # Per-experiment, not universal: the probe supply depends on
                      # the task. Here probes are drawn by rejection sampling from a
                      # continuous domain, so B is free; on a fixed dataset it is a
                      # budget. 128 is where the complex stops changing -- it recovers
                      # the B=512 complex exactly in all eight quadrant contexts,
                      # while 64 does so in only two and 8 finds nothing at all
                      # anywhere -- it is below the signed-rank feasibility floor
                      # for a family this size. Report the achieved
                      # Jaccard against the largest affordable B rather than assuming
                      # this number transfers.
RELEVANCE_MARGIN = 0.01  # interval null H0: |theta| <= epsilon, epsilon in
                         # units of the target neuron's within-class SD. The
                         # margin is a switch, not a dial: every epsilon from
                         # 0.001 to 0.2 gives an identical complex in all eight
                         # contexts under all four corrections, because the
                         # standardised effects are a gap rather than a
                         # continuum -- p75 = 0 against p90 = 2 at B = 512, so
                         # the whole band falls in empty space. Only epsilon = 0
                         # differs, and it disqualifies itself: it stops
                         # converging in probe count in all four poly contexts,
                         # and it is where the ragged trajectories are. 0.01 is
                         # a round number mid-plateau -- see
                         # compute_inference_sensitivity.py.
RELEVANCE_SCALE = "std"
PROBE_TOLERANCE = 0.01  # purity slack below the achievable cosine maximum
PROBE_SEED = 20260819
MAX_DIMENSION = 7  # the ceiling: 7 internal neurons
MAX_ITERATIONS = 1000  # p-values stabilise around K=500 (appendix B.1); 1000 is
                       # comfortably past that plateau.
# Width of the outer construction pool. None or -1 uses all CPUs but four.
# Individual DSC builds are serial because epsilon is non-zero and their
# candidate frontiers are too small to amortize an inner pool.
N_WORKERS = None
DT = 0.01
FDR_METHOD = "bonferroni"  # over the family of simplices tested at each
                   # dimension. The family is the dimension, not the ablation axis:
                   # membership is decided at full ablation alone, and reading
                   # the ablation axis top-down for the filtration is the
                   # fixed-sequence step-down MED procedure, which controls FWER
                   # over ablations with no further adjustment.
                   #
                   # Bonferroni rather than BH or BY because it assumes nothing
                   # about dependence -- simplices sharing a predictor set are
                   # scored off one shared trajectory, so PRDS is not
                   # established and BH does not apply -- and because it
                   # controls the familywise error rather than the false
                   # discovery rate, which is the stronger claim. It is not
                   # uniformly stricter than BY: alpha/m is flat where BY's
                   # (alpha/H_m) * k/m rises with the number of rejections, so
                   # the two cross at k = H_m. Calibrated in
                   # compute_inference_sensitivity.py, which sweeps all four.
SIGNIFICANCE_LEVEL = 0.01


ARCHITECTURE = "poly"
LAYER_SIZES = [2, 3, 4, 4]
OUTPUT_DIR = RESULTS_DIR / f"quadrants_{ARCHITECTURE}_untrained"
REFERENCE_RESULT_DIR = "quadrants_poly_trained"
REFERENCE_MODEL_PATH = "quadrants_poly_trained.pth"
PROBE_SOURCE_LABEL = f"trained:{REFERENCE_RESULT_DIR}/probe_data_onehot.pt"

CLASS_LABELS = ("Q1", "Q2", "Q3", "Q4")
COLORS = ["r", "g", "b", "purple"]


def build_untrained_model(seed: int) -> MLP:
    """Build an untrained MLP with PyTorch default initialisation."""
    torch.manual_seed(int(seed))
    return MLP(layer_sizes=list(LAYER_SIZES))


def load_or_create_untrained_model(seed: int) -> MLP:
    """Load a saved untrained state dict, or create and persist it."""
    model_path = MODELS_DIR / f"quadrants_{ARCHITECTURE}_untrained_seed{seed}.pth"
    model = build_untrained_model(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        print(f"Loading untrained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Creating untrained {ARCHITECTURE} model for seed {seed}")
        torch.save(model.state_dict(), model_path)
        print(f"Saved untrained model state dict to {model_path}")
    model.eval()
    return model


def load_trained_model() -> MLP:
    """Load the trained quadrant model for summary diagnostics."""
    model_path = MODELS_DIR / REFERENCE_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained reference model: {model_path}. "
            "Run the trained quadrant script first."
        )
    return load_trained_quadrant_model(model_path, LAYER_SIZES)


def output_dir_for(seed: int) -> Path:
    return OUTPUT_DIR / f"seed{seed}"


def load_complex(path: Path) -> DirectedSimplicialComplex:
    sc = DirectedSimplicialComplex()
    sc.load(str(path))
    return sc


def probe_data_equal(
    left: list[torch.Tensor] | None,
    right: list[torch.Tensor],
) -> bool:
    if left is None or len(left) != len(right):
        return False
    return all(
        left_points.shape == right_points.shape
        and torch.equal(left_points, right_points)
        for left_points, right_points in zip(left, right)
    )


def clear_probe_dependent_outputs(output_dir: Path) -> None:
    """Remove cached outputs that depend on the selected probe points."""
    patterns = [
        "dsc_class_*.txt",
        "dsc_class_*.pdf",
        "dsc_statistics_*.csv",
        "dsc_analysis_summary.txt",
        f"quadrants_{ARCHITECTURE}_untrained_seed*_onehot.pdf",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def load_reference_onehot_probe_data(output_dir: Path) -> list[torch.Tensor]:
    reference_probe_file = RESULTS_DIR / REFERENCE_RESULT_DIR / "probe_data_onehot.pt"
    reference_data = load_probe_data(str(reference_probe_file))
    if reference_data is None:
        raise FileNotFoundError(
            f"Missing trained reference probe data: {reference_probe_file}. "
            "Run the trained quadrant script first."
        )

    local_probe_file = output_dir / "probe_data_onehot.pt"
    local_source_file = output_dir / "probe_data_onehot_source.txt"
    local_data = load_probe_data(str(local_probe_file))
    local_source = (
        local_source_file.read_text().strip()
        if local_source_file.exists()
        else ""
    )

    if local_source != PROBE_SOURCE_LABEL or not probe_data_equal(local_data, reference_data):
        print(f"Using trained reference probe data from {reference_probe_file}")
        clear_probe_dependent_outputs(output_dir)
        save_probe_data(reference_data, str(local_probe_file))
        local_source_file.write_text(PROBE_SOURCE_LABEL + "\n")

    return reference_data


def onehot_build_specs(
    model: MLP,
    onehot_data: list[torch.Tensor],
    output_dir: Path,
    seed: int,
) -> list[DSCBuildSpec]:
    """Describe the four independent class constructions for one seed."""
    return [
        DSCBuildSpec(
            label=f"seed {seed} / class {class_idx} one-hot",
            model=model,
            probe_data=probe_data,
            output_path=output_dir / f"dsc_class_{class_idx}.txt",
            max_dimension=MAX_DIMENSION,
            dt=DT,
            max_iterations=MAX_ITERATIONS,
            fdr_correction=FDR_METHOD,
            significance_level=SIGNIFICANCE_LEVEL,
            relevance_margin=RELEVANCE_MARGIN,
            relevance_scale=RELEVANCE_SCALE,
        )
        for class_idx, probe_data in enumerate(onehot_data)
    ]


def max_filtration(complexes: list[DirectedSimplicialComplex]) -> float:
    values = [
        filtration
        for sc in complexes
        for simplices in sc.complex.values()
        for _, filtration in simplices
    ]
    return max(values) if values else 0.0




def prepare_untrained_baseline(seed: int) -> dict[str, Any]:
    """Prepare one seed's model, probes, and outer-pool work units."""
    print("=" * 80)
    print(f"UNTRAINED {ARCHITECTURE.upper()} BASELINE, SEED {seed}")
    print("=" * 80)
    print(f"  Probe points per class: {N_PROBE_POINTS}")
    print(f"  Max simplex dimension: {MAX_DIMENSION}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  FDR correction: {FDR_METHOD}")
    print(f"  Significance level: {SIGNIFICANCE_LEVEL}")
    print(f"  Step size (dt): {DT}")

    output_dir = output_dir_for(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every cached DSC .txt and probe .pt in output_dir is reused on sight. Pin
    # the configuration they were produced under so a changed constant fails
    # here rather than silently replaying stale complexes under new metadata.
    check_config_fingerprint(
        {
            "n_probe_points": N_PROBE_POINTS,
            "max_dimension": MAX_DIMENSION,
            "max_iterations": MAX_ITERATIONS,
            "dt": DT,
            "fdr_method": FDR_METHOD,
            "significance_level": SIGNIFICANCE_LEVEL,
            "relevance_margin": RELEVANCE_MARGIN,
            "relevance_scale": RELEVANCE_SCALE,
            "probe_tolerance": PROBE_TOLERANCE,
            "probe_seed": PROBE_SEED,
            "probe_seed_rule": "base + 1000*class + n_points",
            "probe_selection": "purity_rejection_sampling",
        },
        output_dir,
    )


    X, y = generate_quadrant_data()
    model = load_or_create_untrained_model(seed)
    n_classes = 4

    print("Evaluating model on all data points...")
    dataloader = DataLoader(TensorDataset(X, y), shuffle=False)
    model.evaluate(dataloader)

    onehot_data = load_reference_onehot_probe_data(output_dir)
    label_data = [
        torch.full((points.shape[0],), class_idx, dtype=torch.long)
        for class_idx, points in enumerate(onehot_data)
    ]

    print("Evaluating model on selected class data points...")
    dataloader = DataLoader(
        TensorDataset(torch.cat(onehot_data), torch.cat(label_data)), shuffle=False
    )
    model.evaluate(dataloader)

    return {
        "seed": seed,
        "output_dir": output_dir,
        "model": model,
        "onehot_data": onehot_data,
        "specs": onehot_build_specs(model, onehot_data, output_dir, seed),
    }


def finish_untrained_baseline(
    state: dict[str, Any],
    complexes: list[DirectedSimplicialComplex],
) -> None:
    """Write the per-seed analysis after its four constructions finish."""
    seed = int(state["seed"])
    output_dir = Path(state["output_dir"])
    model = state["model"]
    n_classes = len(complexes)
    labels = [f"{CLASS_LABELS[class_idx]} one-hot" for class_idx in range(n_classes)]
    stats = [compute_simplex_statistics(sc) for sc in complexes]
    save_dsc_statistics_csv(
        stats,
        str(output_dir / "dsc_statistics_onehot.csv"),
        labels=labels,
    )
    save_dsc_statistics_csv(
        stats,
        str(output_dir / "dsc_statistics_all.csv"),
        labels=labels,
    )
    create_analysis_summary(
        dict(zip(labels, complexes)),
        str(output_dir),
        model=model,
        layer_sizes=model.layer_sizes,
    )

    print(f"Untrained baseline complete: {output_dir}")


def nonvertex_filtration_median(sc: DirectedSimplicialComplex) -> float:
    values = [
        filtration
        for dim, simplices in sc.complex.items()
        if dim > 0
        for _, filtration in simplices
    ]
    return float(np.median(values)) if values else math.nan


def probe_cosine_stats(
    model: MLP,
    probe_data: torch.Tensor,
    class_idx: int,
) -> dict[str, float]:
    target_vec = torch.zeros(model.output_size)
    target_vec[class_idx] = 1.0
    with torch.no_grad():
        outputs = softmax_probabilities(model(probe_data))
        cosines = torch.nn.functional.cosine_similarity(
            outputs, target_vec.unsqueeze(0), dim=1
        )
    return {
        "probe_cosine_max": float(torch.max(cosines).item()),
        "probe_cosine_mean": float(torch.mean(cosines).item()),
        "probe_cosine_median": float(torch.median(cosines).item()),
    }


def row_for_complex(
    *,
    condition: str,
    class_idx: int,
    sc: DirectedSimplicialComplex,
    model: MLP,
    probe_data: torch.Tensor,
    seed: int | None = None,
    trained_complex: DirectedSimplicialComplex | None = None,
) -> dict[str, Any]:
    stats = compute_simplex_statistics(sc)
    counts = stats["counts_by_dimension"]
    row = {
        "architecture": ARCHITECTURE,
        "condition": condition,
        "seed": "" if seed is None else seed,
        "class_idx": class_idx,
        "class_label": CLASS_LABELS[class_idx],
        "total_simplices": int(stats["total_simplices"]),
        "nonvertex_simplices": int(
            sum(count for dim, count in counts.items() if dim > 0)
        ),
        "dim0_simplices": int(counts.get(0, 0)),
        "dim1_simplices": int(counts.get(1, 0)),
        "dim2_simplices": int(counts.get(2, 0)),
        "dim3_simplices": int(counts.get(3, 0)),
        "median_filtration_nonvertex": nonvertex_filtration_median(sc),
        "jaccard_distance_to_trained": math.nan,
    }
    row.update(probe_cosine_stats(model, probe_data, class_idx))
    if condition == "trained":
        row["jaccard_distance_to_trained"] = 0.0
    elif trained_complex is not None:
        row["jaccard_distance_to_trained"] = float(compute_jaccard_distance(sc, trained_complex))
    return row


def collect_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trained_complexes: dict[tuple[str, int], DirectedSimplicialComplex] = {}

    trained_dir = RESULTS_DIR / REFERENCE_RESULT_DIR
    trained_probe_data = load_probe_data(str(trained_dir / "probe_data_onehot.pt"))
    if trained_probe_data is not None:
        trained_model = load_trained_model()
        for class_idx in range(4):
            dsc_path = trained_dir / f"dsc_class_{class_idx}.txt"
            if not dsc_path.exists():
                continue
            sc = load_complex(dsc_path)
            trained_complexes[(ARCHITECTURE, class_idx)] = sc
            rows.append(
                row_for_complex(
                    condition="trained",
                    class_idx=class_idx,
                    sc=sc,
                    model=trained_model,
                    probe_data=trained_probe_data[class_idx],
                )
            )
    else:
        print(f"Trained output missing, skipping trained rows: {trained_dir}")

    for seed in SEEDS:
        output_dir = output_dir_for(seed)
        probe_data = load_probe_data(str(output_dir / "probe_data_onehot.pt"))
        if probe_data is None:
            print(f"Untrained output missing, skipping summary rows: {output_dir}")
            continue
        model = load_or_create_untrained_model(seed)
        for class_idx in range(4):
            dsc_path = output_dir / f"dsc_class_{class_idx}.txt"
            if not dsc_path.exists():
                continue
            sc = load_complex(dsc_path)
            rows.append(
                row_for_complex(
                    condition="untrained",
                    seed=seed,
                    class_idx=class_idx,
                    sc=sc,
                    model=model,
                    probe_data=probe_data[class_idx],
                    trained_complex=trained_complexes.get((ARCHITECTURE, class_idx)),
                )
            )
    return rows


def write_raw_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "architecture",
        "condition",
        "seed",
        "class_idx",
        "class_label",
        "total_simplices",
        "nonvertex_simplices",
        "dim0_simplices",
        "dim1_simplices",
        "dim2_simplices",
        "dim3_simplices",
        "median_filtration_nonvertex",
        "probe_cosine_max",
        "probe_cosine_mean",
        "probe_cosine_median",
        "jaccard_distance_to_trained",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved raw untrained baseline rows to {path}")


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row[key]
        if value == "":
            continue
        value = float(value)
        if not math.isnan(value):
            values.append(value)
    return values


def mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = finite_values(rows, key)
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def aggregate_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], int(row["class_idx"]))].append(row)

    metrics = [
        "total_simplices",
        "nonvertex_simplices",
        "dim1_simplices",
        "dim2_simplices",
        "dim3_simplices",
        "median_filtration_nonvertex",
        "probe_cosine_max",
        "probe_cosine_mean",
        "probe_cosine_median",
        "jaccard_distance_to_trained",
    ]
    summary_rows = []
    for class_idx in range(4):
        group = grouped.get((ARCHITECTURE, class_idx), [])
        trained = [row for row in group if row["condition"] == "trained"]
        untrained = [row for row in group if row["condition"] == "untrained"]
        summary: dict[str, Any] = {
            "architecture": ARCHITECTURE,
            "class_idx": class_idx,
            "class_label": CLASS_LABELS[class_idx],
            "trained_available": bool(trained),
            "n_untrained_seeds": len(untrained),
        }
        trained_row = trained[0] if trained else {}
        for metric in metrics:
            summary[f"trained_{metric}"] = trained_row.get(metric, math.nan)
            mean, std = mean_std(untrained, metric)
            summary[f"untrained_{metric}_mean"] = mean
            summary[f"untrained_{metric}_std"] = std
        summary_rows.append(summary)
    return summary_rows


def write_summary_rows(summary_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved untrained baseline summary to {path}")



def create_untrained_summary() -> None:
    rows = collect_summary_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_raw_rows(rows, OUTPUT_DIR / "raw.csv")
    summary_rows = aggregate_summary_rows(rows)
    write_summary_rows(summary_rows, OUTPUT_DIR / "summary.csv")

def main(run) -> None:
    outer_workers = resolve_outer_workers(N_WORKERS)
    print(f"Outer workers: {outer_workers} of {os.cpu_count()} CPUs")
    print("Inner workers per DSC: 1")

    run.section("prepare models and trained-run probes")
    states = [prepare_untrained_baseline(seed) for seed in SEEDS]
    specs = [spec for state in states for spec in state["specs"]]

    run.section("outer-parallel DSC constructions")
    complexes = build_or_load_many(
        specs,
        configured_workers=outer_workers,
        progress=run.eta,
    )

    run.section("per-seed analysis")
    offset = 0
    for state in states:
        count = len(state["specs"])
        finish_untrained_baseline(state, complexes[offset:offset + count])
        offset += count

    run.section("cross-seed summary")
    create_untrained_summary()


if __name__ == "__main__":
    with RunTimer("quadrants_poly_untrained") as run:
        main(run)
