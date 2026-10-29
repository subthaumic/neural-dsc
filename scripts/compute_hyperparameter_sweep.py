"""
Trajectory step-size sensitivity for DSC construction.

Sweeps the two hyperparameters that sit outside the hypothesis test, over both
handcrafted quadrant architectures and all four classes:

  * Delta t, the step size of the counterfactual ablation trajectory. Purely an
    integrator setting. Reported as simplex counts per dimension, and as whether
    the filtration ordering of the immediate subcomplex is preserved.
  * the probe purity tolerance, the slack below the achievable output cosine
    that a point must clear to be accepted as a probe. This is a
    probe-*selection* setting rather than a test setting, but it is the one that
    decides which region of input space the causal claim is about: tighten it
    and the accepted region shrinks toward the purest points (and the acceptance
    rate, hence the cost of drawing probes, collapses); loosen it and probes
    drift toward the decision boundary.

The parameters that are part of the decision rule -- the relevance margin
epsilon, the significance level alpha, the multiplicity correction, the ablation
resolution K and the probe count B -- are swept in
`compute_inference_sensitivity.py`, which is where their values are chosen; this
script holds them at those values. K belongs there rather than here because it
is checked as an invariance of the decision rule, not as a property of the
integrator: membership must not depend on it at all.

Produces the appendix's parameter-robustness figures.

Cost. At the fixed non-zero relevance margin, each construction is small: the
complex is usually pruned after only a handful of simplices. Parallelising the
candidate tests *inside* one construction therefore spends more time starting
and coordinating workers than doing useful work. The sweep instead dispatches
one complete (architecture, class, parameter) construction per process and
runs each construction serially. A single outer pool is shared by the whole
sweep, so all available lanes stay occupied without repeatedly spawning an
inner pool.

USAGE:
  python compute_hyperparameter_sweep.py

To change run parameters, edit the CONFIGURATION block below. Sweep results are
checkpointed per context and reused on the next run.
"""

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

import atexit
import json
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from neural_dsc.utils.progress import RunTimer
from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.directed_simplicial_complex import (
    DirectedSimplicialComplex,
    pin_worker_threads,
)
from neural_dsc.utils.probe_selection import (
    draw_probe_points,
    estimate_achievable_purity,
    onehot_target,
    probe_seed,
    uniform_box_sampler,
)
from neural_dsc.io.persistence import (
    check_config_fingerprint,
    load_sweep_results,
    save_probe_data,
    load_probe_data,
)
from neural_dsc.utils.dsc_analysis import (
    compute_simplex_statistics,
    create_analysis_summary,
    print_dsc_summary,
    save_dsc_statistics_csv,
    get_filtration_distribution
)
from neural_dsc.utils.jaccard_index import filtration_jaccard_distance

# Path setup
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

MODEL_FACTORIES = {
    "mono_handcrafted": create_quadrants_mono,
    "poly_handcrafted": create_quadrants_poly,
}

# CONFIGURATION
OUTPUT_DIR = RESULTS_DIR / "hyperparameter_sweep"
N_PROBE_POINTS = 128  # Where the complex stops changing; see the producers on why
                      # this is a per-experiment budget rather than a constant.

RELEVANCE_MARGIN = 0.01  # interval null H0: |theta| <= epsilon, epsilon in
                         # units of the target's within-class SD. Anywhere in
                         # 0.001-0.2 gives the same complex; only epsilon = 0
                         # differs, and it fails to converge in probe count.

RELEVANCE_SCALE = "std"
PROBE_TOLERANCE = 0.01  # purity slack below the achievable cosine maximum
PROBE_SEED = 20260819
MAX_DIMENSION = 7  # poly has 7 internal neurons (its ceiling); mono's is 8.
                   # At 4 the poly complexes were truncated in 29/76 sweep rows.
MAX_ITERATIONS = 1000  # resolution of the ablation grid s = 0 .. 1, repo-wide.
                       # K does not enter the correction at all: membership is
                       # decided at full ablation and every grid contains it, so
                       # K only sets how precisely the last crossing can be
                       # located. Measured, K=250 moves a filtration by at most
                       # 0.003, and membership not at all.
# Width of the outer pool. None or -1 = all CPUs but four (floored at 1).
# Individual DSC constructions always use one worker; at the fixed non-zero
# relevance margin they are too small for an inner process pool to pay off.
N_WORKERS = None
DT_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
PROBE_TOLERANCE_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]  # 0.0 accepts only
# points attaining the achievable maximum exactly. Costly by construction -- for
# the polysemantic net that is a single point in class 1 and none in class 3 --
# so it is included as the limiting case and expected to be the slowest row.
DEFAULT_ALPHA = 0.01
DEFAULT_DT = 0.01
# K and dt are independent, so the sweep holds K fixed. The ablation runs until
# the predictors are actually zero, so every row reaches s = 1 whatever dt is:
# dt sets only the integrator's accuracy, K only the resolution of the ablation grid
# the trajectory is reported on, and sweeping dt at fixed K varies exactly one
# thing.
DEFAULT_FDR = "bonferroni"  # over the simplices tested at each dimension; see
                            # compute_quadrants_*.py for why Bonferroni.
ARCHITECTURES = ["mono_handcrafted", "poly_handcrafted"]
CLASSES = [0, 1, 2, 3]


class Tee:
    """Write stdout/stderr to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_log = log_path.exists() and log_path.stat().st_size > 0
    log_file = open(log_path, "a")
    if append_log:
        log_file.write(
            "\n\n"
            + "=" * 80
            + f"\nRestarted at: {datetime.now().isoformat(timespec='seconds')}\n"
            + "=" * 80
            + "\n"
        )
        log_file.flush()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    def cleanup():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(cleanup)
    return log_path


def count_simplices_by_dimension(dsc):
    """
    Count the number of simplices at each dimension.

    Args:
        dsc: DirectedSimplicialComplex instance

    Returns:
        dict: Mapping from dimension to count
    """
    counts = {}
    for dim, simplices in dsc.complex.items():
        counts[dim] = len(simplices)
    return counts


def analyze_filtration_stability(dsc, threshold=0.5):
    """
    Analyze stability of low-filtration simplices.

    Args:
        dsc: DirectedSimplicialComplex instance
        threshold: Filtration threshold for "low-filtration" simplices

    Returns:
        dict: Statistics about low-filtration simplices
    """
    stats = {}

    for dim, simplices in dsc.complex.items():
        if len(simplices) == 0:
            stats[dim] = {"count": 0, "low_filtration_count": 0, "mean_filtration": np.nan}
            continue

        filtration_values = [filt for _, filt in simplices]
        low_filt_count = sum(1 for filt in filtration_values if filt <= threshold)

        stats[dim] = {
            "count": len(simplices),
            "low_filtration_count": low_filt_count,
            "mean_filtration": np.mean(filtration_values),
            "std_filtration": np.std(filtration_values),
            "min_filtration": np.min(filtration_values),
            "max_filtration": np.max(filtration_values),
        }

    return stats


def select_probe_points(model, n_classes, class_idx, n_points, tolerance=None,
                        strict=True):
    """
    Draw `n_points` class-pure probe points for one class.

    Rejection sampling from the input domain, not top-k over the fixed dataset.
    That decouples the sample size of the causal test from the class purity of
    the probe set: under top-k, raising n walked the selection down the cosine
    ranking and into the decision boundary (at k=252 the poly class-3 probe set
    was 69/247 off-class), and n was capped by the dataset. Neither applies here.
    """
    if tolerance is None:
        tolerance = PROBE_TOLERANCE
    return draw_probe_points(
        model,
        onehot_target(n_classes, class_idx),
        n_points,
        tolerance=tolerance,
        seed=probe_seed(PROBE_SEED, class_idx, n_points),
        label=f"class {class_idx}, n={n_points}, tol={tolerance:g}",
        strict=strict,
    )


def probe_selection_diagnostic_rows(architecture, model, outputs, inputs, labels):
    """
    Audit the purity of the probe sets the sweep will actually use.

    Purity is fixed by `PROBE_TOLERANCE` and does not depend on how many probes
    are drawn, so what is worth recording per class is the threshold it works out
    to, how much of the input domain clears it (the acceptance rate, which sets
    the cost of drawing probes), and how class-pure the accepted region is in the
    labels of the reference dataset.
    """
    n_classes = outputs.shape[1]
    rows = []

    for cls in range(n_classes):
        target = onehot_target(n_classes, cls)
        sampler = uniform_box_sampler(generator=torch.Generator().manual_seed(PROBE_SEED + cls))
        achievable = estimate_achievable_purity(model, target, sampler)

        # Where the accepted region falls in the reference dataset, so
        # off-class leakage stays measurable.
        cosine = torch.nn.functional.cosine_similarity(
            outputs, target.to(outputs.dtype).unsqueeze(0), dim=1
        )
        accepted_mask = cosine >= achievable - PROBE_TOLERANCE

        result = draw_probe_points(
            model, target, N_PROBE_POINTS, tolerance=PROBE_TOLERANCE,
            seed=probe_seed(PROBE_SEED, cls, N_PROBE_POINTS),
            label=f"{architecture} class {cls} (diagnostic)",
            strict=False,
        )
        pts = result.points
        rows.append({
            "architecture": architecture,
            "class": cls,
            "n_probe_points": N_PROBE_POINTS,
            "achievable_purity": result.achievable_purity,
            "threshold": result.threshold,
            "tolerance": PROBE_TOLERANCE,
            "min_purity": result.diagnostics["min_purity"],
            "mean_purity": result.diagnostics["mean_purity"],
            "acceptance_rate": result.acceptance_rate,
            "n_attempts": result.n_attempts,
            "converged": result.converged,
            "true_class_count": int((labels == cls).sum()),
            "dataset_accepted_count": int(accepted_mask.sum()),
            "dataset_accepted_true_class_count": int((accepted_mask & (labels == cls)).sum()),
            "dataset_accepted_off_class_count": int((accepted_mask & (labels != cls)).sum()),
            "x_min": float(pts[:, 0].min()) if pts.numel() else float("nan"),
            "x_max": float(pts[:, 0].max()) if pts.numel() else float("nan"),
            "y_min": float(pts[:, 1].min()) if pts.numel() else float("nan"),
            "y_max": float(pts[:, 1].max()) if pts.numel() else float("nan"),
        })

    return rows


def save_probe_selection_diagnostics(rows, report_path):
    """Save probe-selection diagnostics for all architecture/class contexts."""
    diagnostics_path = report_path / "probe_selection_diagnostics.csv"
    pd.DataFrame(rows).to_csv(diagnostics_path, index=False)
    print(f"  Saved probe selection diagnostics to {diagnostics_path}")


def format_param_value(value):
    """Return a filesystem-safe parameter value for output filenames."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def summarize_experiment(dsc):
    """Collect all derived statistics for one constructed DSC."""
    counts = count_simplices_by_dimension(dsc)
    stability_stats = analyze_filtration_stability(dsc, threshold=0.5)
    std_stats = compute_simplex_statistics(dsc)
    filtration_dist = get_filtration_distribution(dsc)

    return {
        "dsc": dsc,
        "counts": counts,
        "stability": stability_stats,
        "std_stats": std_stats,
        "filtration_dist": filtration_dist,
    }


def run_single_experiment(
    model,
    probe_data,
    alpha,
    dt,
    fdr_method,
    max_dimension,
    max_iterations,
    dsc_path=None,
):
    """
    Run a single DSC construction experiment with given hyperparameters.

    Args:
        model: MLP model
        probe_data: Input data for causal testing
        alpha: Significance threshold
        dt: Trajectory step size
        fdr_method: FDR correction method
        max_dimension: Maximum simplex dimension
        max_iterations: Resolution of the ablation grid
        dsc_path: Optional per-experiment DSC checkpoint path

    Returns:
        dict: DSC object plus derived statistics
    """
    print(f"    Running: α={alpha:.4f}, Δt={dt:.4f}, FDR={fdr_method}")

    dsc = DirectedSimplicialComplex()

    if dsc_path is not None and dsc_path.exists():
        print(f"    Loading existing DSC from {dsc_path}")
        dsc.load(str(dsc_path))
    else:
        dsc.build(
            model=model,
            probe_data=probe_data,
            max_dimension=max_dimension,
            dt=dt,
            max_iterations=max_iterations,
            significance_level=alpha,
            # Parallelism belongs to the sweep's outer pool. At non-zero
            # epsilon these complexes are pruned so quickly that an inner pool
            # costs substantially more to start than it saves.
            n_workers=1,
            fdr_correction=fdr_method,
            relevance_margin=RELEVANCE_MARGIN,
            relevance_scale=RELEVANCE_SCALE,
        )

        if dsc_path is not None:
            dsc.save(str(dsc_path))

    return summarize_experiment(dsc)


def run_sweep_work_unit(unit):
    """
    Run one independently schedulable row of either hyperparameter sweep.

    The outer executor owns process-level parallelism; the DSC construction in
    this worker is deliberately serial. Models are cheap deterministic fixtures,
    so rebuilding one in its worker avoids serialising model state for every
    submitted row.

    `unit` is (architecture, class, sweep_name, parameter_value,
    fixed_probe_data, context_dir). `fixed_probe_data` is used by the dt sweep;
    tolerance rows draw the probe set selected by their own parameter value.

    Returns:
        (architecture, class, sweep_name, parameter_value, result)
    """
    pin_worker_threads()
    architecture, class_idx, sweep_name, value, fixed_probe_data, context_dir = unit
    model = MODEL_FACTORIES[architecture]()
    model.eval()

    print(
        f"\n[{architecture}/Q{class_idx + 1}] {sweep_name}={value:g} "
        "(serial DSC construction)"
    )

    if sweep_name == "dt":
        dsc_path = context_dir / f"dsc_dt_{format_param_value(value)}.txt"
        result = run_single_experiment(
            model,
            fixed_probe_data,
            DEFAULT_ALPHA,
            value,
            DEFAULT_FDR,
            MAX_DIMENSION,
            MAX_ITERATIONS,
            dsc_path=dsc_path,
        )
        result["max_iterations"] = MAX_ITERATIONS
    elif sweep_name == "tolerance":
        # strict=False: a tolerance the architecture cannot supply is a row of
        # this sweep's answer, not an error. The caller records it and moves on.
        probe_result = select_probe_points(
            model,
            4,
            class_idx,
            N_PROBE_POINTS,
            tolerance=value,
            strict=False,
        )
        diagnostics = {
            "n_accepted": int(probe_result.points.shape[0]),
            "achievable_purity": probe_result.achievable_purity,
            "threshold": probe_result.threshold,
            "min_purity": probe_result.diagnostics["min_purity"],
            "mean_purity": probe_result.diagnostics["mean_purity"],
            "acceptance_rate": probe_result.acceptance_rate,
            "n_attempts": probe_result.n_attempts,
            "converged": probe_result.converged,
        }
        if not probe_result.converged:
            print(f"    tolerance {value:g}: only "
                  f"{probe_result.points.shape[0]}/{N_PROBE_POINTS} probes accepted in "
                  f"{probe_result.n_attempts} draws -- skipping this row")
            result = {"probe_diagnostics": diagnostics, "skipped": True}
        else:
            dsc_path = context_dir / f"dsc_tolerance_{format_param_value(value)}.txt"
            result = run_single_experiment(
                model,
                probe_result.points,
                DEFAULT_ALPHA,
                DEFAULT_DT,
                DEFAULT_FDR,
                MAX_DIMENSION,
                MAX_ITERATIONS,
                dsc_path=dsc_path,
            )
            result["probe_diagnostics"] = diagnostics
            result["skipped"] = False
    else:
        raise ValueError(f"unknown sweep {sweep_name!r}")

    return architecture, class_idx, sweep_name, value, result


def completed(results):
    """Rows of a sweep that actually produced a complex."""
    return {k: v for k, v in results.items() if not v.get("skipped")}


def ensure_sweep_dsc_files(results, report_path):
    """Backfill per-DSC text checkpoints when results were loaded from a pickle."""
    for value, result in completed(results).items():
        dsc_path = report_path / f"dsc_dt_{format_param_value(value)}.txt"
        if not dsc_path.exists():
            result["dsc"].save(str(dsc_path))





def analyze_probe_tolerance_stability(tolerance_results, report_path, immediate_threshold):
    """
    Compare each tolerance's immediate subcomplex against the tightest one.

    The tightest converged tolerance is the reference because it is the purest
    probe set the architecture can actually supply; the question is how far the
    tolerance can be loosened -- buying acceptance rate, and so probe-drawing
    cost -- before the complex changes.
    """
    done = completed(tolerance_results)
    rows = []
    if not done:
        return rows

    reference_tolerance = min(done)
    reference_filtrations = _simplex_filtrations(done[reference_tolerance])
    reference = {
        simplex
        for simplex, filtration in reference_filtrations.items()
        if filtration <= immediate_threshold + 1e-12
    }

    for tolerance in sorted(done):
        current_filtrations = _simplex_filtrations(done[tolerance])
        current = {
            simplex
            for simplex, filtration in current_filtrations.items()
            if filtration <= immediate_threshold + 1e-12
        }
        union = current | reference
        diagnostics = done[tolerance]["probe_diagnostics"]
        rows.append({
            "probe_tolerance": tolerance,
            "reference_probe_tolerance": reference_tolerance,
            "immediate_threshold": immediate_threshold,
            "immediate_simplices": len(current),
            "reference_immediate_simplices": len(reference),
            "intersection": len(current & reference),
            "union": len(union),
            "immediate_jaccard_distance_vs_reference": (
                1.0 - len(current & reference) / len(union) if union else 0.0
            ),
            "jaccard_distance_vs_reference": filtration_jaccard_distance(
                current_filtrations, reference_filtrations
            ),
            "matches_reference": (current == reference),
            "acceptance_rate": diagnostics["acceptance_rate"],
            "min_purity": diagnostics["min_purity"],
        })

    csv_path = report_path / "probe_tolerance_stability.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved probe tolerance stability checks to {csv_path}")
    return rows


def save_results_table(dt_results, tolerance_results, report_path):
    """
    Save summary tables of the sweep results.

    Args:
        dt_results: Results from dt sweep
        tolerance_results: Results from probe tolerance sweep
        report_path: Path to save tables
    """
    dt_data = []
    dt_dimensions = sorted(set(dim for r in dt_results.values() for dim in r["counts"].keys()))
    for dt in sorted(dt_results.keys()):
        row = {"dt": dt}
        counts = dt_results[dt]["counts"]
        for dim in dt_dimensions:
            row[f"{dim}_simplices"] = counts.get(dim, 0)
        dt_data.append(row)

    dt_df = pd.DataFrame(dt_data)
    dt_df.to_csv(report_path / "dt_sweep_table.csv", index=False)

    tol_data = []
    done = completed(tolerance_results)
    tol_dimensions = sorted(set(dim for r in done.values() for dim in r["counts"].keys()))
    for tolerance in sorted(tolerance_results):
        result = tolerance_results[tolerance]
        diagnostics = result["probe_diagnostics"]
        row = {
            "probe_tolerance": tolerance,
            "converged": diagnostics["converged"],
            "n_accepted": diagnostics["n_accepted"],
            "acceptance_rate": diagnostics["acceptance_rate"],
            "min_purity": diagnostics["min_purity"],
            "threshold": diagnostics["threshold"],
        }
        for dim in tol_dimensions:
            row[f"{dim}_simplices"] = (
                result["counts"].get(dim, 0) if not result.get("skipped") else None
            )
        tol_data.append(row)

    pd.DataFrame(tol_data).to_csv(report_path / "probe_tolerance_sweep_table.csv", index=False)

    print(f"\nResults tables saved to {report_path}")


def _simplex_filtrations(result):
    """Map non-vertex simplices to filtration values for a sweep result."""
    filtrations = {}
    for dim, simplex_list in result["dsc"].complex.items():
        if dim == 0:
            continue
        for simplex, filtration in simplex_list:
            filtrations[tuple(simplex)] = float(filtration)
    return filtrations


def analyze_dt_ordering_stability(dt_results, report_path):
    """Check whether simplex filtration order at the smallest dt is inverted later."""
    baseline_dt = min(dt_results.keys())
    baseline = _simplex_filtrations(dt_results[baseline_dt])
    rows = []

    for dt in sorted(dt_results.keys()):
        current = _simplex_filtrations(dt_results[dt])
        common_simplices = sorted(set(baseline) & set(current))
        comparable_pairs = 0
        tied_after_discretization = 0
        inversions = 0

        for i, simplex_a in enumerate(common_simplices):
            for simplex_b in common_simplices[i + 1:]:
                base_a = baseline[simplex_a]
                base_b = baseline[simplex_b]
                if base_a == base_b:
                    continue

                cur_a = current[simplex_a]
                cur_b = current[simplex_b]
                comparable_pairs += 1

                if cur_a == cur_b:
                    tied_after_discretization += 1
                    continue

                if (base_a < base_b and cur_a > cur_b) or (base_a > base_b and cur_a < cur_b):
                    inversions += 1

        rows.append({
            "dt": dt,
            "baseline_dt": baseline_dt,
            "common_nonvertex_simplices": len(common_simplices),
            "missing_vs_baseline": len(set(baseline) - set(current)),
            "extra_vs_baseline": len(set(current) - set(baseline)),
            "comparable_pairs": comparable_pairs,
            "tied_after_discretization": tied_after_discretization,
            "inversions": inversions,
            "ordering_preserved": inversions == 0,
        })

    csv_path = report_path / "dt_ordering_stability.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved dt ordering stability checks to {csv_path}")
    return rows


def save_claim_checks(dt_results, tolerance_results, report_path, immediate_threshold):
    """Save small derived checks needed by the appendix-writing card."""
    dt_rows = analyze_dt_ordering_stability(dt_results, report_path)
    tol_rows = analyze_probe_tolerance_stability(
        tolerance_results, report_path, immediate_threshold
    )

    summary = {
        "dt_ordering_preserved_all": all(row["ordering_preserved"] for row in dt_rows),
        "dt_total_inversions": int(sum(row["inversions"] for row in dt_rows)),
        "tolerance_reference": (tol_rows[0]["reference_probe_tolerance"] if tol_rows else None),
        # The loosest tolerance whose immediate subcomplex still matches the
        # tightest -- i.e. how much probe-drawing cost can be saved for free.
        "tolerance_loosest_matching_reference": max(
            (row["probe_tolerance"] for row in tol_rows if row["matches_reference"]),
            default=None,
        ),
        "tolerance_rows_skipped": [
            t for t, r in tolerance_results.items() if r.get("skipped")
        ],
    }

    json_path = report_path / "hyperparameter_claim_checks.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved claim check summary to {json_path}")
    return summary, dt_rows, tol_rows


def _add_context(row, architecture, class_idx):
    """Return a row with architecture/class columns inserted first."""
    return {"architecture": architecture, "class": class_idx, **row}


def _append_dt_table_rows(rows, architecture, class_idx, results):
    """Append aggregate dt-sweep table rows for one architecture/class context."""
    dimensions = sorted(set(dim for r in results.values() for dim in r["counts"].keys()))

    for value in sorted(results.keys()):
        row = {
            "architecture": architecture,
            "class": class_idx,
            "dt": value,
        }
        counts = results[value]["counts"]
        for dim in dimensions:
            row[f"{dim}_simplices"] = counts.get(dim, 0)
        rows.append(row)


def _append_tolerance_table_rows(rows, architecture, class_idx, results):
    """Append aggregate tolerance-sweep table rows for one context."""
    done = completed(results)
    dimensions = sorted(set(dim for r in done.values() for dim in r["counts"].keys()))

    for value in sorted(results):
        result = results[value]
        row = {
            "architecture": architecture,
            "class": class_idx,
            "probe_tolerance": value,
            "converged": result["probe_diagnostics"]["converged"],
            "acceptance_rate": result["probe_diagnostics"]["acceptance_rate"],
            "min_purity": result["probe_diagnostics"]["min_purity"],
        }
        for dim in dimensions:
            row[f"{dim}_simplices"] = (
                result["counts"].get(dim, 0) if not result.get("skipped") else None
            )
        rows.append(row)


def analyze_context_sweeps(
    architecture,
    class_idx,
    model,
    output_dir,
    dt_results,
    tolerance_results,
):
    """Analyze completed sweep rows for one architecture/class context."""
    context_dir = output_dir / architecture / f"class_{class_idx}"
    context_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"CONTEXT: architecture={architecture}, class={class_idx}")
    print("#" * 80)
    print(f"Context output directory: {context_dir}")

    print("\n" + "=" * 60)
    print("SWEEP: Trajectory Step Size (Δt)")
    print("=" * 60)
    ensure_sweep_dsc_files(dt_results, context_dir)

    print("\n" + "=" * 60)
    print("SWEEP 2: Probe Purity Tolerance")
    print("=" * 60)
    save_results_table(dt_results, tolerance_results, context_dir)

    print("\n" + "=" * 60)
    print("GENERATING CONTEXT ANALYSIS SUMMARY")
    print("=" * 60)

    all_stats = []
    all_labels = []

    for dt in sorted(dt_results.keys()):
        label = f"{architecture}/Q{class_idx + 1}/dt={dt}"
        all_labels.append(label)
        all_stats.append(dt_results[dt]["std_stats"])
        print_dsc_summary(dt_results[dt]["dsc"], label, model)

    for tolerance in sorted(completed(tolerance_results)):
        label = f"{architecture}/Q{class_idx + 1}/tolerance={tolerance}"
        all_labels.append(label)
        all_stats.append(tolerance_results[tolerance]["std_stats"])
        print_dsc_summary(tolerance_results[tolerance]["dsc"], label, model)

    save_dsc_statistics_csv(all_stats, str(context_dir / "dsc_statistics_all.csv"),
                            labels=all_labels)

    dt_stats = [dt_results[d]["std_stats"] for d in sorted(dt_results.keys())]
    dt_labels = [f"{architecture}/Q{class_idx + 1}/dt={d}" for d in sorted(dt_results.keys())]
    save_dsc_statistics_csv(dt_stats, str(context_dir / "dsc_statistics_dt_sweep.csv"),
                            labels=dt_labels)

    tol_done = sorted(completed(tolerance_results))
    if tol_done:
        save_dsc_statistics_csv(
            [tolerance_results[t]["std_stats"] for t in tol_done],
            str(context_dir / "dsc_statistics_probe_tolerance_sweep.csv"),
            labels=[f"{architecture}/Q{class_idx + 1}/tolerance={t}" for t in tol_done],
        )

    all_dscs = {}
    for dt in sorted(dt_results.keys()):
        all_dscs[f"{architecture}/Q{class_idx + 1}/dt={dt}"] = dt_results[dt]["dsc"]
    for tolerance in tol_done:
        all_dscs[f"{architecture}/Q{class_idx + 1}/tolerance={tolerance}"] = (
            tolerance_results[tolerance]["dsc"]
        )
    create_analysis_summary(all_dscs, str(context_dir), model=model)

    claim_summary, dt_rows, tol_rows = save_claim_checks(
        dt_results, tolerance_results, context_dir, DEFAULT_DT
    )

    aggregate_stats = []
    aggregate_labels = []
    for dt in sorted(dt_results.keys()):
        aggregate_stats.append(dt_results[dt]["std_stats"])
        aggregate_labels.append(f"{architecture}/Q{class_idx + 1}/dt={dt}")
    for tolerance in tol_done:
        aggregate_stats.append(tolerance_results[tolerance]["std_stats"])
        aggregate_labels.append(f"{architecture}/Q{class_idx + 1}/tolerance={tolerance}")

    return {
        "architecture": architecture,
        "class": class_idx,
        "context_dir": str(context_dir.relative_to(output_dir)),
        "dt_results": dt_results,
        "tolerance_results": tolerance_results,
        "stats": aggregate_stats,
        "labels": aggregate_labels,
        "claim_summary": _add_context(claim_summary, architecture, class_idx),
        "dt_rows": [_add_context(row, architecture, class_idx) for row in dt_rows],
        "tolerance_rows": [_add_context(row, architecture, class_idx) for row in tol_rows],
    }


def resolve_outer_workers(configured_workers):
    """Resolve the outer-pool width, leaving four CPUs free in auto mode."""
    if configured_workers is None or configured_workers == -1:
        return max(1, mp.cpu_count() - 4)
    if configured_workers < 1:
        raise ValueError(f"N_WORKERS must be positive, None, or -1; got {configured_workers}")
    return configured_workers


def save_sweep_checkpoint(results, path):
    """Atomically checkpoint the completed rows of one context/sweep."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    tmp_path.replace(path)
    print(f"  Checkpointed {len(results)} sweep rows to {path}")


def record_sweep_result(context_state, work_result):
    """Insert one worker result and checkpoint its context immediately."""
    architecture, class_idx, sweep_name, value, result = work_result
    state = context_state[(architecture, class_idx)]
    if sweep_name == "dt":
        state["dt_results"][value] = result
        save_sweep_checkpoint(state["dt_results"], state["dt_results_file"])
    elif sweep_name == "tolerance":
        state["tolerance_results"][value] = result
        save_sweep_checkpoint(
            state["tolerance_results"], state["tolerance_results_file"]
        )
    else:
        raise ValueError(f"unknown sweep {sweep_name!r}")


def run_outer_sweep(work_units, context_state, n_workers, run):
    """Dispatch complete DSC constructions through one shared outer pool."""
    if not work_units:
        print("All sweep rows were found in checkpoints; no constructions pending.")
        return

    lanes = min(n_workers, len(work_units))
    print(
        f"Dispatching {len(work_units)} pending constructions over {lanes} outer "
        "workers; each DSC build is serial."
    )

    if lanes == 1:
        for i, unit in enumerate(work_units, 1):
            record_sweep_result(context_state, run_sweep_work_unit(unit))
            run.eta(i, len(work_units), "constructions")
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=lanes,
        mp_context=ctx,
        initializer=pin_worker_threads,
    ) as pool:
        futures = [pool.submit(run_sweep_work_unit, unit) for unit in work_units]
        for i, future in enumerate(as_completed(futures), 1):
            record_sweep_result(context_state, future.result())
            run.eta(i, len(work_units), "constructions")


def main(run):
    """Main experimental pipeline."""
    architectures = list(dict.fromkeys(ARCHITECTURES))
    classes = list(dict.fromkeys(CLASSES))
    invalid_architectures = [
        architecture for architecture in architectures if architecture not in MODEL_FACTORIES
    ]
    if invalid_architectures:
        raise ValueError(f"Unknown architectures in CONFIGURATION: {invalid_architectures}")
    invalid_classes = [class_idx for class_idx in classes if class_idx < 0 or class_idx > 3]
    if invalid_classes:
        raise ValueError(f"Class indices in CONFIGURATION must be in 0..3, got {invalid_classes}")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every cached .pkl and .txt in output_dir is reused on sight. Pin the
    # configuration they were produced under so a changed constant fails here
    # rather than silently replaying stale complexes under new metadata.
    check_config_fingerprint(
        {
            "n_probe_points": N_PROBE_POINTS,
            "max_dimension": MAX_DIMENSION,
            "max_iterations": MAX_ITERATIONS,
            "dt_values": DT_VALUES,
            "probe_tolerance_values": PROBE_TOLERANCE_VALUES,
            "default_alpha": DEFAULT_ALPHA,
            "default_dt": DEFAULT_DT,
            "default_fdr": DEFAULT_FDR,
            "relevance_margin": RELEVANCE_MARGIN,
            "relevance_scale": RELEVANCE_SCALE,
            "probe_tolerance": PROBE_TOLERANCE,
            "probe_seed": PROBE_SEED,
            "probe_seed_rule": "base + 1000*class + n_points",
            "architectures": architectures,
            "classes": classes,
            "probe_selection": "purity_rejection_sampling",
        },
        output_dir,
    )

    setup_logging(output_dir / "hyperparameter_sweep.log")
    run_start_time = time.time()
    n_workers = resolve_outer_workers(N_WORKERS)

    print(f"Output directory: {output_dir}")
    print(f"Log file: {output_dir / 'hyperparameter_sweep.log'}")
    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
    print("\nConfiguration:")
    print(f"  output_dir: {output_dir}")
    print(f"  n_probe_points: {N_PROBE_POINTS}")
    print(f"  relevance_margin: {RELEVANCE_MARGIN} ({RELEVANCE_SCALE})")
    print(f"  probe_tolerance: {PROBE_TOLERANCE}")
    print(f"  max_dimension: {MAX_DIMENSION}")
    print(f"  max_iterations: {MAX_ITERATIONS}")
    print(f"  outer_workers: {n_workers} of {mp.cpu_count()} CPUs")
    print("  inner_workers_per_construction: 1")
    print(f"  dt_values: {DT_VALUES}")
    print(f"  probe_tolerance_values: {PROBE_TOLERANCE_VALUES}")
    print(f"  default_alpha: {DEFAULT_ALPHA}")
    print(f"  default_dt: {DEFAULT_DT}")
    print(f"  default_fdr: {DEFAULT_FDR}")
    print(f"  architectures: {architectures}")
    print(f"  classes: {classes}\n")

    print("Setting up shared quadrant data...")
    X_train, y_train = generate_quadrant_data(n_total=1000, seed=42)
    models = {}
    probe_diagnostic_rows = []

    for architecture in architectures:
        print(f"Preparing {architecture} model...")
        model = MODEL_FACTORIES[architecture]()
        model.eval()
        with torch.no_grad():
            outputs = model(X_train)
            predictions = torch.argmax(outputs, dim=1)
        accuracy = float((predictions == y_train).float().mean())
        print(f"  Accuracy against quadrant labels: {accuracy:.2%}")

        models[architecture] = model
        probe_diagnostic_rows.extend(
            probe_selection_diagnostic_rows(
                architecture, model, outputs, X_train, y_train
            )
        )

    save_probe_selection_diagnostics(probe_diagnostic_rows, output_dir)

    # Load partial or complete per-context checkpoints, then make one work unit
    # for every missing row. The fixed probe set is selected once per context
    # and passed only to its dt rows; tolerance rows select their own probes.
    context_state = {}
    dt_work_units = []
    tolerance_work_units = []
    for architecture in architectures:
        model = models[architecture]
        for class_idx in classes:
            context_dir = output_dir / architecture / f"class_{class_idx}"
            context_dir.mkdir(parents=True, exist_ok=True)
            dt_results_file = context_dir / "dt_sweep_results.pkl"
            tolerance_results_file = context_dir / "probe_tolerance_sweep_results.pkl"
            dt_results = load_sweep_results(str(dt_results_file)) or {}
            tolerance_results = load_sweep_results(str(tolerance_results_file)) or {}
            context_state[(architecture, class_idx)] = {
                "dt_results": dt_results,
                "tolerance_results": tolerance_results,
                "dt_results_file": dt_results_file,
                "tolerance_results_file": tolerance_results_file,
            }

            missing_dt = [dt for dt in DT_VALUES if dt not in dt_results]
            fixed_probe_data = None
            if missing_dt:
                probe_data_file = context_dir / f"probe_data_n{N_PROBE_POINTS}.pt"
                saved_probe_data = load_probe_data(str(probe_data_file))
                if saved_probe_data is None:
                    print(
                        f"Selecting fixed probe data for "
                        f"{architecture}/Q{class_idx + 1}..."
                    )
                    fixed_probe_data = select_probe_points(
                        model, 4, class_idx, N_PROBE_POINTS
                    ).points
                    save_probe_data([fixed_probe_data], str(probe_data_file))
                else:
                    fixed_probe_data = saved_probe_data[0]

            for dt in sorted(missing_dt):
                dt_work_units.append(
                    (
                        architecture,
                        class_idx,
                        "dt",
                        dt,
                        fixed_probe_data,
                        context_dir,
                    )
                )
            for tolerance in sorted(
                value
                for value in PROBE_TOLERANCE_VALUES
                if value not in tolerance_results
            ):
                tolerance_work_units.append(
                    (
                        architecture,
                        class_idx,
                        "tolerance",
                        tolerance,
                        None,
                        context_dir,
                    )
                )

    # Tight probe tolerances and small integration steps are the longer rows, so
    # submitting them first keeps the tail from being stranded after other lanes
    # have gone idle.
    tolerance_work_units.sort(key=lambda unit: unit[3])
    dt_work_units.sort(key=lambda unit: unit[3])
    work_units = tolerance_work_units + dt_work_units
    run.section("outer-parallel DSC constructions")
    run_outer_sweep(work_units, context_state, n_workers, run)

    context_outputs = []
    run.section("context analysis + figures")
    n_contexts = len(architectures) * len(classes)
    for architecture in architectures:
        for class_idx in classes:
            state = context_state[(architecture, class_idx)]
            missing_dt = [dt for dt in DT_VALUES if dt not in state["dt_results"]]
            missing_tolerance = [
                tolerance
                for tolerance in PROBE_TOLERANCE_VALUES
                if tolerance not in state["tolerance_results"]
            ]
            if missing_dt or missing_tolerance:
                raise RuntimeError(
                    f"incomplete sweep for {architecture}/Q{class_idx + 1}: "
                    f"missing dt={missing_dt}, tolerance={missing_tolerance}"
                )
            run.eta(len(context_outputs), n_contexts, "contexts")
            context_outputs.append(
                analyze_context_sweeps(
                    architecture,
                    class_idx,
                    models[architecture],
                    output_dir,
                    state["dt_results"],
                    state["tolerance_results"],
                )
            )
    run.section("aggregate analysis + figures")

    print("\n" + "=" * 60)
    print("GENERATING AGGREGATE ANALYSIS OUTPUTS")
    print("=" * 60)

    aggregate_stats = []
    aggregate_labels = []
    aggregate_dt_stats = []
    aggregate_dt_labels = []
    aggregate_dt_table_rows = []
    aggregate_dt_stability_rows = []
    aggregate_tolerance_table_rows = []
    aggregate_tolerance_stability_rows = []
    aggregate_claim_summaries = []

    for context in context_outputs:
        architecture = context["architecture"]
        class_idx = context["class"]
        aggregate_stats.extend(context["stats"])
        aggregate_labels.extend(context["labels"])
        aggregate_claim_summaries.append(context["claim_summary"])
        aggregate_dt_stability_rows.extend(context["dt_rows"])
        aggregate_tolerance_stability_rows.extend(context["tolerance_rows"])

        _append_dt_table_rows(
            aggregate_dt_table_rows, architecture, class_idx, context["dt_results"]
        )
        _append_tolerance_table_rows(
            aggregate_tolerance_table_rows, architecture, class_idx,
            context["tolerance_results"],
        )

        for dt in sorted(context["dt_results"].keys()):
            aggregate_dt_stats.append(context["dt_results"][dt]["std_stats"])
            aggregate_dt_labels.append(f"{architecture}/Q{class_idx + 1}/dt={dt}")

    save_dsc_statistics_csv(aggregate_stats, str(output_dir / "dsc_statistics_all.csv"),
                            labels=aggregate_labels)
    save_dsc_statistics_csv(
        aggregate_dt_stats,
        str(output_dir / "dsc_statistics_dt_sweep.csv"),
        labels=aggregate_dt_labels,
    )

    dt_table_df = pd.DataFrame(aggregate_dt_table_rows)
    dt_table_df.to_csv(output_dir / "dt_sweep_table_all_contexts.csv", index=False)
    dt_table_df.to_csv(output_dir / "dt_sweep_table.csv", index=False)
    tol_table_df = pd.DataFrame(aggregate_tolerance_table_rows)
    tol_table_df.to_csv(output_dir / "probe_tolerance_sweep_table_all_contexts.csv", index=False)
    tol_table_df.to_csv(output_dir / "probe_tolerance_sweep_table.csv", index=False)
    pd.DataFrame(aggregate_dt_stability_rows).to_csv(
        output_dir / "dt_ordering_stability_all_contexts.csv", index=False
    )
    pd.DataFrame(aggregate_tolerance_stability_rows).to_csv(
        output_dir / "probe_tolerance_stability_all_contexts.csv", index=False
    )
    pd.DataFrame(aggregate_claim_summaries).to_csv(
        output_dir / "hyperparameter_claim_checks_all_contexts.csv", index=False
    )

    aggregate_claim_checks = {
        "contexts": aggregate_claim_summaries,
        "dt_ordering_preserved_all_contexts": all(
            row["dt_ordering_preserved_all"] for row in aggregate_claim_summaries
        ),
        "dt_total_inversions_all_contexts": int(
            sum(row["dt_total_inversions"] for row in aggregate_claim_summaries)
        ),
        "tolerance_loosest_matching_reference_by_context": {
            f"{row['architecture']}/class_{row['class']}":
                row["tolerance_loosest_matching_reference"]
            for row in aggregate_claim_summaries
        },
        "tolerance_rows_skipped_by_context": {
            f"{row['architecture']}/class_{row['class']}": row["tolerance_rows_skipped"]
            for row in aggregate_claim_summaries
        },
    }
    with open(output_dir / "hyperparameter_claim_checks.json", "w") as f:
        json.dump(aggregate_claim_checks, f, indent=2)

    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"\nAnalysis outputs:")
    print("  - {architecture}/class_{idx}/dt_sweep.pdf/png")
    print("  - {architecture}/class_{idx}/dt_stability.pdf/png")
    print("  - {architecture}/class_{idx}/probe_tolerance_sweep.pdf/png")
    print("  - {architecture}/class_{idx}/dsc_*.txt checkpoints")
    print("  - {architecture}/class_{idx}/dsc_statistics_all.csv and dsc_analysis_summary.txt")
    print("  - dsc_statistics_all.csv (aggregate, all contexts)")
    print("  - dsc_statistics_dt_sweep.csv and dsc_statistics_probe_tolerance_sweep.csv")
    print("  - dt_sweep_table_all_contexts.csv")
    print("  - probe_tolerance_sweep_table_all_contexts.csv")
    print("  - dt_ordering_stability_all_contexts.csv")
    print("  - probe_tolerance_stability_all_contexts.csv")
    print("  - hyperparameter_claim_checks.json and hyperparameter_claim_checks_all_contexts.csv")
    print("  - probe_selection_diagnostics.csv")

    total_elapsed = time.time() - run_start_time
    print(f"\nFinished at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Total runtime: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")

    metadata = {
        "output_dir": os.path.relpath(output_dir, REPO_ROOT),
        "started_at": datetime.fromtimestamp(run_start_time).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": total_elapsed,
        "runtime_minutes": total_elapsed / 60,
        "n_probe_points": N_PROBE_POINTS,
        "relevance_margin": RELEVANCE_MARGIN,
        "relevance_scale": RELEVANCE_SCALE,
        "probe_tolerance": PROBE_TOLERANCE,
        "probe_seed": PROBE_SEED,
        "probe_seed_rule": "base + 1000*class + n_points",
        "max_dimension": MAX_DIMENSION,
        "max_iterations": MAX_ITERATIONS,
        "n_workers": n_workers,
        "outer_workers": n_workers,
        "inner_workers_per_construction": 1,
        "dt_values": DT_VALUES,
        "probe_tolerance_values": PROBE_TOLERANCE_VALUES,
        "default_alpha": DEFAULT_ALPHA,
        "default_dt": DEFAULT_DT,
        "default_fdr": DEFAULT_FDR,
        "architectures": architectures,
        "classes": classes,
        "context_count": len(context_outputs),
        "contexts": [
            {
                "architecture": context["architecture"],
                "class": context["class"],
                "context_dir": context["context_dir"],
            }
            for context in context_outputs
        ],
        "output_layout": "per-context outputs live under {architecture}/class_{idx}/; root CSV/JSON files aggregate all contexts",
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  - run_metadata.json")


if __name__ == "__main__":
    with RunTimer("hyperparameter sweep") as run:
        main(run)
