"""
Handcrafted Polysemantic Quadrants Classifier

This script generates all data and plots for the handcrafted polysemantic
quadrant classifier experiments. All results are saved in a canonical report
folder.
Directed Simplicial Complexes (DSCs) and other outputs are only generated if they do not already exist, making the script fully resumable.
The neural network is only trained and saved if a model file for the current version does not already exist.

GENERAL STRUCTURE:
1. Setup: Generate quadrant data and model.
2. One-hot Points: Select and plot points classified as one-hot for each class, build and plot DSCs.
3. Two-hot Points: Select and plot points with two strong activations (two-hot), build and plot DSCs.
4. Random Points: Select random points, build and plot DSC.
5. Jaccard distance: Compute and plot Jaccard distances between all DSCs (one-hot, two-hot, random), including by-dimension analysis.
6++. Visualizations: Generate summary plots showing all selected points, random points, and two-hot points.

USAGE:
  python compute_quadrants_poly_handcrafted.py

To change run parameters, edit the CONFIGURATION block below.

The four one-hot, six two-hot, and random-control constructions run through one
outer process pool. Each individual DSC build is serial.
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

from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.models.gen import create_quadrants_poly
from neural_dsc.utils.jaccard_index import compute_jaccard_distance
from neural_dsc.utils.select import get_twohot_indices
from neural_dsc.utils.probe_selection import (
    draw_probe_points,
    onehot_target,
    probe_seed,
    twohot_target,
)
from neural_dsc.viz.data import (
    get_twohot_colors,
    get_onehot_markers,
    get_twohot_markers,
    get_random_marker,
)
from neural_dsc.utils.progress import RunTimer
from neural_dsc.utils.parallel_dsc import (
    DSCBuildSpec,
    build_or_load_many,
    resolve_outer_workers,
)
from neural_dsc.io.persistence import (
    check_config_fingerprint,
    save_probe_data,
    load_probe_data,
    save_jaccard_data,
    load_jaccard_data
)
from neural_dsc.utils.dsc_analysis import (
    compute_simplex_statistics,
    print_dsc_summary,
    save_dsc_statistics_csv,
    create_analysis_summary
)

# Path handling: get repository root
# The article uses the four adjacent two-hot pairs. The diagonal pairs Q1+Q3
# and Q2+Q4 meet only at the origin and are left out. Each remaining pair keeps
# its position among all six pairs as its seed index, so the adjacent input
# sets are exactly those of the runs behind the article.
DIAGONAL_TWOHOT_PAIRS = frozenset({(0, 2), (1, 3)})

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"

# DSC CONFIGURATION
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
SIGNIFICANCE_LEVEL = 0.01  # was inherited silently from the library default


# -----------------------------
# Main script entry point
# -----------------------------
def main(run):
    output_dir = RESULTS_DIR / "quadrants_poly_handcrafted"
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_workers = resolve_outer_workers(N_WORKERS)

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


    print("=" * 80)
    print("HANDCRAFTED POLYSEMANTIC QUADRANT CLASSIFIER EXPERIMENT")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Probe points per class: {N_PROBE_POINTS}")
    print(f"  Max simplex dimension: {MAX_DIMENSION}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Outer workers: {outer_workers} of {os.cpu_count()} CPUs")
    print("  Inner workers per DSC: 1")
    print(f"  FDR correction: {FDR_METHOD}")
    print(f"  Step size (dt): {DT}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Setup
    run.section("setup (data + model)")
    # -------------------------------------------------------------------------
    n_classes = 4
    num_points = N_PROBE_POINTS

    # 1.2 Data Generation: create quadrant data and evaluate model
    X, y = generate_quadrant_data()

    # 1.3 Neural Network: create or load the model
    # Note: quadrant models are deterministic (handcrafted weights), so we use a fixed filename
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "quadrants_poly_handcrafted.pth"

    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        net = create_quadrants_poly()  # Create architecture
        net.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print("Creating new polysemantic quadrant classifier...")
        net = create_quadrants_poly()
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("Evaluating model on all data points...")
    dataloader = DataLoader(TensorDataset(X, y), shuffle=False)
    net.evaluate(dataloader)

    # -------------------------------------------------------------------------
    # 2. Prepare approximately one-hot encoded probe points
    run.section("one-hot probe selection")
    # -------------------------------------------------------------------------
    # 2.1 select and plot points for each class
    probe_data_file = output_dir / "probe_data_onehot.pt"
    onehot_data = load_probe_data(str(probe_data_file))

    if onehot_data is None:
        print("Finding one-hot encoded points...")
        onehot_data = []
        label_data = []
        for i in range(n_classes):
            # Purity-based rejection sampling from the input domain rather than
            # top-k over the fixed dataset: `num_points` no longer trades off
            # against class purity, so raising it cannot walk the probe set into
            # the decision boundary. See neural_dsc.utils.probe_selection.
            selected_points = draw_probe_points(
                net,
                onehot_target(n_classes, i),
                num_points,
                tolerance=PROBE_TOLERANCE,
                seed=probe_seed(PROBE_SEED, i, num_points),
                label=f"Q{i + 1}",
            ).points
            onehot_data.append(selected_points)
            label_data.append(torch.full((selected_points.shape[0],), i, dtype=torch.long))
        save_probe_data(onehot_data, str(probe_data_file))
    else:
        print(f"Loaded existing one-hot probe data from {probe_data_file}")
        label_data = [torch.full((points.shape[0],), i, dtype=torch.long) for i, points in enumerate(onehot_data)]

    print("Evaluating model on selected class data points...")
    dataloader = DataLoader(
        TensorDataset(torch.cat(onehot_data), torch.cat(label_data)), shuffle=False
    )
    net.evaluate(dataloader)

    # -------------------------------------------------------------------------
    # 3. Prepare approximately two-hot encoded probe points
    run.section("two-hot probe selection")
    # -------------------------------------------------------------------------
    # 3.1 select and plot two-hot encoded points
    all_twohot_pairs = get_twohot_indices(n_classes)
    twohot_seed_index = {pair: idx for idx, pair in enumerate(all_twohot_pairs)}
    twohot_indices = [
        pair for pair in all_twohot_pairs if pair not in DIAGONAL_TWOHOT_PAIRS
    ]
    labels_twohot = [f"Q{i + 1}+Q{j + 1}" for i, j in twohot_indices]
    filelabels_twohot = [f"{i}{j}" for i, j in twohot_indices]

    probe_data_twohot_file = output_dir / "probe_data_twohot.pt"
    twohot_data = load_probe_data(str(probe_data_twohot_file))

    if twohot_data is None:
        print("Finding two-hot encoded points...")
        twohot_data = []
        for idx, (i, j) in enumerate(twohot_indices):
            selected_points = draw_probe_points(
                net,
                twohot_target(n_classes, i, j),
                num_points,
                tolerance=PROBE_TOLERANCE,
                seed=probe_seed(PROBE_SEED + 100, twohot_seed_index[(i, j)], num_points),
                label=f"Two-hot {i},{j}",
            ).points
            twohot_data.append(selected_points)
        save_probe_data(twohot_data, str(probe_data_twohot_file))
    else:
        print(f"Loaded existing two-hot probe data from {probe_data_twohot_file}")

    print("Evaluating model on selected class data points...")
    dataloader = DataLoader(
        TensorDataset(torch.cat(onehot_data), torch.cat(label_data)), shuffle=False
    )
    net.evaluate(dataloader)

    # -------------------------------------------------------------------------
    # 4. Prepare random control points
    run.section("random-probe selection (control)")
    # -------------------------------------------------------------------------
    # 4.1 Select random points
    probe_data_random_file = output_dir / "probe_data_random.pt"
    random_points_loaded = load_probe_data(str(probe_data_random_file))

    if random_points_loaded is None:
        print("Selecting random points...")
        np.random.seed(42)
        rand_indices = np.random.choice(len(X), num_points, replace=False)
        random_points = X[rand_indices]
        save_probe_data([random_points], str(probe_data_random_file))
    else:
        print(f"Loaded existing random probe data from {probe_data_random_file}")
        random_points = random_points_loaded[0]

    # -------------------------------------------------------------------------
    # 4. Build every independent complex in one persistent outer pool
    run.section("outer-parallel DSC constructions")
    # -------------------------------------------------------------------------
    def spec(label, probe_data, output_path):
        return DSCBuildSpec(
            label=label,
            model=net,
            probe_data=probe_data,
            output_path=output_path,
            max_dimension=MAX_DIMENSION,
            dt=DT,
            max_iterations=MAX_ITERATIONS,
            fdr_correction=FDR_METHOD,
            significance_level=SIGNIFICANCE_LEVEL,
            relevance_margin=RELEVANCE_MARGIN,
            relevance_scale=RELEVANCE_SCALE,
        )

    build_specs = [
        spec(
            f"class {class_idx} one-hot",
            points,
            output_dir / f"dsc_class_{class_idx}.txt",
        )
        for class_idx, points in enumerate(onehot_data)
    ]
    build_specs.extend(
        spec(
            f"two-hot {labels_twohot[idx]}",
            points,
            output_dir / f"dsc_twohot_{filelabels_twohot[idx]}.txt",
        )
        for idx, points in enumerate(twohot_data)
    )
    build_specs.append(
        spec("random control", random_points, output_dir / "dsc_random.txt")
    )

    complexes = build_or_load_many(
        build_specs,
        configured_workers=outer_workers,
        progress=run.eta,
    )
    n_onehot = len(onehot_data)
    n_twohot = len(twohot_data)
    onehot_complexes = complexes[:n_onehot]
    two_hot_complexes = complexes[n_onehot:n_onehot + n_twohot]
    sc_random = complexes[-1]

    # -------------------------------------------------------------------------
    # 5. Analyze the completed complexes
    run.section("DSC statistics")
    # -------------------------------------------------------------------------
    print("\nAnalyzing one-hot DSCs...")
    onehot_labels = [f"Q{class_idx + 1} one-hot" for class_idx in range(n_onehot)]
    onehot_stats = [compute_simplex_statistics(sc) for sc in onehot_complexes]
    for label, sc in zip(onehot_labels, onehot_complexes):
        print_dsc_summary(sc, label, net)
    save_dsc_statistics_csv(
        onehot_stats,
        str(output_dir / "dsc_statistics_onehot.csv"),
        labels=onehot_labels,
    )

    print("\nAnalyzing two-hot DSCs...")
    twohot_stats = [compute_simplex_statistics(sc) for sc in two_hot_complexes]
    for label, sc in zip(labels_twohot, two_hot_complexes):
        print_dsc_summary(sc, label, net)
    save_dsc_statistics_csv(
        twohot_stats,
        str(output_dir / "dsc_statistics_twohot.csv"),
        labels=labels_twohot,
    )

    print("\nAnalyzing random DSC...")
    random_stats = compute_simplex_statistics(sc_random)
    print_dsc_summary(sc_random, "Random Sample", net)
    save_dsc_statistics_csv(
        [random_stats],
        str(output_dir / "dsc_statistics_random.csv"),
        labels=["Random"],
    )

    # -------------------------------------------------------------------------
    # 6. Jaccard distance
    run.section("Jaccard distances")
    # -------------------------------------------------------------------------
    # 5.0 Dynamically determine all nonzero dimensions present in any complex
    all_dims = set()
    for sc in onehot_complexes + two_hot_complexes:
        if sc is not None:
            all_dims.update(sc.complex.keys())
    if sc_random is not None:
        all_dims.update(sc_random.complex.keys())
    all_dims = sorted([d for d in all_dims if d != 0])

    # 5.1 Jaccard distances: two-hot vs one-hot
    jaccard_twohot_file = output_dir / "jaccard_twohot_vs_onehot.json"
    jaccard_matrix, _ = load_jaccard_data(str(jaccard_twohot_file))

    if jaccard_matrix is None:
        print("Computing Jaccard distances for two-hot vs one-hot...")
        n_onehot = len(onehot_complexes)
        n_twohot = len(two_hot_complexes)
        jaccard_matrix = np.zeros((n_twohot, n_onehot))
        for i in range(n_twohot):
            for j in range(n_onehot):
                jaccard_matrix[i, j] = compute_jaccard_distance(
                    two_hot_complexes[i], onehot_complexes[j]
                )
        save_jaccard_data(jaccard_matrix, {}, str(jaccard_twohot_file))
    else:
        print(f"Loaded existing Jaccard data from {jaccard_twohot_file}")
        n_onehot = len(onehot_complexes)
        n_twohot = len(two_hot_complexes)

    # 5.2 Jaccard distances: random vs one-hot
    jaccard_random_file = output_dir / "jaccard_random_vs_onehot.json"
    jaccard_matrix_random_onehot, _ = load_jaccard_data(str(jaccard_random_file))

    if jaccard_matrix_random_onehot is None:
        print("Computing Jaccard distances for random vs one-hot/two-hot...")
        jaccard_matrix_random_onehot = np.zeros((1, n_onehot))
        for j in range(n_onehot):
            jaccard_matrix_random_onehot[0, j] = compute_jaccard_distance(
                sc_random, onehot_complexes[j]
            )
        save_jaccard_data(jaccard_matrix_random_onehot, {}, str(jaccard_random_file))
    else:
        print(f"Loaded existing Jaccard data from {jaccard_random_file}")

    for dim in all_dims:
        jaccard_random_dim_file = output_dir / f"jaccard_random_vs_onehot_dim{dim}.json"
        jaccard_matrix_random_onehot_dim, _ = load_jaccard_data(str(jaccard_random_dim_file))

        if jaccard_matrix_random_onehot_dim is None:
            jaccard_matrix_random_onehot_dim = np.zeros((1, n_onehot))
            for j in range(n_onehot):
                jaccard_dict = compute_jaccard_distance(
                    sc_random, onehot_complexes[j], by_dimension=True
                )
                val = jaccard_dict.get(dim, 0) # type: ignore
                jaccard_matrix_random_onehot_dim[0, j] = val
            save_jaccard_data(jaccard_matrix_random_onehot_dim, {}, str(jaccard_random_dim_file))
    # 5.3. Full Jaccard distance matrix (all DSCs)
    all_complexes_block = onehot_complexes + two_hot_complexes + [sc_random]
    labels_onehot = [f"Q{k + 1}" for k in range(len(onehot_complexes))]
    all_labels_block = labels_onehot + labels_twohot + ["Random"]
    n_block = len(all_complexes_block)

    jaccard_full_file = output_dir / "jaccard_matrix_full.json"
    jaccard_matrix_block, _ = load_jaccard_data(str(jaccard_full_file))

    if jaccard_matrix_block is None:
        print("Computing full symmetric Jaccard matrix...")
        jaccard_matrix_block = np.zeros((n_block, n_block))
        for i in range(n_block):
            for j in range(i, n_block):
                if (
                    all_complexes_block[i] is not None
                    and all_complexes_block[j] is not None
                ):
                    val = compute_jaccard_distance(
                        all_complexes_block[i], all_complexes_block[j]
                    )
                else:
                    val = np.nan
                jaccard_matrix_block[i, j] = val
                jaccard_matrix_block[j, i] = val
        save_jaccard_data(jaccard_matrix_block, {}, str(jaccard_full_file))
    else:
        print(f"Loaded existing full Jaccard matrix from {jaccard_full_file}")

    # Save full Jaccard matrix to CSV
    import csv

    jaccard_csv = output_dir / "jaccard_matrix_full.csv"
    with open(jaccard_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + all_labels_block)
        for label, row in zip(all_labels_block, jaccard_matrix_block):
            writer.writerow(
                [label] + [f"{val:.6f}" if not np.isnan(val) else "" for val in row]
            )

    # 5.4 Full by-dimension Jaccard matrices (all DSCs)
    for dim in all_dims:
        jaccard_full_dim_file = output_dir / f"jaccard_matrix_full_dim{dim}.json"
        jaccard_matrix_block_dim, _ = load_jaccard_data(str(jaccard_full_dim_file))

        if jaccard_matrix_block_dim is None:
            print(f"Computing full symmetric Jaccard matrix for dimension {dim}...")
            jaccard_matrix_block_dim = np.zeros((n_block, n_block))
            for i in range(n_block):
                for j in range(i, n_block):
                    if (
                        all_complexes_block[i] is not None
                        and all_complexes_block[j] is not None
                    ):
                        jaccard_dict = compute_jaccard_distance(
                            all_complexes_block[i],
                            all_complexes_block[j],
                            by_dimension=True,
                        )
                        val = jaccard_dict.get(dim, 0) # type: ignore
                    else:
                        val = np.nan
                    jaccard_matrix_block_dim[i, j] = val
                    jaccard_matrix_block_dim[j, i] = val
            save_jaccard_data(jaccard_matrix_block_dim, {}, str(jaccard_full_dim_file))
        else:
            print(f"Loaded existing Jaccard matrix for dimension {dim} from {jaccard_full_dim_file}")
    # -------------------------------------------------------------------------
    # 7. Comprehensive Analysis Summary
    run.section("analysis summary")
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)

    # Create combined DSC dictionary for analysis
    all_dscs = {}
    for class_idx in range(len(onehot_complexes)):
        all_dscs[f"Q{class_idx + 1} one-hot"] = onehot_complexes[class_idx]
    for idx, (i, j) in enumerate(twohot_indices):
        all_dscs[f"Q{i + 1}+Q{j + 1} two-hot"] = two_hot_complexes[idx]
    all_dscs["Random"] = sc_random

    # Create comprehensive analysis summary
    create_analysis_summary(all_dscs, str(output_dir), model=net,
                           layer_sizes=net.layer_sizes)

    # Combine all statistics into single CSV
    all_stats = onehot_stats + twohot_stats + [random_stats]
    all_labels = onehot_labels + labels_twohot + ["Random"]
    save_dsc_statistics_csv(all_stats, str(output_dir / "dsc_statistics_all.csv"),
                            labels=all_labels)

    print("=" * 80)
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"\nAnalysis outputs:")
    print(f"  - dsc_statistics_onehot.csv")
    print(f"  - dsc_statistics_twohot.csv")
    print(f"  - dsc_statistics_random.csv")
    print(f"  - dsc_statistics_all.csv (combined)")
    print(f"  - dsc_statistics.csv (detailed)")
    print(f"  - dsc_analysis_summary.txt (human-readable)")
    print("=" * 80)


if __name__ == "__main__":
    with RunTimer("quadrants_poly_handcrafted") as run:
        main(run)
