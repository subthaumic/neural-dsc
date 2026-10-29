#!/usr/bin/env python3
"""Collect the raw p-value trajectories behind the appendix's stabilisation figure.

Every other question about the decision rule -- which margin, which alpha, which
correction, which trajectory length -- is answered by
`compute_inference_sensitivity.py`, which sweeps them all and records simplex
counts. One thing that sweep cannot answer is what a p-value *does* along a
trajectory, because it discards the raw p-values once it has scored the complex.
This script keeps them.

It builds one complex per requested context at the locked configuration with
`collect_pvalues=True` and writes each cache to disk. A context takes around a
minute on one core, and contexts are independent, so they are dispatched through
one outer process pool. Each construction itself stays serial: unlike the
epsilon=0 sensitivity baseline, these non-zero-epsilon complexes are pruned too
quickly for an inner pool to repay its startup cost.
`plot_pvalue_trajectories.py` draws the figures from the caches without rebuilding
anything.

Outputs, under `results/pvalue_trajectories/`:

  pvalues_<arch>_class_<c>.json   one record per (predictor, target) candidate,
                                  each carrying the full p-value sequence
  dsc_<arch>_class_<c>.txt        the complex those p-values produced
  config_fingerprint.json         the configuration fingerprint
"""

import os

# MUST run before numpy/scipy/torch are imported. OpenBLAS reads its thread
# count when the shared library loads. Each outer worker runs one serial
# construction and has no use for a BLAS thread per core.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neural_dsc.directed_simplicial_complex import (
    DirectedSimplicialComplex,
    pin_worker_threads,
)
from neural_dsc.io.persistence import check_config_fingerprint, save_pvalues
from neural_dsc.mlp import MLP
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.utils.probe_selection import (
    draw_probe_points,
    onehot_target,
    probe_seed,
    softmax_probabilities,
)
from neural_dsc.utils.progress import RunTimer
from neural_dsc.utils.trained_quadrants import file_sha256, load_trained_quadrant_model

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "pvalue_trajectories"

MODELS_DIR = REPO_ROOT / "models"

# Layer sizes must match the trained producers exactly, or load_state_dict fails
# loudly rather than quietly reconstructing a different network.
TRAINED_LAYERS = {"mono": [2, 4, 4, 4], "poly": [2, 3, 4, 4]}


def load_trained(capacity):
    """Load a trained quadrant classifier produced by compute_quadrants_*_trained.py.

    Deliberately does not train on demand. The trained producers own the training
    configuration (seed, epochs, learning rate) and write a training_summary.txt
    beside the checkpoint; a second training path here would be a second source
    of truth for which weights the figures describe.
    """
    path = MODELS_DIR / f"quadrants_{capacity}_trained.pth"
    if not path.exists():
        raise SystemExit(
            f"No trained checkpoint at {path}. Run "
            f"scripts/compute_quadrants_{capacity}_trained.py first."
        )
    return load_trained_quadrant_model(path, TRAINED_LAYERS[capacity])


MODEL_FACTORIES = {
    "mono_handcrafted": create_quadrants_mono,
    "poly_handcrafted": create_quadrants_poly,
    "mono_trained": lambda: load_trained("mono"),
    "poly_trained": lambda: load_trained("poly"),
}
# The ceiling is the internal neuron count: 4+4 for mono, 3+4 for poly.
MAX_DIMENSIONS = {
    "mono_handcrafted": 8, "poly_handcrafted": 7,
    "mono_trained": 8, "poly_trained": 7,
}

# The locked decision rule, as decided by compute_inference_sensitivity.py over
# the fine margin grid. eps sits at the geometric centre of the 0.025-0.1 band
# that converges in probe count in every context under every correction, and B
# is the smallest count at which the five corrections agree.
ALPHA = 0.01
RELEVANCE_MARGIN = 0.01
RELEVANCE_SCALE = "std"
N_PROBE_POINTS = 128
MAX_ITERATIONS = 1000
DT = 0.01
PROBE_TOLERANCE = 0.01
PROBE_SEED = 20260819
PROBE_MAX_ATTEMPTS = 12_800_000
N_WORKERS = None  # None or -1 = all CPUs but four, floored at 1

# The complex is built uncorrected so the cache is the most permissive one: the
# p-values depend on the probe set, the trajectory and epsilon, but not on alpha
# or the correction, so every correction's threshold can be drawn over these
# same p-values afterwards.
FDR_CORRECTION = "none"

# Every handcrafted and trained context. One construction is around a minute on
# one core, so the full set is cheap enough that there is no reason to pick a
# representative class in advance -- the point of the figure is which class is
# representative, and that is not knowable before looking.
DEFAULT_CONTEXTS = [
    (architecture, class_idx)
    for architecture in MODEL_FACTORIES
    for class_idx in range(4)
]


def build_context(architecture, class_idx, n_classes=4):
    """Build one complex, returning it together with its p-value cache."""
    model = MODEL_FACTORIES[architecture]()
    model.eval()

    probe_result = draw_probe_points(
        model,
        onehot_target(n_classes, class_idx),
        N_PROBE_POINTS,
        tolerance=PROBE_TOLERANCE,
        seed=probe_seed(PROBE_SEED, class_idx, N_PROBE_POINTS),
        label=f"{architecture}/class_{class_idx}",
        achievable=1.0 if architecture.endswith("_trained") else None,
        max_attempts=PROBE_MAX_ATTEMPTS,
        output_transform=(
            softmax_probabilities if architecture.endswith("_trained") else None
        ),
    )
    print(f"  {len(probe_result.points)} probe points, "
          f"acceptance {probe_result.acceptance_rate:.4f}, "
          f"achievable purity {probe_result.achievable_purity:.4f}")

    sc = DirectedSimplicialComplex()
    result = sc.build(
        model=model,
        probe_data=probe_result.points,
        max_dimension=MAX_DIMENSIONS[architecture],
        dt=DT,
        max_iterations=MAX_ITERATIONS,
        significance_level=ALPHA,
        fdr_correction=FDR_CORRECTION,
        collect_pvalues=True,
        relevance_margin=RELEVANCE_MARGIN,
        relevance_scale=RELEVANCE_SCALE,
        n_workers=1,
    )
    # With collect_pvalues the builder returns (complex, pvalues).
    sc, pvalues = result
    return sc, pvalues


def resolve_outer_workers(configured_workers):
    """Resolve the outer-pool width, leaving four CPUs free in auto mode."""
    if configured_workers is None or configured_workers == -1:
        return max(1, mp.cpu_count() - 4)
    if configured_workers < 1:
        raise ValueError(
            f"n_workers must be positive, None, or -1; got {configured_workers}"
        )
    return configured_workers


def context_paths(outdir, architecture, class_idx):
    """Return the p-value cache and DSC paths for one context."""
    stem = f"{architecture}_class_{class_idx}"
    return (
        outdir / f"pvalues_{stem}.json",
        outdir / f"dsc_{stem}.txt",
    )


def compute_and_save_context(unit):
    """Build and checkpoint one independently schedulable context."""
    architecture, class_idx, outdir = unit
    pin_worker_threads()
    print(f"\n--- {architecture} / class {class_idx} ---", flush=True)

    sc, pvalues = build_context(architecture, class_idx)
    cache, dsc_path = context_paths(outdir, architecture, class_idx)

    # The p-value JSON is the completion marker used by the resume logic, so
    # publish it only after the complex has been saved successfully. Temporary
    # files prevent an interrupted write from looking like a completed context.
    dsc_tmp = dsc_path.with_name(f".{dsc_path.name}.{os.getpid()}.tmp")
    cache_tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
    sc.save(str(dsc_tmp))
    dsc_tmp.replace(dsc_path)
    save_pvalues(pvalues, str(cache_tmp))
    cache_tmp.replace(cache)

    total = sum(len(simplices) for simplices in sc.complex.values())
    return architecture, class_idx, len(pvalues), total


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contexts", nargs="*", metavar="ARCH:CLASS",
        help="contexts to build, e.g. mono_handcrafted:0 (default: all contexts)",
    )
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="rebuild contexts that already have a cache")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=N_WORKERS,
        help="outer worker count (default: all CPUs but four; -1 is equivalent)",
    )
    return parser.parse_args()


def resolve_contexts(raw):
    if not raw:
        return DEFAULT_CONTEXTS
    contexts = []
    for item in raw:
        architecture, _, class_text = item.partition(":")
        if (
            architecture not in MODEL_FACTORIES
            or not class_text.isdigit()
            or int(class_text) not in range(4)
        ):
            raise SystemExit(
                f"Bad context {item!r}; expected ARCH:CLASS with ARCH in "
                f"{sorted(MODEL_FACTORIES)} and CLASS in 0..3"
            )
        contexts.append((architecture, int(class_text)))
    return list(dict.fromkeys(contexts))


def main(args, run):
    contexts = resolve_contexts(args.contexts)
    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        n_workers = resolve_outer_workers(args.n_workers)
    except ValueError as error:
        raise SystemExit(error) from error

    trained_checkpoint_hashes = {}
    for architecture, _ in contexts:
        if not architecture.endswith("_trained"):
            continue
        capacity = architecture.removesuffix("_trained")
        path = MODELS_DIR / f"quadrants_{capacity}_trained.pth"
        if not path.exists():
            raise SystemExit(
                f"No trained checkpoint at {path}. Run the corresponding "
                "trained quadrant producer first."
            )
        trained_checkpoint_hashes[capacity] = file_sha256(path)

    # Every cached .json and .txt is reused on sight by the plotter, so pin the
    # configuration they were produced under: a changed constant has to fail
    # here rather than leave the figures describing a test nobody ran.
    check_config_fingerprint(
        {
            "alpha": ALPHA,
            "relevance_margin": RELEVANCE_MARGIN,
            "relevance_scale": RELEVANCE_SCALE,
            "n_probe_points": N_PROBE_POINTS,
            "max_iterations": MAX_ITERATIONS,
            "dt": DT,
            "probe_tolerance": PROBE_TOLERANCE,
            "probe_max_attempts": PROBE_MAX_ATTEMPTS,
            "probe_seed": PROBE_SEED,
            "probe_seed_rule": "base + 1000*class + n_points",
            "fdr_correction": FDR_CORRECTION,
            "probe_selection": "purity_rejection_sampling",
            "trained_probe_output_space": "softmax_probabilities",
            "trained_onehot_reference_purity": 1.0,
            "handcrafted_probe_output_space": "raw_nonnegative_outputs",
            "trained_checkpoint_sha256": trained_checkpoint_hashes,
        },
        args.outdir,
    )

    print(f"Output directory: {args.outdir}")
    print(f"Locked rule: alpha={ALPHA}, eps={RELEVANCE_MARGIN} x {RELEVANCE_SCALE}, "
          f"B={N_PROBE_POINTS}, K={MAX_ITERATIONS}, dt={DT}")
    print(f"Outer workers: {n_workers} of {mp.cpu_count()} CPUs")
    print("Inner workers per construction: 1")

    pending = []
    for architecture, class_idx in contexts:
        cache, dsc_path = context_paths(args.outdir, architecture, class_idx)
        if cache.exists() and dsc_path.exists() and not args.force:
            print(
                f"  {architecture}/class_{class_idx}: cached, skipping; "
                "--force to rebuild"
            )
            continue
        if (cache.exists() or dsc_path.exists()) and not args.force:
            print(
                f"  {architecture}/class_{class_idx}: incomplete cache, rebuilding"
            )
        pending.append((architecture, class_idx, args.outdir))

    if not pending:
        print("All requested contexts were found in the cache.")
        return

    lanes = min(n_workers, len(pending))
    print(
        f"Dispatching {len(pending)} contexts over {lanes} outer workers; "
        "each DSC build is serial."
    )
    run.section("outer-parallel context construction")

    if lanes == 1:
        completed = map(compute_and_save_context, pending)
        for i, result in enumerate(completed, 1):
            architecture, class_idx, n_pvalues, total = result
            print(
                f"  completed {architecture}/class_{class_idx}: "
                f"{n_pvalues} p-value records, {total} simplices"
            )
            run.eta(i, len(pending), "contexts")
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=lanes,
        mp_context=ctx,
        initializer=pin_worker_threads,
    ) as pool:
        futures = {
            pool.submit(compute_and_save_context, unit): unit[:2]
            for unit in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            architecture, class_idx = futures[future]
            try:
                _, _, n_pvalues, total = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"failed to build {architecture}/class_{class_idx}"
                ) from error
            print(
                f"  completed {architecture}/class_{class_idx}: "
                f"{n_pvalues} p-value records, {total} simplices"
            )
            run.eta(i, len(pending), "contexts")


if __name__ == "__main__":
    args = parse_args()
    with RunTimer("p-value trajectories") as run:
        main(args, run)
