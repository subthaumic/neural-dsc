#!/usr/bin/env python3
"""Draw what a counterfactual ablation does to the network, along the ablation axis.

A visualisation of the intervention itself, not of a decision taken on it. Give
it a set S of neurons to ablate and a target neuron, and it draws every
internal neuron's activation as a function of

    s = 1 - sum_{i in S} a_i(gamma(s)) / sum_{i in S} a_i(x0),

the fraction of S's baseline activation removed so far, with s = 1 the fully
ablated counterfactual C_S(x0). This is the same trajectory the complexes are
built from -- `MLP.ablation_trajectory`, resampled onto a uniform grid in s --
so the figure shows the intervention the hypothesis tests actually see.

Why the ablation axis and not elapsed time. The path is the minimum-norm descent in
*input* space that drives S to zero; how long that takes is a property of the
integrator and of the local conditioning, and differs from probe to probe. The
ablation is a property of the intervention. Reparameterising by it aligns probes by
how much of S has been removed rather than by the clock, so a mean over probes
is a mean over comparable states.

With `--subsets`, each proper subset of S gets its own panel beside the joint
one. That is the higher-order reading: if ablating S moves the target somewhere
no proper subset of S moves it, the effect belongs to S as a set and not to its
parts, which is exactly the claim a simplex on S makes.

USAGE

  # the 2-simplex (0, 4, 5) on the handcrafted monosemantic net
  python scripts/plot_ablation_trajectory.py --arch mono_handcrafted \
      --class-idx 0 --ablate 0,4 --target 5 --subsets

  # a single-neuron ablation, no subset row
  python scripts/plot_ablation_trajectory.py --arch poly_trained \
      --class-idx 2 --ablate 1 --target 4

Outputs, under `figs/ablation_trajectories/`:

  ablation_<arch>_class<c>_S<...>_to<t>.pdf/png
"""

import os

# MUST run before numpy/scipy/torch are imported. One trajectory on one core has
# no use for a thread per core, and OpenBLAS reads this when the library loads.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.utils.probe_selection import (
    draw_probe_points,
    onehot_target,
    probe_seed,
    softmax_probabilities,
)
from neural_dsc.utils.progress import RunTimer
from neural_dsc.utils.trained_quadrants import load_trained_quadrant_model

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "figs" / "ablation_trajectories"
MODELS_DIR = REPO_ROOT / "models"

# Defaults chosen to match compute_pvalue_trajectories.py, so a figure drawn
# here describes the same probes and the same trajectory as the complex the
# situation was spotted in. Override any of them on the command line.
N_PROBE_POINTS = 128
MAX_ITERATIONS = 1000
DT = 0.01
PROBE_TOLERANCE = 0.01
PROBE_SEED = 20260819
PROBE_MAX_ATTEMPTS = 12_800_000

TRAINED_LAYERS = {"mono": [2, 4, 4, 4], "poly": [2, 3, 4, 4]}

ABLATED_STYLE = dict(linewidth=2.2, linestyle="--", zorder=4)
TARGET_STYLE = dict(color="black", linewidth=3.0, zorder=5)
OTHER_STYLE = dict(linewidth=1.0, alpha=0.45, zorder=2)
BAND_COLOR = "0.55"


def load_trained(capacity):
    """Load a trained quadrant classifier, without training on demand.

    The trained producers own the training configuration and write a summary
    beside the checkpoint; training here would be a second source of truth for
    which weights a figure describes.
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


def build_probes(architecture, class_idx, n_probes, tolerance, seed, n_classes=4):
    """The same probe set the complexes are built on, rebuilt deterministically."""
    model = MODEL_FACTORIES[architecture]()
    model.eval()
    result = draw_probe_points(
        model,
        onehot_target(n_classes, class_idx),
        n_probes,
        tolerance=tolerance,
        seed=probe_seed(seed, class_idx, n_probes),
        label=f"{architecture}/class_{class_idx} n={n_probes}",
        achievable=1.0 if architecture.endswith("_trained") else None,
        max_attempts=PROBE_MAX_ATTEMPTS,
        output_transform=(
            softmax_probabilities if architecture.endswith("_trained") else None
        ),
        verbose=False,
    )
    return model, result


def run_trajectory(model, points, ablate, n_steps, dt):
    """Activations over the ablation grid, plus how completely the ablation landed.

    The report matters as much as the curves. A probe whose predictors never
    reach zero has not reached the counterfactual, and its tail is the
    integrator giving up rather than the network's response, so the completion
    numbers are printed beside every figure.
    """
    acts, report = model.ablation_trajectory(
        points, list(ablate), n_steps=n_steps, dt=dt
    )
    acts = acts.detach().cpu().numpy()
    diagnostics = {
        "n_probes": int(acts.shape[1]),
        "n_ablated": int(report["ablated"].sum()),
        "worst_residual_fraction": float(report["residual_fraction"].max()),
        "n_stalled": int(report["stalled"].sum()),
        "n_truncated": int(report["truncated"].sum()),
        "internal_steps": int(report["internal_steps"]),
    }
    return acts, diagnostics


def draw_panel(ax, acts, ablate, target, title, colors, show_band=True):
    """One ablation-response panel: every neuron's activation against s.

    The target carries a band over the probe set because the mean alone hides
    the case the tests are most sensitive to -- an effect carried by a subset of
    probes rather than by all of them, which moves the pseudomedian the
    signed-rank test estimates without moving the mean much.
    """
    n_steps, _, n_neurons = acts.shape
    ablation_grid = np.linspace(0.0, 1.0, n_steps)
    mean = acts.mean(axis=1)

    if show_band and target is not None:
        lo = np.percentile(acts[:, :, target], 25, axis=1)
        hi = np.percentile(acts[:, :, target], 75, axis=1)
        ax.fill_between(ablation_grid, lo, hi, color=BAND_COLOR, alpha=0.25, zorder=1,
                        label=f"$n_{{{target}}}$ interquartile range over probes")

    for neuron in range(n_neurons):
        if neuron in ablate:
            ax.plot(ablation_grid, mean[:, neuron], color=colors(neuron % 10),
                    label=f"$n_{{{neuron}}}$ (ablated)", **ABLATED_STYLE)
        elif neuron == target:
            ax.plot(ablation_grid, mean[:, neuron], label=f"$n_{{{neuron}}}$ (target)",
                    **TARGET_STYLE)
        else:
            ax.plot(ablation_grid, mean[:, neuron], color=colors(neuron % 10),
                    label=f"$n_{{{neuron}}}$", **OTHER_STYLE)

    ax.axhline(0.0, color="0.4", linewidth=0.8, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of the ablated set removed, $s$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


def subset_label(subset):
    return "\\{" + ", ".join(f"n_{{{i}}}" for i in sorted(subset)) + "\\}"


def main():
    args = parse_args()
    ablate = tuple(sorted(args.ablate))
    target = args.target

    if target is not None and target in ablate:
        raise SystemExit(
            f"target n{target} is inside the ablated set {list(ablate)}; a "
            "neuron cannot be its own counterfactual outcome."
        )

    model, probe_result = build_probes(
        args.arch, args.class_idx, args.probes, args.tolerance, args.seed
    )
    n_neurons = model.internal_neurons
    for neuron in ablate + ((target,) if target is not None else ()):
        if not 0 <= neuron < n_neurons:
            raise SystemExit(
                f"neuron {neuron} is outside this network's internal neurons "
                f"(0..{n_neurons - 1})."
            )
    print(f"{args.arch} / class {args.class_idx}: {n_neurons} internal neurons, "
          f"{args.probes} probes (min purity "
          f"{probe_result.diagnostics['min_purity']:.3f})")

    # The joint ablation, and -- when asked -- every proper non-empty subset of
    # it, which is what makes the higher-order effect visible.
    sets = [ablate]
    if args.subsets and len(ablate) > 1:
        sets = [
            tuple(sub)
            for size in range(1, len(ablate))
            for sub in itertools.combinations(ablate, size)
        ] + [ablate]

    runs = []
    for subset in sets:
        acts, diagnostics = run_trajectory(
            model, probe_result.points, subset, args.steps, args.dt
        )
        runs.append((subset, acts, diagnostics))
        note = ""
        if diagnostics["n_ablated"] < diagnostics["n_probes"]:
            note = (f"  !! {diagnostics['n_probes'] - diagnostics['n_ablated']} probes "
                    f"did not reach the counterfactual "
                    f"(worst residual {diagnostics['worst_residual_fraction']:.2e}, "
                    f"{diagnostics['n_stalled']} stalled, "
                    f"{diagnostics['n_truncated']} truncated)")
        line = (f"  ablate {list(subset)}: "
                f"{diagnostics['n_ablated']}/{diagnostics['n_probes']} probes ablated, "
                f"{diagnostics['internal_steps']} integrator steps{note}")
        if target is not None:
            baseline = acts[0, :, target].mean()
            endpoint = acts[-1, :, target].mean()
            line += (f"\n    n{target}: {baseline:.4f} -> {endpoint:.4f} "
                     f"(change {endpoint - baseline:+.4f})")
        print(line)

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(runs), figsize=(6.0 * len(runs), 5.2),
                             squeeze=False, sharey=True)
    axes = axes[0]
    for ax, (subset, acts, _) in zip(axes, runs):
        endpoint = ""
        if target is not None:
            endpoint = f"  ($n_{{{target}}} \\to {acts[-1, :, target].mean():.3f}$)"
        draw_panel(ax, acts, set(subset), target,
                   f"ablate ${subset_label(subset)}${endpoint}", colors,
                   show_band=args.band)
    axes[0].set_ylabel(f"activation (mean over {args.probes} probes)")
    axes[-1].legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)

    joint = subset_label(ablate)
    title = f"Counterfactual ablation of ${joint}$ — {args.arch}, Q{args.class_idx + 1}"
    if target is not None:
        title += f", target $n_{{{target}}}$"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    stem = (f"ablation_{args.arch}_class{args.class_idx}"
            f"_S{'-'.join(str(i) for i in ablate)}"
            + (f"_to{target}" if target is not None else ""))
    save(fig, stem, args.outdir)


def save(fig, name, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


def neuron_list(text):
    return [int(part) for part in str(text).replace(" ", "").split(",") if part != ""]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch", default="mono_handcrafted",
                        choices=sorted(MODEL_FACTORIES))
    parser.add_argument("--class-idx", type=int, default=0,
                        help="which class context the probes are drawn for")
    parser.add_argument("--ablate", type=neuron_list, required=True,
                        help="comma-separated neuron indices, e.g. 0,4")
    parser.add_argument("--target", type=int, default=None,
                        help="neuron to highlight as the outcome; optional")
    parser.add_argument("--subsets", action="store_true",
                        help="also draw every proper non-empty subset of the "
                             "ablated set, which is what exposes a higher-order "
                             "effect")
    parser.add_argument("--probes", type=int, default=N_PROBE_POINTS)
    parser.add_argument("--steps", type=int, default=MAX_ITERATIONS,
                        help="resolution of the ablation grid")
    parser.add_argument("--dt", type=float, default=DT,
                        help="integrator step; does not change where the "
                             "trajectory ends, only how accurately it gets there")
    parser.add_argument("--tolerance", type=float, default=PROBE_TOLERANCE)
    parser.add_argument("--seed", type=int, default=PROBE_SEED)
    parser.add_argument("--no-band", dest="band", action="store_false",
                        help="drop the interquartile band on the target")
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    with RunTimer("ablation trajectory figure"):
        main()
