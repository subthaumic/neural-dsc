"""
Compare the figures drawn by this repository with the copies in the article.

Every figure the article includes is listed in `ARTICLE_FIGURES` with the file
this repository draws for it. Both are rasterised (pdftoppm, 72 dpi) and
compared pixel by pixel, because two PDFs of the same picture differ in their
metadata. A figure counts as identical when no pixel differs.

USAGE:
  python scripts/compare_article_figures.py ARTICLE_FIGS_DIR

ARTICLE_FIGS_DIR is the `figs/` directory of the article source. Needs
`pdftoppm` (poppler) on the PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGS = REPO_ROOT / "figs"
Q = FIGS / "quadrants"

# article path (relative to the article's figs/) -> figure drawn here
ARTICLE_FIGURES = {
    # Figure 2
    "figure2/mono_q1_input.pdf": Q / "mono_handcrafted/quadrants_mono_handcrafted_q0_onehot.pdf",
    "figure2/mono_q1_activations.pdf": Q / "mono_handcrafted/quadrants_mono_handcrafted_figure2_activations.pdf",
    "figure2/mono_q1_dsc.pdf": Q / "mono_handcrafted/quadrants_mono_handcrafted_figure2_dsc.pdf",
    "figure2/poly_q1_input.pdf": Q / "poly_handcrafted/quadrants_poly_handcrafted_q0_onehot.pdf",
    "figure2/poly_q1_activations.pdf": Q / "poly_handcrafted/quadrants_poly_handcrafted_figure2_activations.pdf",
    "figure2/poly_q1_dsc.pdf": Q / "poly_handcrafted/quadrants_poly_handcrafted_figure2_dsc.pdf",
    # Networks and their predictions
    "quadrant_mono_network.pdf": Q / "mono_handcrafted/quadrants_mono_handcrafted_network.pdf",
    "quadrant_poly_network.pdf": Q / "poly_handcrafted/quadrants_poly_handcrafted_network.pdf",
    "quadrant_mono_trained_network.pdf": Q / "mono_trained/quadrants_mono_trained_network.pdf",
    "quadrant_poly_trained_network.pdf": Q / "poly_trained/quadrants_poly_trained_network.pdf",
    "quadrants_mono_predicted_scatter.pdf": Q / "mono_handcrafted/quadrants_mono_handcrafted_predicted_scatter.pdf",
    "quadrants_poly_predicted_scatter.pdf": Q / "poly_handcrafted/quadrants_poly_handcrafted_predicted_scatter.pdf",
    "quadrants_mono_trained_predicted_scatter.pdf": Q / "mono_trained/quadrants_mono_trained_predicted_scatter.pdf",
    "quadrants_poly_trained_predicted_scatter.pdf": Q / "poly_trained/quadrants_poly_trained_predicted_scatter.pdf",
    # Construction and sensitivity
    "inference_sensitivity/probe_count_convergence.pdf": FIGS / "inference_sensitivity/probe_count_convergence.pdf",
    "pvalue_trajectories/pvalue_trajectories_mono_handcrafted_class_0.pdf": FIGS / "pvalue_trajectories/pvalue_trajectories_mono_handcrafted_class_0.pdf",
    "pvalue_trajectories/pvalue_trajectories_poly_handcrafted_class_3.pdf": FIGS / "pvalue_trajectories/pvalue_trajectories_poly_handcrafted_class_3.pdf",
    "hyperparameter_sweep/dt_filtration_stability.pdf": FIGS / "hyperparameter_sweep/dt_filtration_stability.pdf",
    "hyperparameter_sweep/probe_tolerance_stability.pdf": FIGS / "hyperparameter_sweep/probe_tolerance_stability.pdf",
    # Handcrafted complexes
    "mono_onehot_dsc_logic.pdf": FIGS / "handcrafted_logic/mono_onehot_dsc_logic.pdf",
    "poly_onehot_dsc_logic.pdf": FIGS / "handcrafted_logic/poly_onehot_dsc_logic.pdf",
    "poly_q1_vs_random_dsc.pdf": FIGS / "handcrafted_logic/poly_q1_vs_random_dsc.pdf",
    "mono_jaccard_matrix_full.pdf": Q / "mono_handcrafted/jaccard_matrix_full.pdf",
    "poly_jaccard_matrix_full.pdf": Q / "poly_handcrafted/jaccard_matrix_full.pdf",
    # Trained complexes
    **{f"trained_dsc/{arch}_q{k + 1}.pdf": Q / f"{arch}_trained/dsc_class_{k}.pdf"
       for arch in ("mono", "poly") for k in range(4)},
    # Comparisons
    "silver_standard.pdf": FIGS / "silver_standard/silver_standard.pdf",
    **{f"dsc_vs_activation/dsc_vs_activation_quadrants_{name}.pdf":
       FIGS / f"dsc_vs_activation/dsc_vs_activation_quadrants_{name}.pdf"
       for name in ("mono_handcrafted", "poly_handcrafted", "mono_trained", "poly_trained")},
}


def rasterise(pdf: Path, out_dir: Path, tag: str) -> list[np.ndarray]:
    prefix = out_dir / tag
    subprocess.run(["pdftoppm", "-r", "72", "-png", str(pdf), str(prefix)], check=True)
    pages = sorted(out_dir.glob(f"{tag}*.png"))
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.int16) for p in pages]


def compare(a: Path, b: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pa, pb = rasterise(a, tmp, "a"), rasterise(b, tmp, "b")
    if len(pa) != len(pb):
        return f"page count {len(pa)} vs {len(pb)}"
    for x, y in zip(pa, pb):
        if x.shape != y.shape:
            return f"size {x.shape[1]}x{x.shape[0]} vs {y.shape[1]}x{y.shape[0]}"
        diff = np.abs(x - y).max(axis=2)
        if diff.any():
            return f"{(diff > 0).mean():.2%} of pixels differ"
    return "identical"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("article_figs", type=Path)
    args = parser.parse_args()

    n_same = 0
    for rel, ours in ARTICLE_FIGURES.items():
        theirs = args.article_figs / rel
        if not ours.exists():
            result = "not drawn here"
        elif not theirs.exists():
            result = "not in the article"
        else:
            result = compare(theirs, ours)
        n_same += result == "identical"
        print(f"{result:>26}  {rel}")
    print(f"\n{n_same}/{len(ARTICLE_FIGURES)} figures identical")
    sys.exit(0 if n_same == len(ARTICLE_FIGURES) else 1)


if __name__ == "__main__":
    main()
