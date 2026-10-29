#!/usr/bin/env python3
"""The silver-standard figure, drawn from the run compute_silver_standard.py saved.

Split out of the producer so the comparison and the picture of it are separate
jobs: the producer hard-fails on a fingerprint mismatch and has to be rerun
whenever the quadrant producers move, while this only needs the JSON it left
behind, and can be re-drawn as often as the figure needs changing.

USAGE:
  python scripts/plot_silver_standard.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_PATH = REPO_ROOT / "results" / "silver_standard" / "silver_standard_results.json"
FIGS_DIR = REPO_ROOT / "figs" / "silver_standard"

ARCH_LABELS = {
    "mono": "Axis-feature",
    "poly": "Bottleneck",
}

BASELINE_LABELS = {
    "acdc": "ACDC",
    "eap": "EAP",
    "eap_ig": "EAP-IG",
    "eap_ig_kl": "EAP-IG (KL)",
}


def plot_summary(all_results, output_dir):
    """
    Two things worth seeing: what a baseline could have found, and how well it
    ranked what the DSC found.

    Left column is the composition of each DSC 1-skeleton. Only the bottom
    segment is inside any baseline's range, so the height of the other two is
    the part of the comparison that is a difference between formalisms rather
    than a disagreement about the network. Right column is the threshold-free
    statistic: the probability that a baseline scores a DSC edge above a
    non-DSC one, over the whole representable edge set.
    """
    kinds = [
        ("forward_adjacent", "forward-adjacent (in range)", "#3b6ea5"),
        ("within_layer", "within-layer", "#c98b3a"),
        ("other", "backwards or skipping", "#a34b4b"),
    ]
    baselines = sorted({
        b
        for arch in all_results.values()
        for cls in arch["classes"].values()
        for b in cls["baselines"]
    })

    order = sorted(all_results)
    fig, axes = plt.subplots(len(order), 2, figsize=(13, 4.4 * len(order)), squeeze=False)

    for row, arch_name in enumerate(order):
        arch = all_results[arch_name]
        classes = sorted(arch["classes"])
        positions = np.arange(len(classes))
        labels = [rf"$Q_{{{int(c.removeprefix('class_')) + 1}}}$" for c in classes]
        arch_label = ARCH_LABELS.get(arch_name, arch_name)

        ax = axes[row][0]
        bottom = np.zeros(len(classes))
        for key, label, colour in kinds:
            heights = np.array(
                [arch["classes"][c]["n_by_relation_kind"][key] for c in classes],
                dtype=float,
            )
            ax.bar(positions, heights, 0.6, bottom=bottom, label=label, color=colour)
            bottom += heights
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("1-simplices")
        ax.set_title(f"{arch_label}: composition of the 1-skeleton")
        if row == 0:
            ax.legend(fontsize=9)

        ax = axes[row][1]
        width = 0.8 / len(baselines)
        for i, baseline in enumerate(baselines):
            values = [
                arch["classes"][c]["baselines"][baseline]["membership_auc"] or np.nan
                for c in classes
            ]
            ax.bar(
                positions + i * width,
                values,
                width,
                label=BASELINE_LABELS.get(baseline, baseline),
            )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(positions + 0.4 - width / 2)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("AUC for DSC membership")
        ax.set_title(f"{arch_label}: baseline ranking of DSC edges")
        if row == 0:
            ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    path = output_dir / "silver_standard.pdf"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


def main():
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"No silver-standard results at {RESULTS_PATH.relative_to(REPO_ROOT)}. "
            f"Run scripts/compute_silver_standard.py first."
        )
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary(json.loads(RESULTS_PATH.read_text()), FIGS_DIR)


if __name__ == "__main__":
    main()
