import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import math


def plot_onehot_points(X, y, class_data, n_classes, output_path, colors, markers):
    output_path = Path(output_path)
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        class_indices = y == i
        plt.scatter(
            X[class_indices, 0].numpy(),
            X[class_indices, 1].numpy(),
            c="lightgray",
            alpha=1,
            label=None,
        )
    for i, points in enumerate(class_data):
        plt.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            c=colors[i],
            marker=markers[i],
            alpha=1,
            label=f"Q{i} one-hot",
        )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    quadrants_onehot_pdf = output_path / "quadrants_onehot.pdf"
    if not os.path.exists(quadrants_onehot_pdf):
        plt.tight_layout()
        plt.savefig(quadrants_onehot_pdf)
    plt.close()


def plot_twohot_points(
    X, y, two_hot_data, n_classes, two_hot_indices, output_path, colors, markers
):
    output_path = Path(output_path)
    twohot_colors = get_twohot_colors(colors, two_hot_indices)
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        class_indices = y == i
        plt.scatter(
            X[class_indices, 0].numpy(),
            X[class_indices, 1].numpy(),
            c="lightgray",
            alpha=1,
            label=None,
        )
    for idx, points in enumerate(two_hot_data):
        plt.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            color=twohot_colors[idx],
            marker=markers[idx] if isinstance(markers, (list, tuple)) else markers,
            alpha=1,
            label=f"Q{two_hot_indices[idx][0]}+Q{two_hot_indices[idx][1]}",
        )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    plt.grid(alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    plt.legend(new_handles, new_labels, loc="upper left")
    plt.tight_layout()
    twohot_only_pdf = output_path / "quadrants_twohot.pdf"
    plt.savefig(twohot_only_pdf)
    plt.close()


def get_twohot_colors(colors, two_hot_indices):
    """
    Generate blended colors for two-hot indices based on base colors.
    """
    twohot_colors = []
    for i, j in two_hot_indices:
        c1 = np.array(mcolors.to_rgb(colors[i]))
        c2 = np.array(mcolors.to_rgb(colors[j]))
        avg_color = tuple((c1 + c2) / 2)
        twohot_colors.append(avg_color)
    return twohot_colors


def get_onehot_markers(n_classes):
    return ["s"] * n_classes


def get_twohot_markers(n_classes):
    # Number of two-hot combinations is n_classes choose 2
    n_twohot = math.comb(n_classes, 2)
    return ["^"] * n_twohot


def get_random_marker():
    return "*"
