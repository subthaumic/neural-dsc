"""
Accuracy of every quadrant network on the evaluation set.

The evaluation set is the 1,000 points drawn uniformly from [-3, 3]^2 by
`generate_quadrant_data()` (seed 42), the same set for every network. The
article reports these accuracies in its appendix: the handcrafted networks
(1.0 and 0.866), the trained ones (0.997 and 0.999) and the six untrained ones
(0.243--0.258), together with which quadrants each untrained network predicts.

Reads the checkpoints in `models/`; computes nothing else. Writes
`results/accuracies.csv` and prints the table.

USAGE:
  python scripts/report_accuracies.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch

from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
OUTPUT = REPO_ROOT / "results" / "accuracies.csv"

ARCHITECTURES = {
    "mono": ("[2,4,4,4]", create_quadrants_mono),
    "poly": ("[2,3,4,4]", create_quadrants_poly),
}
NETWORKS = [
    (arch, kind)
    for arch in ARCHITECTURES
    for kind in ("handcrafted", "trained", "untrained_seed0", "untrained_seed1", "untrained_seed2")
]


def main():
    X, y = generate_quadrant_data()
    rows = []
    for arch, kind in NETWORKS:
        widths, factory = ARCHITECTURES[arch]
        net = factory()
        net.load_state_dict(torch.load(MODELS_DIR / f"quadrants_{arch}_{kind}.pth"))
        net.eval()
        with torch.no_grad():
            predicted = net(X).argmax(dim=1)
        accuracy = (predicted == y).float().mean().item()
        counts = Counter(predicted.tolist())
        rows.append({
            "network": f"{arch}_{kind}",
            "layer_widths": widths,
            "n_points": len(y),
            "accuracy": round(accuracy, 6),
            "predicted_counts": " ".join(f"Q{q + 1}:{counts.get(q, 0)}" for q in range(4)),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation set: {len(y)} points, true counts "
          + " ".join(f"Q{q + 1}:{int((y == q).sum())}" for q in range(4)))
    for row in rows:
        print(f"  {row['network']:<24} {row['layer_widths']:<10} accuracy {row['accuracy']:.3f}"
              f"   predicted {row['predicted_counts']}")
    print(f"Written to {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
