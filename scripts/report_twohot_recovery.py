"""
Recovery of the two-hot input sets from the Jaccard distance.

For each adjacent pair of quadrants Q_i, Q_j, the complex of the two-hot inputs
X_{Q_iQ_j} counts as recovered when the complexes of X_{Q_i} and X_{Q_j} are
both closer to it than the complexes of the other two one-hot input sets. The
separation of the pair is the smallest distance to the other two one-hot
complexes minus the largest distance to its own two; it is positive exactly
when the pair is recovered.

This is the table on two-hot composition in the article's appendix: pairs
recovered, the smallest separation over the four adjacent pairs, and the range
of the distances among the four one-hot complexes.

Reads `jaccard_matrix_full.csv` of the four quadrant producers; computes
nothing else. Writes `results/twohot_recovery.csv` and prints the table.

USAGE:
  python scripts/report_twohot_recovery.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUTPUT = RESULTS / "twohot_recovery.csv"

NETWORKS = {
    "mono_handcrafted": "[2,4,4,4], handcrafted",
    "poly_handcrafted": "[2,3,4,4], handcrafted",
    "mono_trained": "[2,4,4,4], trained",
    "poly_trained": "[2,3,4,4], trained",
}
ONEHOT = ("Q1", "Q2", "Q3", "Q4")


def read_matrix(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as stream:
        rows = list(csv.reader(stream))
    columns = rows[0][1:]
    return {row[0]: dict(zip(columns, map(float, row[1:]))) for row in rows[1:]}


def main():
    rows = []
    for name, label in NETWORKS.items():
        d = read_matrix(RESULTS / f"quadrants_{name}" / "jaccard_matrix_full.csv")
        separations = {}
        for pair in (column for column in d["Q1"] if "+" in column):
            i, j = pair.split("+")
            own = max(d[i][pair], d[j][pair])
            other = min(d[q][pair] for q in ONEHOT if q not in (i, j))
            separations[pair] = other - own
        onehot = [d[a][b] for a in ONEHOT for b in ONEHOT if a < b]
        rows.append({
            "network": label,
            "pairs_recovered": sum(s > 0 for s in separations.values()),
            "n_pairs": len(separations),
            "smallest_separation": round(min(separations.values()), 6),
            "onehot_dJ_min": round(min(onehot), 6),
            "onehot_dJ_max": round(max(onehot), 6),
            **{f"separation_{pair}": round(s, 6) for pair, s in separations.items()},
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'network':<24} recovered  smallest separation  one-hot d_J range")
    for row in rows:
        span = (f"{row['onehot_dJ_min']:.3f}" if row["onehot_dJ_min"] == row["onehot_dJ_max"]
                else f"{row['onehot_dJ_min']:.3f}--{row['onehot_dJ_max']:.3f}")
        print(f"{row['network']:<24} {row['pairs_recovered']}/{row['n_pairs']}        "
              f"{row['smallest_separation']:+.3f}               {span}")
    print(f"Written to {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
