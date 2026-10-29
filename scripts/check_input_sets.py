"""
Regenerate every input set of the quadrant experiments and compare it with the
stored one, bit for bit.

The input sets are where a code change would first show: every complex, table
and figure is computed from them. Drawing them takes seconds, while building
the complexes takes hours, so this is the cheap end-to-end check that the
selection code, the seeds and the stored models still reproduce the data in
`results/`.

Checked, for the handcrafted and trained networks of both architectures:
  - the four one-hot input sets (`probe_data_onehot.pt`),
  - the four adjacent two-hot input sets (`probe_data_twohot.pt`),
  - the random control (`probe_data_random.pt`),
and, for the untrained networks, that they reuse the trained one-hot sets.

USAGE:
  python scripts/check_input_sets.py [--results DIR]

Exits with status 1 on the first mismatch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from neural_dsc.data.gen import generate_quadrant_data
from neural_dsc.io.persistence import load_probe_data
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.utils.probe_selection import (
    draw_probe_points,
    onehot_target,
    probe_seed,
    softmax_probabilities,
    twohot_target,
)
from neural_dsc.utils.select import get_twohot_indices

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "models"
N_CLASSES = 4


def load_producer(name):
    """Import a producer script as a module, for its configuration constants."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_network(arch, kind):
    net = create_quadrants_mono() if arch == "mono" else create_quadrants_poly()
    state = torch.load(MODELS_DIR / f"quadrants_{arch}_{kind}.pth")
    net.load_state_dict(state)
    net.eval()
    return net


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


DIAGONAL = frozenset({(0, 2), (1, 3)})


def stored_twohot_set(stored, pair, all_pairs, adjacent):
    """The stored two-hot set for `pair`.

    The runs behind the article stored all six pairs, in the order of
    `get_twohot_indices`; this repository stores the four adjacent ones. Both
    layouts are accepted.
    """
    if len(stored) == len(all_pairs):
        return stored[all_pairs.index(pair)]
    if len(stored) == len(adjacent):
        return stored[adjacent.index(pair)]
    raise ValueError(f"unexpected number of stored two-hot sets: {len(stored)}")


def check_producer(arch, kind, results_root):
    producer = load_producer(f"compute_quadrants_{arch}_{kind}")
    out = results_root / f"quadrants_{arch}_{kind}"
    net = load_network(arch, kind)
    n = producer.N_PROBE_POINTS
    extra = {}
    if kind == "trained":
        extra = dict(achievable=1.0, max_attempts=producer.PROBE_MAX_ATTEMPTS,
                     output_transform=softmax_probabilities)
    failures = []

    stored = load_probe_data(str(out / "probe_data_onehot.pt"))
    for i in range(N_CLASSES):
        drawn = draw_probe_points(
            net, onehot_target(N_CLASSES, i), n,
            tolerance=producer.PROBE_TOLERANCE,
            seed=probe_seed(producer.PROBE_SEED, i, n),
            label=f"{arch} {kind} Q{i + 1}", verbose=False, **extra,
        ).points
        if not same(drawn, stored[i]):
            failures.append(f"one-hot Q{i + 1}")

    stored = load_probe_data(str(out / "probe_data_twohot.pt"))
    all_pairs = get_twohot_indices(N_CLASSES)
    adjacent = [p for p in all_pairs if p not in DIAGONAL]
    for pair in adjacent:
        seed_index = all_pairs.index(pair)
        drawn = draw_probe_points(
            net, twohot_target(N_CLASSES, *pair), n,
            tolerance=producer.PROBE_TOLERANCE,
            seed=probe_seed(producer.PROBE_SEED + 100, seed_index, n),
            label=f"{arch} {kind} two-hot {pair}", verbose=False, **extra,
        ).points
        if not same(drawn, stored_twohot_set(stored, pair, all_pairs, adjacent)):
            failures.append(f"two-hot Q{pair[0] + 1}+Q{pair[1] + 1}")

    if kind == "trained":
        # The trained producers draw the control from their training-time
        # evaluation sample, not from the 1,000-point evaluation set.
        X, _ = generate_quadrant_data(n_total=producer.EVAL_DATA_SIZE,
                                      seed=producer.EVAL_DATA_SEED)
    else:
        X, _ = generate_quadrant_data()
    np.random.seed(42)
    rand_indices = np.random.choice(len(X), n, replace=False)
    stored = load_probe_data(str(out / "probe_data_random.pt"))
    if not same(X[rand_indices], stored[0]):
        failures.append("random control")

    return failures


def check_untrained(arch, results_root):
    reference = load_probe_data(str(results_root / f"quadrants_{arch}_trained" / "probe_data_onehot.pt"))
    failures = []
    for seed_dir in sorted((results_root / f"quadrants_{arch}_untrained").glob("seed*")):
        local = load_probe_data(str(seed_dir / "probe_data_onehot.pt"))
        if local is None:
            continue
        for i, (a, b) in enumerate(zip(local, reference)):
            if not same(a, b):
                failures.append(f"{seed_dir.name} one-hot Q{i + 1}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    ok = True
    for arch in ("mono", "poly"):
        for kind in ("handcrafted", "trained"):
            failures = check_producer(arch, kind, args.results)
            status = "identical" if not failures else "MISMATCH: " + ", ".join(failures)
            print(f"quadrants_{arch}_{kind}: 4 one-hot, 4 two-hot, random control -- {status}")
            ok &= not failures
        failures = check_untrained(arch, args.results)
        status = "identical" if not failures else "MISMATCH: " + ", ".join(failures)
        print(f"quadrants_{arch}_untrained: reuses the trained one-hot sets -- {status}")
        ok &= not failures

    if not ok:
        sys.exit(1)
    print("All input sets reproduced exactly.")


if __name__ == "__main__":
    main()
