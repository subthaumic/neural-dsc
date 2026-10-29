#!/usr/bin/env python3
"""Does a 2-simplex ever arrive after its edges, and does that leave a hole?

The filtration closes a simplex under its faces, f_sigma = max(s*, max over
faces), so nothing can appear before its own boundary. Nothing forces it to
appear *with* its boundary either: if the joint ablation of {i,j} only moves k
at a higher dose than either pairwise link needed, the triangle's three edges
stand alone over an interval and the complex has a 1-cycle there. Whether that
ever happens is an empirical question about the networks, and the introduction's
claim that a cycle marks a joint influence absent where the pairwise ones are
present is only interesting if it does.

Three things are reported.

`LATE FILLING` counts simplices whose own s* exceeds every face's filtration
value. This is necessary for a 1-cycle but not sufficient -- the cycle it opens
may already be a boundary of other triangles.

`PERSISTENT H1` is the real test, by GF(2) column reduction over the filtration.
Two conventions are computed because they answer different questions:

  undirected  forget the ordering, keep each vertex set's earliest arrival.
              This is ordinary simplicial homology and is what "a hole in the
              causal structure" should mean.
  ordered     take the ordered simplices as they are. Beware: a reciprocal pair
              (a,b) and (b,a) is then a 1-cycle that no single 2-simplex can
              bound, so this convention reports digons as holes. The DIGON
              CHECK column says how much of the ordered count is that artifact.

Bars shorter than a few grid steps (the ablation grid is 1/(N_GRID-1) wide) are
at the resolution limit and should not be read as structure.

Reads the saved complexes only; no network is run.

USAGE:
  python scripts/report_cycles.py
  python scripts/report_cycles.py --self-test
"""

import re
import sys
from itertools import permutations
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"


def parse(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^\((\d+(?:,\d+)*)\),\s*([0-9.eE+-]+)\s*$", line.strip())
        if m:
            out[tuple(int(x) for x in m.group(1).split(","))] = float(m.group(2))
    return out


def faces(s):
    return [s[:i] + s[i + 1:] for i in range(len(s))]


def underlying(f):
    """Forget the ordering; each vertex set keeps its earliest arrival."""
    out = {}
    for s, v in f.items():
        k = tuple(sorted(set(s)))
        if len(k) == len(s):
            out[k] = min(out.get(k, float("inf")), v)
    for s in out:                      # closure must already hold; check it
        for t in faces(s):
            if t and t in out:
                assert out[t] <= out[s] + 1e-12, f"closure violated at {s}"
    return out


def persistence(cx, max_dim=2):
    """Standard GF(2) column reduction. Returns (dim, birth, death) bars."""
    dim = lambda s: len(s) - 1
    order = sorted(cx, key=lambda s: (cx[s], dim(s), s))
    idx = {s: i for i, s in enumerate(order)}
    cols, killer_of, bars = {}, {}, []
    for j, s in enumerate(order):
        col = {idx[t] for t in faces(s) if t in idx and dim(t) == dim(s) - 1}
        while col:
            lo = max(col)
            if lo not in killer_of:
                killer_of[lo] = j
                cols[j] = col
                if cx[s] - cx[order[lo]] > 1e-12:
                    bars.append((dim(order[lo]), cx[order[lo]], cx[s]))
                break
            col ^= cols[killer_of[lo]]
        else:
            cols[j] = set()
    paired = set(killer_of) | set(killer_of.values())
    for j, s in enumerate(order):
        if j not in paired and not cols.get(j):
            bars.append((dim(s), cx[s], float("inf")))
    return bars


def self_test():
    hollow = {(0,): 0, (1,): 0, (2,): 0, (0, 1): 1, (0, 2): 1, (1, 2): 1}
    h1 = lambda cx: sorted((b, d) for k, b, d in persistence(cx) if k == 1)
    inf = float("inf")

    assert h1(hollow) == [(1, inf)], "hollow triangle should carry one essential H1"
    assert h1({**hollow, (0, 1, 2): 2}) == [(1, 2)], "late filling should give a bar"
    assert h1({**hollow, (0, 1, 2): 1}) == [], "immediate filling should give none"

    square = {(0,): 0, (1,): 0, (2,): 0, (3,): 0,
              (0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): 1}
    assert h1(square) == [(1, inf)], "hollow square should carry one essential H1"

    tet = {(v,): 0 for v in range(4)}
    tet.update({(a, b): 1 for a in range(4) for b in range(a + 1, 4)})
    tet.update({t: 2 for t in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]})
    # K4 at t=1 has |E|-|V|+1 = 3 independent cycles; the four faces kill them
    assert h1(tet) == [(1, 2)] * 3, "tetrahedron boundary should give three H1 bars"
    assert [(b, d) for k, b, d in persistence(tet) if k == 2] == [(2, inf)], "H2"

    print("self-test: 6 cases pass")


def contexts():
    for root in sorted(RESULTS.glob("quadrants_*")):
        for cf in sorted(root.glob("dsc_class_*.txt")):
            f = parse(cf)
            if f:
                name = (f"{root.name.replace('quadrants_', '')}"
                        f"/Q{int(cf.stem.split('_')[-1]) + 1}")
                yield name, f


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    self_test()

    print("\n" + "=" * 78)
    print("LATE FILLING -- simplices whose own s* exceeds every face's filtration")
    print("=" * 78)
    print(f"{'context':26s} {'dim':>3} {'n':>5} {'late':>5} {'max gap':>9}")
    for name, f in contexts():
        for dim in sorted({len(s) - 1 for s in f}):
            if dim < 2:
                continue
            at = [(s, v) for s, v in f.items() if len(s) - 1 == dim]
            gaps = [v - max(f[t] for t in faces(s)) for s, v in at
                    if all(t in f for t in faces(s))]
            late = [g for g in gaps if g > 1e-12]
            if late:
                print(f"{name:26s} {dim:>3} {len(at):>5} {len(late):>5} "
                      f"{max(late):>9.4f}")

    print("\n" + "=" * 78)
    print("PERSISTENT H1")
    print("=" * 78)
    print(f"{'context':26s} {'undirected':>22s} {'ordered':>22s} {'digons':>7}")
    print(f"{'':26s} {'finite / essential / max':>22s} {'finite / essential / max':>22s}")
    for name, f in contexts():
        cells = []
        for cx in (underlying(f), f):
            bars = [b for b in persistence(cx) if b[0] == 1]
            fin = [b for b in bars if b[2] != float("inf")]
            ess = len(bars) - len(fin)
            longest = max((d - b for _, b, d in fin), default=0.0)
            cells.append(f"{len(fin):>6} / {ess:>3} / {longest:>7.4f}")
        edges = {s for s in f if len(s) == 2}
        digons = len({tuple(sorted(s)) for s in edges if (s[1], s[0]) in edges})
        print(f"{name:26s} {cells[0]:>22s} {cells[1]:>22s} {digons:>7}")

    print("\n" + "=" * 78)
    print("UNFILLED TRIPLES -- all three edges present, no 2-simplex in any order")
    print("=" * 78)
    for name, f in contexts():
        und = underlying(f)
        vs = sorted(v[0] for v in und if len(v) == 1)
        open_tris = [(a, b, c)
                     for i, a in enumerate(vs)
                     for j, b in enumerate(vs[i + 1:], i + 1)
                     for c in vs[j + 1:]
                     if all(e in und for e in [(a, b), (a, c), (b, c)])
                     and (a, b, c) not in und]
        if open_tris:
            print(f"{name:26s} {open_tris}")
            for tri in open_tris:
                shown = [f"({a},{b})@{f[(a, b)]:.3f}"
                         for a, b in permutations(tri, 2) if (a, b) in f]
                print(f"{'':26s}   {tri}: {'  '.join(shown)}")


if __name__ == "__main__":
    main()
