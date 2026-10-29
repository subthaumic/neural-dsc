#!/usr/bin/env python3
"""Split the cost of the correction into threshold effects and closure effects.

`report_correction_sensitivity.py` reports that Bonferroni and the uncorrected
threshold select different complexes on five of the eight trained contexts, and
that the gap is large -- polysemantic Q3 goes from 24/64/92 simplices to
18/32/30. Read beside the p-value trajectories that looks wrong: the four
cutoffs sit close together there and almost no curve crosses between them.

Both are true, and this script says why. Two different things remove a simplex.

  THRESHOLD  the simplex was posed, tested, and its p-value fell above the
             corrected cutoff.
  CLOSURE    the simplex was never posed at all, because the construction is
             inductive and one of its faces had already failed.

Only the first is visible in a trajectory plot. The second is invisible there by
construction -- an untested candidate has no curve to draw -- and on the trained
networks it is much the larger effect: 21 simplices lost to the threshold
against 183 lost to closure.

The consequence is that sensitivity to the correction concentrates at the base
of the induction. Six marginal edges in polysemantic Q3 cost 32 triangles, 62
tetrahedra and the whole of dimension four, while at dimensions two through four
the threshold changes nothing whatsoever.

Reads the cached p-value trajectories only; no network is run.

USAGE:
  python scripts/report_threshold_vs_closure.py
"""

import json
from pathlib import Path

ALPHA = 0.01
D = Path("results/pvalue_trajectories")

print(f"{'context':22s} {'dim':>3} {'m':>5} {'raw':>5} {'bonf':>5} "
      f"{'thresh diff':>11} {'cascade':>8} {'admitted':>9}")
print("-" * 76)
tot_thresh = tot_cascade = 0
for arch in ("mono", "poly"):
    for cls in range(4):
        f = D / f"pvalues_{arch}_trained_class_{cls}.json"
        if not f.exists():
            continue
        recs = json.loads(f.read_text())
        kept = {}
        for dim in sorted({r["dimension"] for r in recs}):
            at = [r for r in recs if r["dimension"] == dim]
            m = len(at)
            def keep(cut):
                return {tuple(r["predictor"]) + (r["target"],) for r in at
                        if r["all_pvalues"][-1] <= cut}
            raw, bonf = keep(ALPHA), keep(ALPHA / m)
            if dim == 1:
                admitted, cascade = bonf, 0
            else:
                admitted = {s for s in bonf
                            if all(s[:i] + s[i+1:] in kept[dim-1]
                                   for i in range(len(s))
                                   if len(s[:i] + s[i+1:]) == dim)}
                cascade = len(bonf) - len(admitted)
            kept[dim] = admitted
            td = len(raw) - len(bonf)
            tot_thresh += td; tot_cascade += cascade
            print(f"{arch}/trained/Q{cls+1:<11d} {dim:>3} {m:>5} {len(raw):>5} "
                  f"{len(bonf):>5} {td:>11} {cascade:>8} {len(admitted):>9}")
print("-" * 76)
print(f"across all trained contexts: {tot_thresh} decided by the threshold, "
      f"{tot_cascade} lost to face closure")
