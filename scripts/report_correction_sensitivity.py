#!/usr/bin/env python3
"""What the choice of multiplicity correction costs, read off cached p-values.

The complexes are built under Bonferroni, applied to the family of simplices
tested at each dimension. Bonferroni because it assumes nothing about how the
family's p-values depend on each other -- simplices sharing a predictor set are
scored from one shared ablation trajectory, so PRDS is not established and BH
does not apply -- and because it controls the familywise error rather than the
false discovery rate, which is the stronger claim. That is an argument from what
we can assume, not from what the data shows, so this script asks what the
assumption actually costs, which `compute_inference_sensitivity.py` cannot
answer because it discards the raw p-values once it has scored a complex.

BY and BH are reported alongside it, and the uncorrected threshold as the
reference against which any correction's cost is measured. Note that Bonferroni
is not uniformly stricter than BY: alpha/m is flat where BY's (alpha/H_m) * k/m
rises with the number of rejections, so the two cross at k = H_m and which one
binds is a property of the data.

Every table reads its families from a replay of the construction, not from the
cache grouped by dimension. Construction is inductive, so a simplex is only
posed when all of its faces survived, and a stricter correction silences whole
dimensions rather than merely thinning them.

Four things carry the argument.

`separation` reports where each threshold falls and, directly, how many
simplices each method admits that production does not. The gap in decades around
production's threshold says how isolated its own verdict is -- but a wide gap
does not by itself mean the methods agree, because another threshold can sit at
the far side of that same void. When the extra counts are zero the conservative
choice is free, which is the outcome to hope for; when they are not, the
STRUCTURE table says what it costs.

`endpoint vs any ablation` is what the decision rule discards. Membership is settled
at full ablation, so a candidate significant at some intermediate ablation but not
at s = 1 is not admitted. Counting those says how much the fixed-sequence
reading actually excludes, and `ragged` counts the trajectories that cross,
cross back, and cross again -- the ones for which the filtration marks where the
flickering stopped rather than where the effect began.

`disagreements` lists every candidate the three thresholds do not agree on. If
this is empty the correction is not doing any work at these effect sizes.

`symmetry` is a consistency check, not an objective. The handcrafted networks
treat the four quadrants identically by construction, so a correct method must
return isomorphic complexes across their four classes; a rule that returns three
different answers is detectably wrong. Trained networks have no such guarantee
-- initialisation and SGD break the symmetry for real -- so the check is
reported for them but proves nothing either way. Choosing a rule *because* it
symmetrises a trained network would be fitting the method to the answer.

Reads `results/pvalue_trajectories/pvalues_*.json`; writes only to stdout.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_dsc.directed_simplicial_complex import build_dsc_from_cached_pvalues
from neural_dsc.io.persistence import load_pvalues
from neural_dsc.utils.fdr_correction import family_cutoff, persistent_crossing_index

from compute_pvalue_trajectories import (  # noqa: E402  (path set above)
    ALPHA,
    DT,
    MAX_DIMENSIONS,
    MODEL_FACTORIES,
    N_PROBE_POINTS,
    OUTPUT_DIR,
)

# The one-sided signed-rank test on B pairs cannot report a p below 2^-B, so a
# cutoff at that value is bounded by the probe count rather than by the effect.
SIGNED_RANK_FLOOR = 2.0 ** -N_PROBE_POINTS

# The correction in production, the two references it is worth reading against,
# and the uncorrected baseline any cost is measured from. Bonferroni belongs
# here for the same reason BY does -- it needs no dependence assumption -- but
# it controls the familywise error rather than the false discovery rate. Its
# threshold alpha/m is flat where BY's rises with the number of rejections, so
# BY is the stricter of the two when few hypotheses are rejected (k < H_m) and
# the more permissive once many are; which one binds is a property of the data,
# not a fact about the methods.
METHODS = ["none", "bh", "by", "bonferroni"]
PRODUCTION = "bonferroni"  # the correction the complexes are built under
METHOD_LABELS = {"none": "uncorrected", "bh": "BH", "by": "BY", "bonferroni": "Bonferroni"}
# Narrow forms, for table columns where the full names do not fit.
METHOD_ABBR = {"none": "raw", "bh": "BH", "by": "BY", "bonferroni": "Bonf"}


def contexts_on_disk(results_dir):
    """Every cached context, in the order MODEL_FACTORIES declares."""
    found = {}
    for path in sorted(results_dir.glob("pvalues_*.json")):
        m = re.fullmatch(r"pvalues_(.+)_class_(\d+)", path.stem)
        if m and m.group(1) in MODEL_FACTORIES:
            found[(m.group(1), int(m.group(2)))] = path
    order = {arch: i for i, arch in enumerate(MODEL_FACTORIES)}
    return sorted(found.items(), key=lambda kv: (order[kv[0][0]], kv[0][1]))


def short(architecture):
    return architecture.replace("_handcrafted", "/hand").replace("_trained", "/train")


_MODEL_CACHE = {}


def model_for(architecture):
    if architecture not in _MODEL_CACHE:
        model = MODEL_FACTORIES[architecture]()
        model.eval()
        _MODEL_CACHE[architecture] = model
    return _MODEL_CACHE[architecture]


def families(records, architecture, alpha, method=PRODUCTION):
    """The candidates the construction actually posed at each dimension.

    The family is the set of simplices tested at one dimension. It is not the
    ablation axis -- each candidate contributes a single test, taken at full
    ablation -- and it is not a regrouping of the cache either. Construction is
    inductive, so a k-simplex is posed only when all of its (k-1)-faces made it
    into the complex; a stricter correction that drops one edge also silences
    every simplex that edge would have supported, and those are never tested.
    Reading them out of the cache would credit the run with hypotheses it never
    asked, and would contradict the STRUCTURE table, which rebuilds properly.

    Returns {dimension: [(simplex, ablation_grid, p_values)]}, empty dimensions dropped.
    """
    _, posed = build_dsc_from_cached_pvalues(
        records, model_for(architecture), MAX_DIMENSIONS[architecture], DT,
        alpha, method, verbose=False, return_families=True,
    )
    return {dim: entry["family"] for dim, entry in sorted(posed.items())
            if entry["family"]}


def cutoffs(family, alpha):
    """Each method's threshold for one dimension, from the final p-values.

    All four are scored on the same family -- the one production posed -- so
    they are comparable to each other.
    """
    finals = np.array([float(p_vals[-1]) for _, _, p_vals in family])
    out = {}
    for method in METHODS:
        cut = family_cutoff(finals, alpha=alpha, method=method)
        out[method] = float(cut) if np.isfinite(cut) else None
    out["none"] = alpha  # the returned value is the largest float below alpha
    return out


def separation_table(loaded, alpha):
    """Where each threshold falls, and which of them part company.

    The gap is measured around production's threshold, so it says how isolated
    that verdict is -- how far the nearest retained p-value sits from the last
    rejected one. A wide gap does not by itself mean the methods agree, because
    another threshold can sit at the far side of the same void, so agreement is
    reported directly as a count of simplices each method admits and production
    does not (and vice versa).
    """
    print("\n" + "=" * 108)
    print("SEPARATION -- where the thresholds fall, and which of them part company")
    print("=" * 108)
    print(f"{'context':<16}{'dim':>4}{'m':>5}" + "".join(
        f"{METHOD_ABBR[m]:>6}" for m in METHODS)
        + f"{'differs from prod':>22}{'largest prod-fired':>20}"
          f"{'smallest prod-kept':>20}{'gap (dec)':>11}{'floor':>7}")
    disagreements = {m: [0, 0] for m in METHODS}  # [admits BY does not, vice versa]
    at_floor_anywhere = False
    for (arch, cls), records, _ in loaded:
        for dim, family in families(records, arch, alpha).items():
            finals = np.array([float(p_vals[-1]) for _, _, p_vals in family])
            cuts = cutoffs(family, alpha)
            fires = {
                m: finals <= (cuts[m] if cuts[m] is not None else -np.inf)
                for m in METHODS
            }
            by_fires = fires[PRODUCTION]
            for m in METHODS:
                disagreements[m][0] += int((fires[m] & ~by_fires).sum())
                disagreements[m][1] += int((~fires[m] & by_fires).sum())
            # Compact "who differs from BY here" cell.
            deltas = [f"{METHOD_ABBR[m]}{int(fires[m].sum()) - int(by_fires.sum()):+d}"
                      for m in METHODS
                      if int(fires[m].sum()) != int(by_fires.sum())]
            delta_cell = ",".join(deltas) if deltas else "-"

            fired, kept = finals[by_fires], finals[~by_fires]
            gap = (np.log10(kept.min() / max(fired.max(), 1e-300))
                   if fired.size and kept.size else float("nan"))
            at_floor = bool(fired.size and fired.min() <= SIGNED_RANK_FLOOR * 1.01)
            at_floor_anywhere |= at_floor

            def cell(value, spec):
                return "-" if value != value else format(value, spec)
            print(f"{short(arch)+f'/Q{cls + 1}':<16}{dim:>4}{len(family):>5}"
                  + "".join(f"{int(fires[m].sum()):>6}" for m in METHODS)
                  + f"{delta_cell:>22}"
                  + f"{cell(fired.max() if fired.size else np.nan, '.2e'):>20}"
                  + f"{cell(kept.min() if kept.size else np.nan, '.2e'):>20}"
                  + f"{cell(gap, '.1f'):>11}"
                  + f"{('yes' if at_floor else '-'):>7}")

    print()
    for method in METHODS:
        if method == PRODUCTION:
            continue
        extra, missing = disagreements[method]
        if not extra and not missing:
            print(f"  {METHOD_LABELS[method]:<12} selects exactly the same simplices "
                  f"as {METHOD_LABELS[PRODUCTION]}.")
        else:
            print(f"  {METHOD_LABELS[method]:<12} admits {extra} that "
                  f"{METHOD_LABELS[PRODUCTION]} does not, and misses {missing} "
                  f"that {METHOD_LABELS[PRODUCTION]} admits.")
    if not any(sum(v) for m, v in disagreements.items() if m != PRODUCTION):
        print("\n  Every threshold falls in the same void, so the conservative choice "
              "is free,\n  which is the reason to make it.")
    if at_floor_anywhere:
        print(f"\n  Some rejected p-values sit at the signed-rank floor 2^-B = "
              f"{SIGNED_RANK_FLOOR:.1e} (B = {N_PROBE_POINTS}). A test cannot\n  report a "
              f"smaller p whatever the effect, so those candidates are tied and cannot be\n"
              f"  ranked against each other. The null is also coarse near the floor at "
              f"small B, so\n  a marginal candidate can land either side of a threshold by "
              f"discreteness alone.\n  Raise B to separate them.")


def endpoint_table(loaded, alpha):
    """What deciding at full ablation discards, and how monotone the survivors are."""
    print("\n" + "=" * 100)
    print("ENDPOINT vs ANY ABLATION -- what the fixed-sequence reading excludes")
    print("=" * 100)
    print(f"{'context':<16}{'dim':>4}{'admitted':>10}{'sig only earlier':>20}"
          f"{'ragged':>9}{'largest gap':>13}")
    total_earlier = total_ragged = 0
    for (arch, cls), records, _ in loaded:
        for dim, family in families(records, arch, alpha).items():
            cuts = cutoffs(family, alpha)
            decisive = cuts[PRODUCTION] if cuts[PRODUCTION] is not None else 0.0
            admitted = only_earlier = ragged = 0
            worst = 0.0
            for _simplex, _ablation_grid, p_vals in family:
                p_values = np.asarray(p_vals, dtype=float)
                flags = p_values <= decisive
                last = persistent_crossing_index(flags)
                if last is None:
                    # Significant somewhere, but not where the decision is taken.
                    only_earlier += bool(flags.any())
                    continue
                admitted += 1
                first = int(np.argmax(flags))
                if first != last:
                    ragged += 1
                    worst = max(worst, (last - first) / max(1, p_values.size - 1))
            total_earlier += only_earlier
            total_ragged += ragged
            print(f"{short(arch)+f'/Q{cls + 1}':<16}{dim:>4}{admitted:>10}"
                  f"{only_earlier:>18}{ragged:>9}{worst:>13.3f}")
    print(f"\n  {total_earlier} candidates fire at some ablation but not at full ablation, "
          f"and are excluded.\n  {total_ragged} of the admitted ones cross back before "
          f"settling, so their filtration marks\n  where the flickering stopped rather "
          f"than where the effect began.")


def disagreement_table(loaded, alpha):
    print("\n" + "=" * 100)
    print(f"DISAGREEMENTS -- every candidate the thresholds decide differently "
          f"(alpha = {alpha})")
    print("=" * 100)
    total = 0
    for (arch, cls), records, _ in loaded:
        for dim, family in families(records, arch, alpha).items():
            cuts = cutoffs(family, alpha)
            for simplex, _ablation_grid, p_vals in family:
                final = float(p_vals[-1])
                fired = {
                    m: (cuts[m] is not None and final <= cuts[m]) for m in METHODS
                }
                if len(set(fired.values())) == 1:
                    continue
                total += 1
                marks = " ".join(
                    f"{METHOD_LABELS[m]}={'Y' if fired[m] else 'n'}" for m in METHODS
                )
                print(f"  {short(arch)+f'/Q{cls + 1}':<14}{str(simplex[:-1]):>12}"
                      f" -> {simplex[-1]:<3} d{dim}  p={final:.2e}  {marks}")
    if not total:
        print("  none -- the three thresholds select exactly the same simplices")
    print(f"\n  {total} disagreeing candidates.")


def structure_table(loaded, alpha):
    """Complex size under each correction, and the per-architecture symmetry check."""
    print("\n" + "=" * 100)
    print("STRUCTURE -- complexes rebuilt from the same p-values under each correction")
    print("=" * 100)
    print(f"{'context':<16}" + "".join(
        f"{METHOD_LABELS[m] + ' (d1/d2/d3)':>24}" for m in METHODS))
    signatures = {}
    for (arch, cls), records, _ in loaded:
        cells = []
        for method in METHODS:
            sc = build_dsc_from_cached_pvalues(
                records, model_for(arch), MAX_DIMENSIONS[arch], DT, alpha, method,
                verbose=False,
            )
            counts = {d: len(s) for d, s in sc.complex.items() if d > 0}
            cells.append(f"{counts.get(1,0)}/{counts.get(2,0)}/{counts.get(3,0)}")
            degrees = Counter(src for e in sc.complex.get(1, {}) for src, _ in [e[0]])
            signatures.setdefault((arch, method), []).append(
                tuple(sorted(degrees.values(), reverse=True))
            )
        print(f"{short(arch)+f'/Q{cls + 1}':<16}" + "".join(f"{c:>24}" for c in cells))

    print("\n  Out-degree multiset per class "
          "(handcrafted: must be identical; trained: no guarantee)")
    for arch in MODEL_FACTORIES:
        if (arch, METHODS[0]) not in signatures:
            continue
        print(f"  {short(arch)}")
        for method in METHODS:
            sigs = signatures[(arch, method)]
            unique = len(set(sigs))
            verdict = "identical" if unique == 1 else f"{unique} distinct"
            print(f"    {METHOD_LABELS[method]:<14}"
                  + "  ".join(str(s) for s in sigs) + f"   [{verdict}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    entries = contexts_on_disk(args.results)
    if not entries:
        raise SystemExit(
            f"No p-value caches in {args.results}. Run compute_pvalue_trajectories.py first."
        )

    loaded = []
    for key, path in entries:
        records = load_pvalues(str(path))
        loaded.append(
            (key, records,
             np.asarray([r["all_pvalues"] for r in records], dtype=float))
        )

    n_ablation_points = loaded[0][2].shape[1]
    n_candidates = sum(len(r) for _, r, _ in loaded)
    print(f"{len(loaded)} contexts, {n_candidates} candidates, alpha = {args.alpha}, "
          f"{n_ablation_points} ablation points per trajectory")
    print("Correction applied per dimension, to the final p-value of each candidate.")
    # The floor is taken from the constant the caches were built under rather
    # than recorded in them, so it is printed here: a smallest observed p-value
    # far below it means the caches came from a different probe count.
    print(f"Signed-rank floor at B = {N_PROBE_POINTS}: 2^-B = {SIGNED_RANK_FLOOR:.2e}; "
          f"smallest p observed = "
          f"{min(float(M.min()) for _, _, M in loaded):.2e}")

    separation_table(loaded, args.alpha)
    endpoint_table(loaded, args.alpha)
    disagreement_table(loaded, args.alpha)
    structure_table(loaded, args.alpha)


if __name__ == "__main__":
    main()
