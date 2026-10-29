"""
Silver Standard: DSC 1-skeletons against three circuit-discovery baselines.

Runs ACDC (Conmy et al. 2023), EAP (Syed, Rager & Conmy 2023) and EAP-IG
(Hanna, Liu & Variengien 2024) on the handcrafted quadrant classifiers, and
sets each against the 1-skeleton of the directed simplicial complex built from
the same probe points under the same locked rule.

WHAT IS AND IS NOT BEING COMPARED

A baseline edge and a DSC edge are different relations on the same eight (or
seven) neurons. A baseline's `u -> v` says that the *weight* from u to v
carries something the task needs, so its edge set is confined to the weight
graph: adjacent layers, forward. A DSC's `u -> v` says that driving u to zero
along a minimum-norm path in *input* space significantly moves v, and nothing
confines that to the wiring -- the input is two-dimensional, so suppressing any
neuron perturbs every neuron downstream of the coordinates it moves, including
neurons in the same layer and in earlier ones.

Measured here, two thirds of a DSC's 1-skeleton is of that second kind. Scoring
those edges as disagreement would be measuring the difference in what the two
formalisms can say, not a difference in what they found, so the overlap
statistics are reported on the forward-adjacent sub-relation and the rest is
reported as what it is: outside every baseline's range. The representable
ceiling is printed with every overlap number so the two cannot be confused.

For the same reason there is no vertex-overlap statistic. `build` seeds
`complex[0]` with every internal neuron at filtration 0 unconditionally, so a
DSC's vertex set is always all of them and any baseline circuit touching all of
them agrees 100% by construction.

Baselines produce a score per edge, not only a set, so the comparison runs at
two levels: containment and Jaccard on the circuits at matched sparsity, and
the AUC of each baseline's score read as a classifier of DSC membership over
the whole representable edge set. The second does not depend on any threshold.

Objectives differ between baselines and it matters. ACDC scores against the KL
divergence from the clean output; that metric is stationary at the clean
output, so single-step EAP on it returns nothing but rounding noise (see
`neural_dsc.acdc.eap`). EAP is therefore run on a task metric, and EAP-IG on both, so
that the ACDC/EAP-IG comparison can be read at a shared objective.

PROBE POINTS AND COMPLEXES

Probes are drawn by rejection sampling from the continuous input domain, with
the seed derived through `probe_seed` exactly as the quadrant scripts derive
it. The draw is therefore the same one, and the DSCs are the same complexes
that appear in the paper's main figures -- they are loaded from the quadrant
results directory when its configuration fingerprint matches this one, and
rebuilt only when it does not.

USAGE:
  python compute_silver_standard.py

To change run parameters, edit the CONFIGURATION block below.
"""

import os

# MUST run before numpy/scipy/torch are imported. See the note in
# compute_quadrants_mono_handcrafted.py.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import json
from pathlib import Path

import numpy as np
import torch

from neural_dsc.acdc import ACDC
from neural_dsc.acdc.eap import (
    edge_attribution_scores,
    kl_to_clean,
    target_logit_diff,
    top_k_circuit,
)
from neural_dsc.acdc.graph import hidden_edges, relation_kind, to_flat_edges
from neural_dsc.directed_simplicial_complex import DirectedSimplicialComplex
from neural_dsc.io.persistence import check_config_fingerprint
from neural_dsc.models.gen import create_quadrants_mono, create_quadrants_poly
from neural_dsc.utils.probe_selection import draw_probe_points, onehot_target, probe_seed
from neural_dsc.utils.progress import RunTimer

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "silver_standard"

# CONFIGURATION
# The locked rule, matching every other producer: see
# compute_inference_sensitivity.py for where these come from.
N_PROBE_POINTS = 128
SIGNIFICANCE_LEVEL = 0.01
FDR_METHOD = "bonferroni"  # over the family of simplices tested at each dimension
RELEVANCE_MARGIN = 0.01  # interval null H0: |theta| <= epsilon, in units of the
                         # target's within-class SD
RELEVANCE_SCALE = "std"
PROBE_TOLERANCE = 0.01  # purity slack below the achievable cosine maximum
PROBE_SEED = 20260819   # the quadrant scripts' seed; changing it decouples this
                        # comparison from the complexes in the main figures
MAX_ITERATIONS = 1000
DT = 0.01

N_CLASSES = 4
CORRUPTION = "flip"  # negates one or both input coordinates, moving every probe
                     # out of its quadrant. The only corruption here that drives
                     # accuracy to zero on this task.
CORRUPTION_SEED = 20260819

ACDC_THRESHOLD = 0.01  # KL, in nats, below which an edge's marginal damage
                       # counts as negligible. Not the significance level it
                       # shares a value with: ACDC has no null hypothesis.
ACDC_MAX_ITER = 1000   # a ceiling, not a budget; the searches here finish in
                       # far fewer, and a truncated search would silently
                       # return a denser circuit than the threshold implies.
IG_STEPS = 16  # gradient samples along the corrupted-to-clean path for EAP-IG

# Each architecture, with the quadrant run whose complexes it reuses.
ARCHITECTURES = {
    "mono": {
        "factory": create_quadrants_mono,
        "max_dimension": 8,  # the ceiling: 8 internal neurons
        "quadrant_results": RESULTS_DIR / "quadrants_mono_handcrafted",
        "producer": "compute_quadrants_mono_handcrafted.py",
    },
    "poly": {
        "factory": create_quadrants_poly,
        "max_dimension": 7,  # the ceiling: 7 internal neurons
        "quadrant_results": RESULTS_DIR / "quadrants_poly_handcrafted",
        "producer": "compute_quadrants_poly_handcrafted.py",
    },
}


def dsc_config(max_dimension):
    """
    The configuration a cached DSC must have been built under to be reusable.

    Byte-identical in content and key order to the dict the quadrant scripts
    fingerprint, so a match here is a guarantee that the loaded complex is the
    one in the main figures rather than a lookalike.
    """
    return {
        "dt": DT,
        "fdr_method": FDR_METHOD,
        "max_dimension": max_dimension,
        "max_iterations": MAX_ITERATIONS,
        "n_probe_points": N_PROBE_POINTS,
        "probe_seed": PROBE_SEED,
        "probe_seed_rule": "base + 1000*class + n_points",
        "probe_selection": "purity_rejection_sampling",
        "probe_tolerance": PROBE_TOLERANCE,
        "relevance_margin": RELEVANCE_MARGIN,
        "relevance_scale": RELEVANCE_SCALE,
        "significance_level": SIGNIFICANCE_LEVEL,
    }


def draw_probes(model, class_idx, spec, label):
    """
    Probe points for one class, from the continuous input domain.

    Checked against the draw the quadrant run actually used. The configuration
    fingerprint cannot do this: it records the constants, not the sampler, so a
    change inside `neural_dsc.utils.probe_selection` would leave every constant
    matching while the points moved. The baselines below run on these probes
    and the complex was built on those, so a divergence would compare a DSC on
    one probe set against a circuit on another and report the difference as a
    disagreement about the network.
    """
    probes = draw_probe_points(
        model,
        onehot_target(N_CLASSES, class_idx),
        N_PROBE_POINTS,
        tolerance=PROBE_TOLERANCE,
        seed=probe_seed(PROBE_SEED, class_idx, N_PROBE_POINTS),
        label=label,
    ).points

    cached = spec["quadrant_results"] / "probe_data_onehot.pt"
    if not cached.exists():
        raise FileNotFoundError(
            f"no probe data in {spec['quadrant_results'].relative_to(REPO_ROOT)}; "
            f"run {spec['producer']} first."
        )
    reference = torch.load(cached, weights_only=False)[class_idx]
    if probes.shape != reference.shape or not torch.equal(probes, reference):
        raise RuntimeError(
            f"the probe draw for class {class_idx} no longer reproduces the one "
            f"{spec['producer']} used (shapes {tuple(probes.shape)} vs "
            f"{tuple(reference.shape)}). Something in neural_dsc.utils.probe_selection "
            f"changed the sampler without changing the configuration, so the "
            f"complex and the baselines would be measuring different probe "
            f"sets. Rerun {spec['producer']}."
        )
    return probes


def load_dsc(class_idx, spec):
    """
    The DSC for one class, from the quadrant run that produced it.

    This script does not build complexes. The quadrant scripts are the
    producers, they run on the server, and reusing their output is what makes
    the comparison a comparison against the complexes in the paper's figures
    rather than against lookalikes rebuilt under the same nominal settings.

    A cached complex is accepted only if the quadrant run's fingerprint records
    exactly this configuration -- not merely that a file with the right name
    exists. Anything else is a hard failure naming the script to run first,
    because the alternative is an eight-DSC build starting wherever this
    happened to be invoked.
    """
    cached = spec["quadrant_results"] / f"dsc_class_{class_idx}.txt"
    fingerprint = spec["quadrant_results"] / "config_fingerprint.json"
    wanted = dsc_config(spec["max_dimension"])

    if not cached.exists() or not fingerprint.exists():
        raise FileNotFoundError(
            f"no complex for class {class_idx} in "
            f"{spec['quadrant_results'].relative_to(REPO_ROOT)}. Run "
            f"{spec['producer']} first; this script compares, it does not build."
        )

    with open(fingerprint) as stream:
        stored = json.load(stream).get("config")
    if stored != wanted:
        differing = sorted(
            key
            for key in set(wanted) | set(stored or {})
            if (stored or {}).get(key) != wanted.get(key)
        )
        raise RuntimeError(
            f"{cached.relative_to(REPO_ROOT)} was built under a different "
            f"configuration; it disagrees on {differing}. Rerun "
            f"{spec['producer']} under the current rule rather than comparing "
            f"against a stale complex."
        )

    sc = DirectedSimplicialComplex()
    sc.load(str(cached))
    print(f"    reused {cached.relative_to(REPO_ROOT)}")
    return sc


def dsc_one_skeleton(sc):
    """Ordered (predictor, target) pairs with their filtration values."""
    return {tuple(simplex): value for simplex, value in sc.complex.get(1, [])}


def run_baselines(model, probes, class_idx):
    """
    All three baselines on one probe set, against one shared counterfactual.

    Returns a dict of baseline name -> {"scores": edge -> float, "circuit":
    set of edges}. ACDC's circuit is what its threshold selects; the
    attribution circuits are the same number of highest-magnitude edges, so the
    three are compared at matched sparsity rather than at thresholds that mean
    different things.
    """
    acdc = ACDC(model, threshold=ACDC_THRESHOLD, metric="kl_div")
    corrupted = acdc.create_corrupted_data(
        probes, CORRUPTION, seed=CORRUPTION_SEED + class_idx
    )

    with torch.no_grad():
        targets = torch.full((len(probes),), class_idx, dtype=torch.long)
        clean_accuracy = (model(probes).argmax(1) == targets).float().mean().item()
        corrupted_accuracy = (
            (model(corrupted).argmax(1) == targets).float().mean().item()
        )

    result = acdc.discover_circuit(probes, corrupted, max_iterations=ACDC_MAX_ITER)
    if len(result.iteration_history) >= ACDC_MAX_ITER:
        raise RuntimeError(
            f"ACDC hit its iteration ceiling ({ACDC_MAX_ITER}) for class "
            f"{class_idx}; the returned circuit is denser than the threshold "
            f"implies. Raise ACDC_MAX_ITER."
        )
    k = len(result.edges)

    task_metric = target_logit_diff(class_idx)
    baselines = {
        "acdc": {"scores": dict(result.edge_scores), "circuit": set(result.edges)},
    }
    for name, metric, steps in (
        ("eap", task_metric, 1),
        ("eap_ig", task_metric, IG_STEPS),
        ("eap_ig_kl", kl_to_clean, IG_STEPS),
    ):
        scores = edge_attribution_scores(model, probes, corrupted, metric, n_steps=steps)
        baselines[name] = {"scores": scores, "circuit": top_k_circuit(scores, k)}

    # The AUC compares baselines to each other, so they have to be scored over
    # the same edges. ACDC only records a score for an edge it examined, and it
    # stops examining a layer as soon as nothing in it is below threshold --
    # which still leaves every edge of that layer scored, but only because the
    # last pass scored them all before breaking. Check rather than rely on it.
    for name, data in baselines.items():
        missing = hidden_edges(model) - set(data["scores"])
        if missing:
            raise RuntimeError(
                f"{name} scored only {len(data['scores'])} of "
                f"{len(hidden_edges(model))} edges for class {class_idx}; "
                f"the membership AUC would be computed over a different edge "
                f"set than the other baselines'. Missing: {sorted(missing)}"
            )

    diagnostics = {
        "clean_accuracy": clean_accuracy,
        "corrupted_accuracy": corrupted_accuracy,
        "performance_drop": clean_accuracy - corrupted_accuracy,
        "acdc_iterations": len(result.iteration_history),
        "acdc_final_metric": result.final_metric,
        "matched_sparsity_k": k,
        "n_representable_edges": len(acdc.active_edges),
    }
    return baselines, diagnostics


def membership_auc(model, scores, dsc_edges):
    """
    How well a baseline's score separates the DSC's edges from the rest.

    The probability that a randomly chosen DSC edge outscores a randomly chosen
    non-DSC edge, over the whole representable edge set, ties counted as half.
    Threshold-free, so it does not inherit the arbitrariness of where a circuit
    is cut, and it is the one statistic here that uses the baselines' scores
    rather than only their circuits.

    Returns None when every representable edge is in the DSC or none is, where
    the quantity is undefined rather than zero.
    """
    flat = {
        (to_flat_edges(model, [edge]).pop()): abs(value)
        for edge, value in scores.items()
    }
    positive = [v for e, v in flat.items() if e in dsc_edges]
    negative = [v for e, v in flat.items() if e not in dsc_edges]
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative
    )
    return wins / (len(positive) * len(negative))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def compare_class(model, sc, baselines):
    """Set every baseline against one class's 1-skeleton."""
    skeleton = dsc_one_skeleton(sc)
    by_kind = {"forward_adjacent": {}, "within_layer": {}, "other": {}}
    for edge, value in skeleton.items():
        by_kind[relation_kind(model, edge)][edge] = value

    representable = to_flat_edges(model, hidden_edges(model))
    dsc_forward = set(by_kind["forward_adjacent"])

    comparison = {
        "n_dsc_edges": len(skeleton),
        "n_by_relation_kind": {k: len(v) for k, v in by_kind.items()},
        "n_representable_edges": len(representable),
        "representable_ceiling": len(dsc_forward) / len(skeleton) if skeleton else 0.0,
        "dsc_forward_adjacent": sorted(dsc_forward),
        "dsc_filtration": {str(k): v for k, v in sorted(skeleton.items())},
        "baselines": {},
    }

    for name, data in baselines.items():
        circuit = to_flat_edges(model, data["circuit"])
        comparison["baselines"][name] = {
            "n_circuit_edges": len(circuit),
            "circuit": sorted(circuit),
            "shared": sorted(dsc_forward & circuit),
            "dsc_only": sorted(dsc_forward - circuit),
            "baseline_only": sorted(circuit - dsc_forward),
            "dsc_contained_in_baseline": dsc_forward <= circuit,
            "jaccard_restricted": jaccard(dsc_forward, circuit),
            "recall_of_dsc": (
                len(dsc_forward & circuit) / len(dsc_forward) if dsc_forward else None
            ),
            "membership_auc": membership_auc(model, data["scores"], dsc_forward),
        }

    # Baselines against each other, so the DSC's disagreement can be read
    # against how much the established methods disagree among themselves.
    names = sorted(baselines)
    comparison["baseline_agreement"] = {
        f"{a}|{b}": jaccard(
            to_flat_edges(model, baselines[a]["circuit"]),
            to_flat_edges(model, baselines[b]["circuit"]),
        )
        for i, a in enumerate(names)
        for b in names[i + 1:]
    }
    return comparison


def run_architecture(name, spec, output_dir):
    print(f"\n{'=' * 70}\n{name} handcrafted quadrant classifier\n{'=' * 70}")
    model = spec["factory"]()
    model.eval()
    results = {
        "internal_neurons": model.internal_neurons,
        "hidden_sizes": list(model.hidden_sizes),
        "classes": {},
    }
    for class_idx in range(N_CLASSES):
        print(f"\n  class {class_idx}")
        probes = draw_probes(model, class_idx, spec, f"  class {class_idx}")
        sc = load_dsc(class_idx, spec)
        baselines, diagnostics = run_baselines(model, probes, class_idx)
        if diagnostics["performance_drop"] <= 0:
            raise RuntimeError(
                f"{CORRUPTION!r} corruption did not reduce accuracy for {name} "
                f"class {class_idx} ({diagnostics['clean_accuracy']:.3f} -> "
                f"{diagnostics['corrupted_accuracy']:.3f}); the counterfactual "
                f"carries no signal and every baseline score is noise."
            )
        comparison = compare_class(model, sc, baselines)
        comparison["diagnostics"] = diagnostics
        results["classes"][f"class_{class_idx}"] = comparison
        print_class_summary(class_idx, comparison)

    return model, results


def print_class_summary(class_idx, comparison):
    kinds = comparison["n_by_relation_kind"]
    print(
        f"    DSC 1-skeleton: {comparison['n_dsc_edges']} edges "
        f"({kinds['forward_adjacent']} forward-adjacent, "
        f"{kinds['within_layer']} within-layer, {kinds['other']} other) -- at most "
        f"{comparison['representable_ceiling']:.0%} of them is expressible by any "
        f"baseline"
    )
    header = f"    {'baseline':<12} {'edges':>6} {'shared':>7} {'recall':>7} {'J':>6} {'AUC':>6}"
    print(header)
    for name, data in comparison["baselines"].items():
        auc = data["membership_auc"]
        print(
            f"    {name:<12} {data['n_circuit_edges']:>6} {len(data['shared']):>7} "
            f"{data['recall_of_dsc']:>7.2f} {data['jaccard_restricted']:>6.2f} "
            f"{(f'{auc:.2f}' if auc is not None else 'n/a'):>6}"
            + ("  (DSC contained)" if data["dsc_contained_in_baseline"] else "")
        )



def write_summary(all_results, output_dir):
    lines = [
        "Silver Standard: DSC 1-skeletons against ACDC, EAP and EAP-IG",
        "=" * 70,
        "",
        "Overlap statistics are restricted to the forward-adjacent sub-relation,",
        "the only part of a DSC's 1-skeleton any weight-graph method is able to",
        "express. The share of the skeleton that lies there is the ceiling column.",
        "",
    ]
    for arch_name, arch in sorted(all_results.items()):
        lines += [f"{arch_name} handcrafted ({arch['internal_neurons']} internal neurons)",
                  "-" * 70]
        for class_key in sorted(arch["classes"]):
            c = arch["classes"][class_key]
            kinds = c["n_by_relation_kind"]
            lines.append(
                f"  {class_key}: {c['n_dsc_edges']} DSC edges "
                f"({kinds['forward_adjacent']} fwd / {kinds['within_layer']} within "
                f"/ {kinds['other']} other), ceiling {c['representable_ceiling']:.0%}"
            )
            for name, data in c["baselines"].items():
                auc = data["membership_auc"]
                lines.append(
                    f"    {name:<12} circuit {data['n_circuit_edges']:>2}/"
                    f"{c['n_representable_edges']:<2} shared {len(data['shared'])}"
                    f"  recall {data['recall_of_dsc']:.2f}"
                    f"  J {data['jaccard_restricted']:.2f}"
                    f"  AUC {auc:.2f}" if auc is not None
                    else f"    {name:<12} circuit {data['n_circuit_edges']:>2}/"
                    f"{c['n_representable_edges']:<2} shared {len(data['shared'])}"
                    f"  recall {data['recall_of_dsc']:.2f}"
                    f"  J {data['jaccard_restricted']:.2f}  AUC n/a"
                )
            lines.append(
                "    baseline-vs-baseline J: "
                + ", ".join(f"{k} {v:.2f}" for k, v in c["baseline_agreement"].items())
            )
        lines.append("")
    path = output_dir / "summary_report.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(dsc_config(max_dimension=None))
    fingerprint["max_dimension"] = {
        name: spec["max_dimension"] for name, spec in ARCHITECTURES.items()
    }
    fingerprint.update(
        acdc_threshold=ACDC_THRESHOLD,
        architectures=sorted(ARCHITECTURES),
        corruption=CORRUPTION,
        corruption_seed=CORRUPTION_SEED,
        ig_steps=IG_STEPS,
    )
    check_config_fingerprint(fingerprint, OUTPUT_DIR)

    all_results = {}
    for name, spec in ARCHITECTURES.items():
        _, results = run_architecture(name, spec, OUTPUT_DIR)
        all_results[name] = results

    with open(OUTPUT_DIR / "silver_standard_results.json", "w") as stream:
        json.dump(all_results, stream, indent=2, sort_keys=True, default=str)
    print(f"\nWrote {(OUTPUT_DIR / 'silver_standard_results.json').relative_to(REPO_ROOT)}")

    write_summary(all_results, OUTPUT_DIR)
    print("\n  figure: scripts/plot_silver_standard.py")


if __name__ == "__main__":
    with RunTimer("silver standard comparison"):
        main()
