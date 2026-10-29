from typing import DefaultDict
import numpy as np
import time
import os
import pickle
import re
import math
from collections import defaultdict
from neural_dsc.utils.fdr_correction import (
    FAMILY_METHODS,
    family_cutoff,
    persistent_crossing_index,
)
import multiprocessing as mp
import torch
import copy


def pin_worker_threads():
    """
    Limit numeric libraries in this process to a single thread.
    This avoids thread oversubscription when using multiprocessing.
    """
    # Limit OpenBLAS threads to prevent resource exhaustion
    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
    os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate/vecLib threads
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr threads
    torch.set_num_threads(1)  # Control torch's internal OpenMP threads


# Defaults for the relevance-test globals. `worker_initializer` always sets
# them, but callers that set the worker globals directly (tests, and any code
# predating relevance testing) would otherwise hit a NameError in
# `process_simplex`. Defaulting here makes those paths fall back to the point
# null rather than crash.
worker_relevance_margin = 0.0
worker_relevance_scale = "std"


def worker_initializer(
    model_state_dict,
    layer_sizes,
    probe_data,
    vertices,
    dt,
    max_iterations,
    significance_level,
    fdr_correction="none",
    collect_pvalues=False,
    relevance_margin=0.0,
    relevance_scale="std",
):
    """
    Initializes a worker with its own model instance and shared data
    """
    # First limit this process to use a single thread for numeric operations
    pin_worker_threads()

    # Import locally to avoid serialization issues
    import gc
    from neural_dsc.mlp import MLP

    global worker_model, worker_probe_data, worker_vertices, worker_dt, worker_max_iterations, worker_significance_level, worker_fdr_correction, worker_collect_pvalues, worker_relevance_margin, worker_relevance_scale

    # Create a completely fresh model instance for this worker
    worker_model = MLP(layer_sizes=layer_sizes)

    # Load state dict and set to evaluation mode
    worker_model.load_state_dict(model_state_dict)
    worker_model.eval()
    worker_model.requires_grad_(False)  # No gradients needed for inference

    # Make a separate copy of probe_data for this worker
    # This ensures each worker has its own memory space for the data
    worker_probe_data = probe_data.clone().detach().cpu()

    # Store the immutable parameters
    worker_vertices = vertices
    worker_dt = dt
    worker_max_iterations = max_iterations
    worker_significance_level = significance_level
    worker_fdr_correction = fdr_correction
    worker_collect_pvalues = collect_pvalues
    worker_relevance_margin = relevance_margin
    worker_relevance_scale = relevance_scale

    # Force garbage collection to clean up any shared references
    gc.collect()


def process_simplex(args):
    """
    Check possible extensions of a single simplex using the worker's model and
    pre-initialized parameters.

    Args:
        args: Tuple of (simplex_data, face_filtrations)
            simplex_data: Tuple containing (simplex, parent_filtration)
            face_filtrations: Mapping from codimension-1 faces to filtration values
    """
    # Import locally to avoid serialization issues
    from neural_dsc.utils.pearl_causality import pearl_causal_test

    # Access global variables initialized in worker_initializer
    global worker_model, worker_probe_data, worker_vertices, worker_dt, worker_max_iterations, worker_significance_level, worker_fdr_correction, worker_collect_pvalues, worker_relevance_margin, worker_relevance_scale

    simplex_data, face_filtrations = args
    simplex, parent_filtration = copy.deepcopy(simplex_data)
    faces_set = set(face_filtrations)

    family = []
    total_tests = 0
    pvalue_info = []

    # Find all viable target candidates for extending the simplex
    viable_targets = []
    for candidate in worker_vertices:
        # Skip if candidate vertex is already in the simplex
        if candidate in simplex:
            continue
        new_simplex = simplex + (candidate,)
        # Pre-check: Ensure all (k-1)-faces of new_simplex exist
        viable = all(
            new_simplex[:j] + new_simplex[j + 1 :] in faces_set
            for j in range(len(new_simplex))
        )
        if viable:
            viable_targets.append(candidate)

    if not viable_targets:
        return family, total_tests, pvalue_info

    # Run causal test for all viable targets at once
    total_tests = len(viable_targets)

    batch_results = pearl_causal_test(
        predictor_neurons=simplex,
        target_neurons=viable_targets,
        model=worker_model,
        probe_data=worker_probe_data,
        dt=worker_dt,
        max_iterations=worker_max_iterations,
        significance_level=worker_significance_level,
        relevance_margin=worker_relevance_margin,
        relevance_scale=worker_relevance_scale,
    )

    # Collect p-values if requested (before correction)
    if worker_collect_pvalues:
        dimension = len(simplex)
        for candidate, (_, _, p_vals) in batch_results.items():
            min_p = np.min(p_vals)
            pvalue_info.append({
                'predictor': simplex,
                'target': candidate,
                'dimension': dimension,
                'min_pvalue': min_p,
                'all_pvalues': p_vals,
            })

    # No decision is taken here. The family a correction applies to is every
    # simplex tested at this dimension, which spans many workers, so the raw
    # trajectories go back to the caller and the reduce happens once the
    # dimension is complete.
    for candidate, (t_vals, _, p_vals) in batch_results.items():
        family.append((simplex + (candidate,), t_vals, p_vals))

    return family, total_tests, pvalue_info


class DirectedSimplicialComplex:
    """
    Represents a directed simplicial complex, where simplices are ordered tuples of vertices.
    The complex is filtered, meaning each simplex is associated with a filtration value.
    This structure is built by analyzing causal relationships in a neural network.
    """

    def __init__(self):
        """
        Initializes an empty directed simplicial complex.
        The complex is stored as a dictionary mapping dimension (int)
        to a list of tuples (simplex, filtration_value).
        A simplex is represented as an ordered tuple of vertices (e.g., (0, 1, 2)).
        """
        self.complex = {}

    def __repr__(self):
        """
        Returns a string representation of the complex that could be used for debugging.
        """
        dims = sorted(self.complex.keys())
        if not dims:
            return "DirectedSimplicialComplex(empty)"
        dimensions = []
        for dim in dims:
            dimensions.append(f"{len(self.complex[dim])} {dim}-simplices")

        return f"DirectedSimplicialComplex({', '.join(dimensions)})"

    def __str__(self):
        """
        Returns a human-readable string summarizing the complex.
        The summary includes total simplex counts, counts per dimension,
        and details of maximal simplices (facets) grouped by filtration value
        and then by their unordered vertex set.
        """
        total_simplices = sum(len(simplices) for simplices in self.complex.values())
        dims = sorted(self.complex.keys())
        if not dims:
            return "Empty directed simplicial complex"
        max_dim = max(dims)
        dim_counts = [f"dim {d}: {len(self.complex[d])}" for d in dims]
        s = (
            f"Directed simplicial complex with {total_simplices} simplices"
            f" (max dimension: {max_dim}):\\n  " + ", ".join(dim_counts)
        )
        s += "\\nMaximal Simplices"
        
        printed_count_per_len = DefaultDict(int)
        ellipsis_message_added_for_len = DefaultDict(bool)

        filtration_dict = {}
        for (
            dim,
            simplex_list,
        ) in self.complex.items():  # Changed dsc.complex to self.complex
            for simplex, filtration in simplex_list:
                filtration_dict.setdefault(filtration, []).append(simplex)
        filtration_keys = sorted(filtration_dict.keys())

        for filtration in filtration_keys:
            lines_for_this_filtration_block = []
            
            from collections import defaultdict # This import is fine here as it's per filtration

            group_dict = defaultdict(list)
            for simplex in filtration_dict[filtration]:
                key = tuple(sorted(simplex))
                group_dict[key].append(simplex)
            
            sorted_items = sorted(
                group_dict.items(), key=lambda x: (len(x[0]), x[0])
            )

            for vertex_tuple, directed_simplices in sorted_items:
                current_len = len(vertex_tuple)
                
                if printed_count_per_len[current_len] < 20:
                    n = len(vertex_tuple)
                    # math.factorial relies on module-level import of math
                    total_possible = math.factorial(n) if n > 1 else 1
                    found = len(directed_simplices)
                    lines_for_this_filtration_block.append(
                        f"\\n  {vertex_tuple}: {found}/{total_possible} directed simplices"
                    )
                    printed_count_per_len[current_len] += 1
                else:
                    if not ellipsis_message_added_for_len[current_len]:
                        lines_for_this_filtration_block.append(
                            f"\\n  (...more simplices of length {current_len} truncated...)"
                        )
                        ellipsis_message_added_for_len[current_len] = True
            
            if lines_for_this_filtration_block:
                s += f"\\nFiltration value {filtration}:"
                s += "".join(lines_for_this_filtration_block)
        return s

    @staticmethod
    def admit_family(family, face_filtrations, alpha, method):
        """Decide one dimension's tests together, and date what survives.

        `family` holds (simplex, ablation_grid, p_values) for every candidate tested
        at this dimension. Each contributes exactly one hypothesis, evaluated at
        full ablation, and the correction is applied across those. The ablation axis
        is then read at the resulting cutoff to place the simplex in the
        filtration; it is not a second family, and it is not corrected over.

        Returns (admitted, n_rejected) with admitted a list of
        (simplex, filtration) ready to append to the complex.
        """
        if not family:
            return [], 0
        finals = np.array([float(p_vals[-1]) for _, _, p_vals in family])
        cutoff = family_cutoff(finals, alpha=alpha, method=method)
        admitted = []
        for simplex, t_vals, p_vals in family:
            index = persistent_crossing_index(np.asarray(p_vals) <= cutoff)
            if index is None:
                continue
            admitted.append((
                simplex,
                DirectedSimplicialComplex.compute_simplex_filtration(
                    float(t_vals[index]), simplex, face_filtrations),
            ))
        return admitted, len(admitted)

    @staticmethod
    def codimension_one_faces(simplex):
        """Return all ordered codimension-one faces of a simplex."""
        return tuple(
            simplex[:idx] + simplex[idx + 1 :]
            for idx in range(len(simplex))
        )

    @staticmethod
    def face_filtration_map(simplex_list):
        """Map each simplex in a dimension list to its filtration value."""
        return {tuple(simplex): float(filtration) for simplex, filtration in simplex_list}

    @classmethod
    def face_filtration_values(cls, simplex, face_filtrations):
        """Return filtration values for all codimension-one faces of simplex."""
        values = []
        missing = []
        for face in cls.codimension_one_faces(simplex):
            if face not in face_filtrations:
                missing.append(face)
            else:
                values.append(float(face_filtrations[face]))
        if missing:
            raise ValueError(f"Missing codimension-one faces for {simplex}: {missing}")
        return values

    @classmethod
    def compute_simplex_filtration(cls, candidate_filtration, simplex, face_filtrations):
        """Return max(candidate birth, all codimension-one face filtrations)."""
        face_values = cls.face_filtration_values(simplex, face_filtrations)
        return max([float(candidate_filtration)] + face_values)

    def validate_filtration_closure(self, tolerance=1e-12):
        """
        Report simplex-face filtration violations.

        Returns a list of dictionaries. Each row records a simplex whose
        filtration is below one of its codimension-one faces.
        """
        violations = []
        for dim in sorted(self.complex):
            if dim == 0:
                continue
            face_filtrations = self.face_filtration_map(self.complex.get(dim - 1, []))
            for simplex, filtration in self.complex.get(dim, []):
                simplex = tuple(simplex)
                filtration = float(filtration)
                for face in self.codimension_one_faces(simplex):
                    face_filtration = face_filtrations.get(face)
                    if face_filtration is None:
                        violations.append({
                            "dimension": dim,
                            "simplex": simplex,
                            "filtration": filtration,
                            "face": face,
                            "face_filtration": None,
                            "gap": None,
                        })
                    elif face_filtration > filtration + tolerance:
                        violations.append({
                            "dimension": dim,
                            "simplex": simplex,
                            "filtration": filtration,
                            "face": face,
                            "face_filtration": face_filtration,
                            "gap": face_filtration - filtration,
                        })
        return violations

    def repair_filtration_closure(self):
        """
        Raise simplex filtrations to satisfy filtration closure.

        The current simplex filtration is treated as the already-computed
        causal-test birth. The repair is dimension-wise, so repaired face
        filtrations propagate to higher-dimensional simplices.
        """
        changed_simplices = 0
        max_increase = 0.0
        changes = []

        for dim in sorted(self.complex):
            if dim == 0:
                continue
            face_filtrations = self.face_filtration_map(self.complex.get(dim - 1, []))
            repaired = []
            for simplex, filtration in self.complex.get(dim, []):
                simplex = tuple(simplex)
                old_filtration = float(filtration)
                new_filtration = self.compute_simplex_filtration(
                    old_filtration,
                    simplex,
                    face_filtrations,
                )
                if new_filtration > old_filtration:
                    changed_simplices += 1
                    max_increase = max(max_increase, new_filtration - old_filtration)
                    changes.append({
                        "dimension": dim,
                        "simplex": simplex,
                        "old_filtration": old_filtration,
                        "new_filtration": new_filtration,
                        "increase": new_filtration - old_filtration,
                    })
                repaired.append((simplex, new_filtration))
            self.complex[dim] = repaired

        return {
            "changed_simplices": changed_simplices,
            "max_increase": max_increase,
            "changes": changes,
        }

    def build(
        self,
        model,
        probe_data,
        max_dimension=2,
        dt=0.01,
        max_iterations=100,
        significance_level=0.01,
        n_workers=None,
        fdr_correction="none",
        collect_pvalues=False,
        relevance_margin=0.0,
        relevance_scale="std",
    ):
        """
        Constructs a filtered directed simplicial complex from a model and probe_data.
        Uses Pearl's counterfactuals and a Wilcoxon signed-rank test.

        The construction is inductive:
          - 0-simplices: each neuron is a vertex with filtration value 0.
          - 1-simplices: add an edge for each pair (A, B) if reducing A significantly affects B.
          - k-simplices, k ≥ 1: Extend a directed (k-1)-simplex (A1,...,Ak) to (A1,...,Ak,B)
            if all its (k-1)-faces exist and reducing {A1,...,Ak} significantly affects B.

        The filtration value for higher simplices is the maximum of all face
        filtrations and the simplex's own last crossing -- the smallest ablation from
        which its test stays significant to full ablation.

        Args:
            model: a trained neural network model.
            probe_data: a tensor of input data to feed to the model.
            max_dimension: maximum simplex dimension to construct.
            dt: integrator step for the counterfactual ablation.
            max_iterations: resolution of the filtration axis, i.e. the number
                of ablation points the trajectory is reported on. Not a horizon the
                ablation has to finish within -- the descent runs until the
                predictors are zero. The filtration value is the ablation, the share
                of the predictor set's baseline activation removed, so values
                run over [0, 1] and 1 is the fully ablated counterfactual.
            significance_level: significance threshold for tests.
            n_workers: number of parallel workers. None or -1 auto-detects as
                all CPUs but four, floored at 1.
            fdr_correction: multiplicity correction over the simplices tested
                at each dimension, one of "none", "bonferroni" (FWER), "bh"
                (FDR under independence or PRDS) or "by" (FDR under arbitrary
                dependence). The family is the set of candidate simplices, each
                contributing a single test at full ablation; the ablation axis is
                not a family and is not corrected over, because reading it from
                the top down is a fixed-sequence step-down procedure that
                controls the error over ablations on its own. Correcting over the
                ablation axis instead would divide the budget by the resolution K,
                penalising a finer trajectory as though it were more hypotheses.
            collect_pvalues: if True, collect all p-values computed during construction
            relevance_margin: if > 0, each causal test uses the interval null
                H0: |theta| <= epsilon (a relevance test, also known as a
                minimum-effect test) instead of the point null H0: theta = 0,
                with epsilon set to this fraction of the target neuron's
                activation scale. 0 reproduces the point-null behaviour.
            relevance_scale: how that per-target scale is measured; see
                `neural_dsc.utils.pearl_causality.relevance_epsilon`.

        Returns:
            self (or tuple of (self, pvalue_info) if collect_pvalues=True)
        """
        # Initialize p-value collection if requested
        if collect_pvalues:
            self.collected_pvalues = []
        null_desc = (
            f"relevance test, epsilon = {relevance_margin} x {relevance_scale}"
            if relevance_margin > 0
            else "point null"
        )
        print(
            "Starting construction of filtered directed simplicial complex"
            f" (max dimension: {max_dimension}, FDR correction: {fdr_correction},"
            f" {null_desc})..."
        )
        start_time = time.time()
        
        # Build list of vertices.
        vertices = list(range(model.internal_neurons))

        # 0-simplices: each vertex as a tuple with filtration value 0.
        self.complex[0] = [((v,), 0) for v in vertices]
        print(f"Created {len(self.complex[0])} 0-simplices")

        if n_workers is None or n_workers == -1:
            # Auto-detect: use every CPU but four, leaving headroom for the OS
            # and for whatever else the machine is doing. The max(1, ...) floor
            # is the small-machine exception — a four-core laptop gets a single
            # worker rather than a negative count, which for these per-sample
            # serial trajectories costs little anyway.
            n_workers = max(1, mp.cpu_count() - 4)

        # Get model state for serialization
        model_state_dict = model.state_dict()
        layer_sizes = model.layer_sizes

        # Ensure probe_data is detached from computation graph for serialization
        if isinstance(probe_data, torch.Tensor):
            probe_data = probe_data.detach().cpu()

        # Initialize the parallel processing context using 'spawn' method for better BLAS compatibility
        ctx = None
        pool = None
        if n_workers > 1:
            print(
                f"Creating worker pool with {n_workers} workers using 'spawn' method..."
            )
            try:
                # Use spawn method for better compatibility with BLAS libraries
                ctx = mp.get_context("spawn")

                # Initialize a pool with worker initialization
                pool = ctx.Pool(
                    processes=n_workers,
                    initializer=worker_initializer,
                    initargs=(
                        model_state_dict,
                        layer_sizes,
                        probe_data,
                        vertices,
                        dt,
                        max_iterations,
                        significance_level,
                        fdr_correction,
                        collect_pvalues,
                        relevance_margin,
                        relevance_scale,
                    ),
                )
                print("Worker pool created successfully")

            except Exception as e:
                print(f"Error initializing worker pool: {e}")
                print("Falling back to sequential processing")
                n_workers = 1
                pool = None

        # Fail before the expensive part rather than at the first reduce. Older
        # names ("benjamini_hochberg", "alpha_spending_*", "complex_wise") all
        # corrected over the ablation axis, so silently mapping one onto a family
        # correction would quietly change what was tested.
        if fdr_correction not in FAMILY_METHODS:
            raise ValueError(
                f"unknown fdr_correction {fdr_correction!r}; expected one of "
                f"{FAMILY_METHODS}. Corrections over trajectory time steps have "
                "been removed: the family is the simplices tested at a dimension."
            )

        # Build higher-dimensional simplices inductively
        for k in range(1, max_dimension + 1):
            print(f"\nConstructing {k}-simplices")
            dim_start_time = time.time()
            self.complex[k] = []
            total_tests = 0
            # Every candidate tested at this dimension, gathered before any
            # of them is decided.
            family = []

            # Determine (k-1)-simplices for face checks and filtration closure
            face_filtrations = self.face_filtration_map(self.complex[k - 1])
            faces_set = set(face_filtrations)

            if n_workers == 1:
                # Sequential processing
                print("  Processing simplices sequentially...")
                from neural_dsc.utils.pearl_causality import (
                    pearl_causal_test,
                )  # Import here to avoid serialization issues

                for i, simplex_data in enumerate(self.complex[k - 1]):
                    try:
                        simplex, parent_filtration = simplex_data
                        batch_total_tests = 0

                        # Find all viable target candidates for extending the simplex
                        viable_targets = []
                        for candidate in vertices:
                            # Skip if candidate vertex is already in the simplex
                            if candidate in simplex:
                                continue
                            new_simplex = simplex + (candidate,)
                            # Pre-check: Ensure all (k-1)-faces of new_simplex exist
                            viable = all(
                                new_simplex[:j] + new_simplex[j + 1 :] in faces_set
                                for j in range(len(new_simplex))
                            )
                            if viable:
                                viable_targets.append(candidate)

                        if viable_targets:
                            # Run causal test for all viable targets at once
                            batch_total_tests = len(viable_targets)
                            batch_results = pearl_causal_test(
                                predictor_neurons=simplex,
                                target_neurons=viable_targets,
                                model=model,
                                probe_data=probe_data,
                                dt=dt,
                                max_iterations=max_iterations,
                                significance_level=significance_level,
                                relevance_margin=relevance_margin,
                                relevance_scale=relevance_scale,
                            )

                            # Collect p-values if requested
                            if collect_pvalues:
                                for candidate, (t_vals, sig_flags, p_vals) in batch_results.items():
                                    # Store minimum p-value and test info
                                    min_p = np.min(p_vals)
                                    self.collected_pvalues.append({
                                        'predictor': simplex,
                                        'target': candidate,
                                        'dimension': k,
                                        'min_pvalue': min_p,
                                        'all_pvalues': p_vals,
                                    })

                            # Decided once the dimension is complete: the
                            # correction's family is every simplex tested at
                            # this dimension, not this one predictor's batch.
                            for candidate, (t_vals, _, p_vals) in batch_results.items():
                                family.append(
                                    (simplex + (candidate,), t_vals, p_vals))

                        total_tests += batch_total_tests

                        # Print progress
                        print(
                            f"  Processed simplex {i+1}/{len(self.complex[k-1])}...",
                            end="\r",
                        )
                    except Exception as exc:
                        print(f"  Error processing simplex: {exc}")
            else:
                print(
                    f"  Processing {len(self.complex[k-1])} {k-1}-simplices in parallel..."
                )

                # Prepare arguments - each simplex with the face filtration map
                simplex_data_list = self.complex[k - 1]
                total_simplices = len(simplex_data_list)

                # Calculate appropriate chunksize if not specified
                chunksize = max(1, min(32, total_simplices // (n_workers * 4)))
                print(f"  chunksize: {chunksize}")

                # Create argument list with individual simplices paired with faces_set
                args_list = [
                    (simplex_data, face_filtrations)
                    for simplex_data in simplex_data_list
                ]

                # Use imap_unordered for better load balancing
                processed_count = 0
                for result in pool.imap_unordered( # type: ignore
                    process_simplex, args_list, chunksize=chunksize
                ):
                    entries, tests, pvalue_info = result
                    family.extend(entries)
                    total_tests += tests
                    if collect_pvalues and pvalue_info:
                        self.collected_pvalues.extend(pvalue_info)
                    # Update progress counter
                    processed_count += 1
                    print(
                        f"  Processed {processed_count}/{total_simplices} simplices...",
                        end="\r",
                    )
                # Make sure we have a final newline after progress reporting
                print("")

            admitted, successful_tests = self.admit_family(
                family, face_filtrations, significance_level, fdr_correction)
            self.complex[k].extend(admitted)

            dim_elapsed = time.time() - dim_start_time
            print(
                f"  Created {len(self.complex[k])} {k}-simplices"
                f" (from {total_tests} tests, {successful_tests} successful)"
            )
            print(f"  Time for dimension {k}: {dim_elapsed:.2f} seconds")

            # Stop if no new simplices were added at this dimension.
            if not self.complex[k]:
                print(f"No {k}-simplices found, stopping construction.")
                break

        # Clean up the worker pool
        if pool is not None:
            pool.close()
            pool.join()

        total_elapsed = time.time() - start_time
        total_simplices = sum(len(simplices) for simplices in self.complex.values())
        print(
            "\nConstruction complete."
            f" Built {total_simplices} total simplices in {total_elapsed:.2f} seconds"
        )

        # Return p-values if collected
        if collect_pvalues:
            print(f"  Collected {len(self.collected_pvalues)} p-value sets")
            return self, self.collected_pvalues
        return self

    def save(self, filepath, format="text"):
        """
        Saves the filtered directed simplicial complex to a file.

        Args:
            filepath: path to save the file.
            format: either "text" or "pickle".

        Returns:
            None
        """
        # Create directory if it doesn't exist.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self.complex, f)
        elif format == "text":
            with open(filepath, "w") as f:
                f.write("# Filtered Directed Simplicial Complex\n")
                total_simplices = sum(
                    len(simplices) for simplices in self.complex.values()
                )
                f.write(f"Total simplices: {total_simplices}\n")
                for dim in sorted(self.complex.keys()):
                    f.write(f"Dimension {dim}: {len(self.complex[dim])} simplices\n")
                for dim in sorted(self.complex.keys()):
                    f.write(f"## Dimension {dim}\n\n")
                    sorted_simplices = sorted(self.complex[dim], key=lambda x: x[1])
                    for simplex, filtration in sorted_simplices:
                        # Format all dimensions as tuples with commas
                        simplex_str = "(" + ",".join(map(str, simplex)) + ")"
                        f.write(f"{simplex_str}, {filtration:.6f}\n")
                    f.write("\n")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'text' or 'pickle'.")

        print(f"Simplicial complex saved to {filepath}")

    def load(self, filepath, format="text"):
        """
        Loads a filtered directed simplicial complex from a file.

        Args:
            filepath: path to the file.
            format: either "text" or "pickle".

        Returns:
            None
        """
        if format == "pickle":
            with open(filepath, "rb") as f:
                self.complex = pickle.load(f)
        elif format == "text":
            self.complex = {}  # Clear existing complex
            simplicial_complex = defaultdict(list)
            current_dim = None

            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for dimension headers
                    dim_match = re.match(r"## Dimension (\d+)", line)
                    if dim_match:
                        current_dim = int(dim_match.group(1))
                        continue

                    # Check for simplex entries
                    if current_dim is not None:
                        # Parse lines like: "(0,), 0.000000" or "(0,1), 1.000000"
                        simplex_match = re.match(r"\(([0-9,]+)\),\s+([0-9.-]+)", line)
                        if simplex_match:
                            vertices_str = simplex_match.group(1)
                            filtration_str = simplex_match.group(2)
                            # Parse vertices string into a tuple of integers
                            vertex_strings = [v for v in vertices_str.split(",") if v]
                            simplex = tuple(int(v) for v in vertex_strings)

                            filtration = float(filtration_str)
                            simplicial_complex[current_dim].append(
                                (simplex, filtration)
                            )

            # Copy data to instance.complex
            for dim, simplices in simplicial_complex.items():
                self.complex[dim] = simplices

            # Log summary of what was loaded
            total_simplices = sum(len(simplices) for simplices in self.complex.values())
            print(f"Loaded {total_simplices} total simplices:")
            for dim in sorted(self.complex.keys()):
                print(f"  Dimension {dim}: {len(self.complex[dim])} simplices")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'text' or 'pickle'.")

        print(f"Simplicial complex loaded from {filepath}")


def _index_pvalues_by_predictor(collected_pvalues):
    pvalue_index = {}
    for pv in collected_pvalues:
        key = (pv["dimension"], pv["predictor"])
        if key not in pvalue_index:
            pvalue_index[key] = {}
        pvalue_index[key][pv["target"]] = pv["all_pvalues"]
    return pvalue_index


def build_dsc_from_cached_pvalues(
    collected_pvalues,
    model,
    max_dimension,
    dt,
    significance_level,
    fdr_method,
    n_steps=None,
    verbose=True,
    return_families=False,
):
    """
    Re-derive a complete complex from p-values cached by an earlier `build`.

    The p-values a causal test produces do not depend on the significance level
    or on the multiple-comparison correction -- only on the probe set, the
    trajectory, and the relevance margin. So one expensive construction can be
    re-scored under many (alpha, correction) pairs for almost nothing, which is
    what makes a joint sweep over those two axes affordable.

    `n_steps` extends the same trick to the ablation grid. A trajectory always
    reaches full ablation, so `max_iterations` sets only how finely [0, 1] is
    sampled, and subsampling the cached sequence -- endpoint included -- is
    exactly what a shorter run would have produced. Membership is unaffected,
    because it is decided at the endpoint every grid contains; what moves is how
    precisely the last crossing, and hence the filtration value, can be located.

    **The cache must come from the most permissive setting.** Construction is
    inductive: a k-simplex is only tested when all its (k-1)-faces exist, so a
    more permissive setting yields more surviving simplices and demands p-values
    that a stricter run never computed. Collect with `fdr_correction="none"` at
    the largest alpha in the grid, whose complex is a superset of every corrected
    one. A missing entry raises rather than silently truncating the complex.

    Args:
        collected_pvalues: the `pvalue_info` list returned by
            `DirectedSimplicialComplex.build(..., collect_pvalues=True)`.
        model: the model the p-values were computed for (for `internal_neurons`).
        max_dimension: maximum simplex dimension to reconstruct.
        dt: the trajectory step size the p-values were computed at.
        significance_level: alpha to re-score at.
        fdr_method: correction over the simplices tested at each dimension, per
            `family_cutoff`.
        n_steps: subsample the ablation axis to `n_steps` points including both
            endpoints, equivalent to rebuilding at `max_iterations = n_steps - 1`.
            None keeps the whole cache. Must not exceed the cached length --
            there is no way to refine a grid that was never integrated.
        verbose: print per-dimension progress.
        return_families: also return, per dimension, the candidates the
            construction actually evaluated there and the threshold it applied.
            Construction is inductive, so which candidates exist at dimension k
            depends on what survived at k-1: a stricter correction can drop a
            face and with it every simplex that face would have supported, which
            is then never tested at all. Anything reporting on the construction
            has to read the families from here rather than regrouping the cache
            by dimension, or it will describe candidates the run never posed.

    Returns:
        DirectedSimplicialComplex, or (complex, families) when
        `return_families`, where families maps dimension -> {"family":
        [(simplex, t_values, p_values)], "cutoff": float}.

    Raises:
        ValueError: if the cache lacks a p-value the reconstruction needs, or if
            `n_steps` exceeds the cached trajectory length.
    """
    # Imported here rather than at module level, matching the convention the
    # rest of this file already follows for this symbol.

    sc = DirectedSimplicialComplex()
    vertices = list(range(model.internal_neurons))
    sc.complex[0] = [((v,), 0) for v in vertices]

    pvalue_index = _index_pvalues_by_predictor(collected_pvalues)
    families = {}

    for k in range(1, max_dimension + 1):
        if verbose:
            print(f"\nConstructing {k}-simplices from cached p-values ({fdr_method})")
        dim_start_time = time.time()
        sc.complex[k] = []
        total_tests = 0
        family = []

        face_filtrations = sc.face_filtration_map(sc.complex[k - 1])
        faces_set = set(face_filtrations)

        for simplex, parent_filtration in sc.complex[k - 1]:
            viable_targets = []
            for candidate in vertices:
                if candidate in simplex:
                    continue
                new_simplex = simplex + (candidate,)
                viable = all(
                    new_simplex[:j] + new_simplex[j + 1 :] in faces_set
                    for j in range(len(new_simplex))
                )
                if viable:
                    viable_targets.append(candidate)

            if not viable_targets:
                continue

            key = (k, simplex)
            target_map = pvalue_index.get(key, {})
            missing_targets = [t for t in viable_targets if t not in target_map]
            if missing_targets:
                raise ValueError(
                    "Missing cached p-values for "
                    f"dimension {k}, predictor {simplex}, targets {missing_targets}"
                )

            for candidate in viable_targets:
                pvals = np.asarray(target_map[candidate])
                if n_steps is not None:
                    if n_steps > len(pvals):
                        raise ValueError(
                            f"n_steps={n_steps} exceeds the cached trajectory "
                            f"length {len(pvals)} for dimension {k}, predictor "
                            f"{simplex}, target {candidate}"
                        )
                    # Subsample the ablation axis rather than truncate it. Under the
                    # ablation filtration the last point is full ablation, which is
                    # where the decision is taken, so dropping the tail would
                    # change the hypothesis rather than its resolution.
                    take = np.linspace(0, len(pvals) - 1, n_steps)
                    pvals = pvals[np.rint(take).astype(int)]
                t_values = np.linspace(0.0, 1.0, len(pvals), dtype=np.float32)
                family.append((simplex + (candidate,), t_values, pvals))

            total_tests += len(viable_targets)

        if return_families:
            finals = [float(p_vals[-1]) for _, _, p_vals in family]
            families[k] = {
                "family": family,
                "cutoff": family_cutoff(
                    finals, alpha=significance_level, method=fdr_method
                ),
            }
        admitted, successful_tests = sc.admit_family(
            family, face_filtrations, significance_level, fdr_method)
        sc.complex[k].extend(admitted)

        dim_elapsed = time.time() - dim_start_time
        if verbose:
            print(
                f"  Created {len(sc.complex[k])} {k}-simplices"
                f" (from {total_tests} tests, {successful_tests} successful)"
            )
            print(f"  Time for dimension {k}: {dim_elapsed:.2f} seconds")

        if not sc.complex[k]:
            if verbose:
                print(f"No {k}-simplices found, stopping construction.")
            break

    return (sc, families) if return_families else sc
