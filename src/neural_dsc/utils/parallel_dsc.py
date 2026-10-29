"""Outer-level scheduling for independent DSC constructions.

The quadrant producers build many small complexes at a fixed non-zero relevance
margin. Parallelising candidate tests inside any one of them repeatedly pays the
process-pool startup cost for only a small frontier. This module instead keeps
each construction serial and dispatches complete constructions through one
shared spawn pool.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from neural_dsc.directed_simplicial_complex import (
    DirectedSimplicialComplex,
    pin_worker_threads,
)


@dataclass(frozen=True)
class DSCBuildSpec:
    """Everything one outer worker needs to build a single complex."""

    label: str
    model: torch.nn.Module
    probe_data: torch.Tensor
    output_path: Path
    max_dimension: int
    dt: float
    max_iterations: int
    fdr_correction: str
    significance_level: float
    relevance_margin: float
    relevance_scale: str
    # Processes this one build may use internally, for its loop over predictor
    # sets. 1 keeps the build serial, which is right whenever there are more
    # complexes to construct than processes to construct them with -- the
    # quadrant runs have fourteen tiny contexts and nothing to gain from
    # splitting any of them. It is wrong for MNIST, where a single context is
    # hours of work and one dimension holds 240 predictor sets.
    n_workers: int = 1


def resolve_outer_workers(configured_workers: int | None) -> int:
    """Resolve the pool width, leaving four CPUs free in automatic mode."""
    if configured_workers is None or configured_workers == -1:
        return max(1, mp.cpu_count() - 4)
    if configured_workers < 1:
        raise ValueError(
            f"outer worker count must be positive, None, or -1; "
            f"got {configured_workers}"
        )
    return configured_workers


def _build_and_save(spec: DSCBuildSpec) -> tuple[str, float, int]:
    """Build one complex and atomically publish its checkpoint."""
    pin_worker_threads()
    print(f"\n--- {spec.label} ---", flush=True)
    started = time.time()

    sc = DirectedSimplicialComplex()
    sc.build(
        model=spec.model,
        probe_data=spec.probe_data,
        max_dimension=spec.max_dimension,
        dt=spec.dt,
        max_iterations=spec.max_iterations,
        n_workers=spec.n_workers,
        fdr_correction=spec.fdr_correction,
        significance_level=spec.significance_level,
        relevance_margin=spec.relevance_margin,
        relevance_scale=spec.relevance_scale,
    )

    output_path = Path(spec.output_path)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    sc.save(str(temporary))
    temporary.replace(output_path)

    total = sum(len(simplices) for simplices in sc.complex.values())
    return spec.label, time.time() - started, total


def _load_complex(path: Path) -> DirectedSimplicialComplex:
    sc = DirectedSimplicialComplex()
    sc.load(str(path))
    return sc


def build_or_load_many(
    specs: Sequence[DSCBuildSpec],
    configured_workers: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[DirectedSimplicialComplex]:
    """Build every missing spec in one outer pool, then load them in input order."""
    if not specs:
        return []

    labels = [spec.label for spec in specs]
    paths = [Path(spec.output_path) for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("DSC build labels must be unique")
    if len(set(paths)) != len(paths):
        raise ValueError("DSC output paths must be unique")

    pending = []
    for spec in specs:
        if Path(spec.output_path).exists():
            print(f"  {spec.label}: loaded cached DSC (no construction)")
        else:
            pending.append(spec)

    n_workers = resolve_outer_workers(configured_workers)
    if pending:
        lanes = min(n_workers, len(pending))
        inner = max(1, max(spec.n_workers for spec in pending))
        print(
            f"Dispatching {len(pending)} pending DSC constructions over "
            f"{lanes} lanes"
            + (
                f" x up to {inner} processes each ({lanes * inner} total)."
                if inner > 1
                else "; each build is serial."
            )
        )

        if lanes == 1:
            for index, spec in enumerate(pending, 1):
                label, elapsed, total = _build_and_save(spec)
                print(
                    f"  completed {label}: {total} simplices in {elapsed:.2f}s"
                )
                if progress is not None:
                    progress(index, len(pending), "constructions")
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=lanes,
                mp_context=ctx,
                initializer=pin_worker_threads,
            ) as pool:
                futures = {
                    pool.submit(_build_and_save, spec): spec.label
                    for spec in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    label = futures[future]
                    try:
                        _, elapsed, total = future.result()
                    except Exception as error:
                        raise RuntimeError(f"failed to build {label}") from error
                    print(
                        f"  completed {label}: {total} simplices in {elapsed:.2f}s"
                    )
                    if progress is not None:
                        progress(index, len(pending), "constructions")
    else:
        print("All DSC constructions were found in the cache.")

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"DSC construction left missing outputs: {missing}")
    return [_load_complex(path) for path in paths]
