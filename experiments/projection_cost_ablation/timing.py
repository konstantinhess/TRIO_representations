"""Separate once-per-target compilation and prepared projection timing paths."""
from __future__ import annotations

import time

import numpy as np

from trio_paper.projection_2d import CachedEllipsoidProjector
from trio_paper.timing import run_alternating_batch_timing


class PreparedProjection:
    """Compiled certificate, active list, query array, and EVD cache."""
    def __init__(self, frozen, target, queries):
        self.certificate = frozen.compile(float(target))
        self.active = np.flatnonzero(self.certificate.active).astype(np.int64)
        self.queries = np.asarray(queries, dtype=np.float64)
        self.projector = CachedEllipsoidProjector(frozen)
        self.cached_eigendecompositions = int(len(self.projector.eigenvalues))

    def timed_batch(self):
        started = time.perf_counter()
        for query in self.queries:
            self.projector.project_compiled(self.certificate, query, self.active)
        return {"elapsed_seconds": time.perf_counter() - started, "query_count": int(len(self.queries))}


def time_compilation(frozen, target, repetitions):
    rows = []
    for repetition in range(int(repetitions)):
        started = time.perf_counter()
        certificate = frozen.compile(float(target))
        rows.append({
            "repetition": repetition,
            "elapsed_seconds": time.perf_counter() - started,
            "active_ellipsoids": int(certificate.active.sum()),
        })
    return rows


def time_projection(prepared, repetitions, warmups=1):
    runners = {key: value.timed_batch for key, value in prepared.items()}
    return run_alternating_batch_timing(runners, repetitions=int(repetitions), warmups=int(warmups))
