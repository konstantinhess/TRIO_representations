"""Shared alternating batch-timing protocol for prepared projection runners."""
from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

PROTOCOL_ID = "trio_projection_timing_v1"


def _summary(rows):
    elapsed = np.asarray([row["elapsed_seconds"] for row in rows], dtype=np.float64)
    return {
        "repetitions": int(len(rows)),
        "mean_seconds": float(elapsed.mean()),
        "median_seconds": float(np.median(elapsed)),
        "iqr_seconds": float(np.quantile(elapsed, .75) - np.quantile(elapsed, .25)),
        "min_seconds": float(elapsed.min()),
        "max_seconds": float(elapsed.max()),
    }


def run_alternating_batch_timing(
    runners: Mapping[str, Callable[[], Mapping[str, Any]]], *, repetitions: int = 20, warmups: int = 1,
):
    """Warm all prepared runners, then alternate their order across repetitions."""
    if not runners:
        raise ValueError("At least one prepared runner is required")
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    if not gc.isenabled():
        raise RuntimeError("The canonical protocol requires normal Python GC")
    names = tuple(runners)
    for _ in range(warmups):
        for runner in runners.values():
            runner()
    rows = []
    for repetition in range(repetitions):
        order = names if repetition % 2 == 0 else tuple(reversed(names))
        for sequence, name in enumerate(order):
            payload = dict(runners[name]())
            if "elapsed_seconds" not in payload:
                raise ValueError(f"Runner {name!r} omitted elapsed_seconds")
            rows.append({
                "protocol_id": PROTOCOL_ID,
                "backbone_key": name,
                "repetition": repetition,
                "sequence_in_repetition": sequence,
                "gc_enabled": True,
                **payload,
            })
    return rows, {name: _summary([row for row in rows if row["backbone_key"] == name]) for name in names}
