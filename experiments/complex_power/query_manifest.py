"""Frozen equal-mass, nontrivial projection suites for the z^d DGP.

This module is deliberately model-free: it derives target thresholds and
queries only from the canonical semantic convention and the analytic true-DGP
projector.  Learned checkpoints are neither loaded nor inspected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .ground_truth_projection import (
    DEFAULT_TOLERANCE,
    REFERENCE_IMPLEMENTATION_VERSION,
    alpha_and_offset,
    project_stored_threshold,
    raw_field,
)


SUPPORTED_POWERS = (4, 6, 8, 10, 12)
TARGET_MASSES = (0.2, 0.5, 0.8)
QUERY_COUNT_PER_TARGET = 20
QUERY_RNG_SEED = 20260911
MANIFEST_VERSION = 1


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_feasible_mass(power: int, threshold: float, *, epsabs: float = 1e-13) -> float:
    """Area fraction of ``Re(exp(-i alpha) z^power) <= threshold`` on the disk.

    Rotation does not affect uniform-disk area.  The positive-threshold branch
    is the supplied analytic formula; negative thresholds follow by symmetry.
    """
    if power not in SUPPORTED_POWERS:
        raise ValueError(f"Unsupported power {power}")
    if threshold <= -1.0:
        return 0.0
    if threshold >= 1.0:
        return 1.0
    if threshold < 0.0:
        return 1.0 - raw_feasible_mass(power, -threshold, epsabs=epsabs)
    if threshold == 0.0:
        return 0.5
    beta = math.acos(threshold)
    integral, error = quad(
        lambda angle: (threshold / math.cos(angle)) ** (2.0 / power),
        0.0,
        beta,
        epsabs=epsabs,
        epsrel=1e-13,
        limit=200,
    )
    if error > 1e-10:
        raise RuntimeError(f"Unreliable mass quadrature: estimated error {error}")
    return 1.0 - beta / math.pi + integral / math.pi


def solve_positive_equal_mass_threshold(power: int, mass: float = 0.8) -> float:
    if not 0.5 < mass < 1.0:
        raise ValueError("This helper solves only positive thresholds for mass in (0.5, 1)")
    return float(brentq(lambda value: raw_feasible_mass(power, value) - mass, 0.0, 1.0, xtol=1e-14, rtol=1e-14))


def thresholds_for_power(power: int, semantic_target: dict) -> list[dict[str, float | int]]:
    """Return the frozen 20/50/80% threshold table in raw and stored forms."""
    _, offset = alpha_and_offset(semantic_target)
    positive = solve_positive_equal_mass_threshold(power)
    raw_thresholds = (-positive, 0.0, positive)
    rows = []
    for target_index, (mass, raw_threshold) in enumerate(zip(TARGET_MASSES, raw_thresholds)):
        verified = raw_feasible_mass(power, raw_threshold)
        rows.append({
            "target_index": target_index,
            "target_mass": mass,
            "raw_threshold": raw_threshold,
            # stored_label = raw_field - offset; hence stored threshold is
            # raw_threshold - offset, the inverse of the canonical conversion.
            "stored_threshold": raw_threshold - offset,
            "verified_true_mass": verified,
            "mass_error": verified - mass,
        })
    return rows


def _query_generator(power: int) -> np.random.Generator:
    # Each power has an independent deterministic stream, so adding a power
    # cannot perturb any existing frozen suite.
    return np.random.default_rng(QUERY_RNG_SEED + 1_000_003 * power)


def build_power_suite(power: int, semantic_target: dict) -> dict[str, Any]:
    alpha, offset = alpha_and_offset(semantic_target)
    threshold_rows = thresholds_for_power(power, semantic_target)
    rng = _query_generator(power)
    queries: list[dict[str, Any]] = []
    query_id = 0
    for threshold_row in threshold_rows:
        accepted = 0
        attempts = 0
        while accepted < QUERY_COUNT_PER_TARGET:
            attempts += 1
            if attempts > 1_000_000:
                raise RuntimeError(f"Failed to generate nontrivial queries for z^{power}, target {threshold_row['target_index']}")
            radius = math.sqrt(float(rng.random()))
            theta = 2.0 * math.pi * float(rng.random())
            point = np.array((radius * math.cos(theta), radius * math.sin(theta)), dtype=float)
            if raw_field(point, power, alpha) <= float(threshold_row["raw_threshold"]):
                continue
            reference = project_stored_threshold(point, power, float(threshold_row["stored_threshold"]), semantic_target)
            if reference.J_star is None or reference.J_star <= 1e-12:
                # Numerical-degeneracy guard only; this does not impose a
                # substantive boundary-distance margin.
                continue
            z_star = np.asarray(reference.z_star, dtype=float)
            residual = raw_field(z_star, power, alpha) - float(threshold_row["raw_threshold"])
            queries.append({
                "query_id": query_id,
                **threshold_row,
                "x0": point.tolist(),
                "z_star": list(reference.z_star),
                "projection_distance_star": reference.projection_distance_star,
                "J_star": reference.J_star,
                "reference_status": reference.reference_status,
                "branch": reference.branch,
                "branch_parameter": reference.branch_parameter,
                "dense_crosscheck_gap": reference.dense_crosscheck_gap,
                "reference_feasibility_residual": residual,
            })
            query_id += 1
            accepted += 1
    suite_hash = _json_hash(queries)
    return {
        "power": power,
        "query_rng_seed": QUERY_RNG_SEED + 1_000_003 * power,
        "alpha_radians": alpha,
        "semantic_label_offset": offset,
        "thresholds": threshold_rows,
        "queries": queries,
        "query_suite_hash": suite_hash,
    }


def build_manifest(canonical_config_path: Path) -> dict[str, Any]:
    config = json.loads(canonical_config_path.read_text())
    semantic_target = config["semantic_target"]
    here = Path(__file__).resolve().parent
    suites = {str(power): build_power_suite(power, semantic_target) for power in SUPPORTED_POWERS}
    return {
        "kind": "projection_mass_matched_nontrivial",
        "manifest_version": MANIFEST_VERSION,
        "reference_implementation_version": REFERENCE_IMPLEMENTATION_VERSION,
        "query_rng_seed_rule": "QUERY_RNG_SEED + 1000003 * power",
        "query_rng_seed_base": QUERY_RNG_SEED,
        "semantic_target": semantic_target,
        "semantic_label_convention": {
            "stored_label": "Re(exp(-i alpha) z^d) - rotated_target_center_offset",
            "stored_threshold_from_raw": "g_stored = g_raw - rotated_target_center_offset",
            "alpha_and_offset_source": "ground_truth_projection.alpha_and_offset",
        },
        "source_hashes": {
            "canonical_config": _file_hash(canonical_config_path),
            "ground_truth_projection": _file_hash(here / "ground_truth_projection.py"),
            "benchmark_builder": _file_hash(Path(__file__)),
        },
        "suites": suites,
    }


def write_frozen_manifest(output: Path, canonical_config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen benchmark manifest: {output}")
    manifest = build_manifest(canonical_config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_power_suite(manifest_path: Path, power: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load a frozen suite for learned evaluation; no resampling/reprojection."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "projection_mass_matched_nontrivial":
        raise ValueError("Not a mass-matched nontrivial projection manifest")
    suite = manifest["suites"].get(str(power))
    if suite is None:
        raise KeyError(f"Manifest has no z^{power} suite")
    queries = []
    references = []
    for row in suite["queries"]:
        if float(row["J_star"]) <= 1e-12:
            raise AssertionError("Frozen suite contains a trivial reference")
        queries.append({
            "query_id": int(row["query_id"]), "target_index": int(row["target_index"]),
            "target_mass": float(row["target_mass"]), "threshold": float(row["stored_threshold"]),
            "raw_threshold": float(row["raw_threshold"]), "x0": np.asarray(row["x0"], dtype=float),
        })
        references.append(dict(row))
    if len(queries) != 60 or [sum(q["target_index"] == index for q in queries) for index in range(3)] != [20, 20, 20]:
        raise AssertionError("Frozen suite must contain exactly 20 queries per target")
    return queries, references, str(suite["query_suite_hash"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen equal-mass nontrivial z^d projection benchmark.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = write_frozen_manifest(args.output, args.config, overwrite=args.overwrite)
    for power in SUPPORTED_POWERS:
        suite = manifest["suites"][str(power)]
        print(json.dumps({
            "power": power,
            "thresholds": suite["thresholds"],
            "query_count": len(suite["queries"]),
            "query_suite_hash": suite["query_suite_hash"],
            "max_dense_crosscheck_gap": max(row["dense_crosscheck_gap"] for row in suite["queries"]),
        }))


if __name__ == "__main__":
    main()
