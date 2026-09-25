"""Validation, one-seed timing, and restart-safe sweep entry point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backbones import build_variants, deterministic_queries, load_config
from .geometry import validate_geometry
from .timing import PreparedProjection, time_compilation, time_projection


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hashes():
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    files = [
        experiment / name for name in ("config.json", "backbones.py", "geometry.py", "timing.py", "run.py")
    ] + [repository / "src" / "trio_paper" / name for name in ("models.py", "projection_2d.py", "timing.py")]
    return {str(path.relative_to(repository)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _preflight(config, seed):
    variants, spatial = build_variants(config, seed)
    identity = validate_geometry(
        variants, spatial["desired_radii"], config["target"],
        tolerance=config["geometry_radius_tolerance"], grid_resolution=config["membership_grid_resolution"],
    )
    return variants, spatial, identity


def run_seed(config, seed, output, *, compile_repetitions=None, projection_repetitions=None, query_count=None):
    variants, _, identity = _preflight(config, seed)
    queries = deterministic_queries(config, seed)
    if query_count is not None:
        queries = queries[:int(query_count)]
    compile_repetitions = int(compile_repetitions or config["compile_timing_repetitions"])
    projection_repetitions = int(projection_repetitions or config["projection_timing_repetitions"])
    target = float(config["target"])
    compilation_rows, prepared = [], {}
    for key, variant in variants.items():
        rows = time_compilation(variant.frozen, target, compile_repetitions)
        compilation_rows.extend({"backbone_key": key, "backbone": variant.label, **row} for row in rows)
        prepared[key] = PreparedProjection(variant.frozen, target, queries)
        if prepared[key].cached_eigendecompositions != int(config["experts"]):
            raise AssertionError("Every eigendecomposition must be cached before timing")
        np.testing.assert_array_equal(prepared[key].active, np.arange(int(config["experts"])))
    timing_rows, projection_summaries = time_projection(
        prepared, projection_repetitions, config["projection_warmups"],
    )
    summary = []
    for key, variant in variants.items():
        compile_values = np.asarray([
            row["elapsed_seconds"] for row in compilation_rows if row["backbone_key"] == key
        ], dtype=np.float64)
        batch_mean = float(projection_summaries[key]["mean_seconds"])
        active = int(len(prepared[key].active))
        summary.append({
            "seed": int(seed),
            "backbone_key": key,
            "backbone": variant.label,
            "exact_parameter_count": variant.parameter_count,
            "experts": int(config["experts"]),
            "active_ellipsoids": active,
            "compile_mean_seconds": float(compile_values.mean()),
            "projection_mean_seconds_per_query": batch_mean / len(queries),
            "projection_seconds_per_active_ellipse": batch_mean / (len(queries) * active),
            "cached_eigendecompositions": prepared[key].cached_eigendecompositions,
            "eigendecompositions_inside_timer": 0,
            "radial_operations_inside_projection_timer": 0,
        })
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")},
        "source_hashes": _hashes(),
    }
    output = Path(output)
    _write_json(output / "resolved_config.json", {**config, "seed": int(seed), "query_seed": int(seed) + 1})
    _write_json(output / "reproducibility_metadata.json", metadata)
    _write_json(output / "geometry_identity.json", identity)
    _write_csv(output / "compilation_timings.csv", compilation_rows)
    _write_csv(output / "projection_timing_repetitions.csv", timing_rows)
    _write_csv(output / "summary.csv", summary)
    _write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Controlled TRIO projection-cost ablation")
    parser.add_argument("command", choices=("validate", "seed", "sweep"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", default="results/projection_cost_ablation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        variants, _, identity = _preflight(config, args.seed)
        print(json.dumps({
            "seed": args.seed,
            "parameter_counts": {key: value.parameter_count for key, value in variants.items()},
            "geometry_identity": identity,
        }, indent=2))
        return
    root = Path(args.output)
    seeds = [args.seed] if args.command == "seed" else list(config["seeds"])
    status_path = root / "sweep_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"seeds": {}}
    for seed in seeds:
        seed_output = root / f"seed_{seed}"
        if (seed_output / "summary.csv").exists():
            status["seeds"][str(seed)] = "complete"
            _write_json(status_path, status)
            continue
        seed_output.mkdir(parents=True, exist_ok=True)
        if args.command == "seed":
            run_seed(config, seed, seed_output)
        else:
            command = [sys.executable, "-m", "experiments.projection_cost_ablation.run", "seed", "--config", args.config, "--seed", str(seed), "--output", str(root)]
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                status["seeds"][str(seed)] = "failed"
                _write_json(status_path, status)
                raise RuntimeError(f"Seed {seed} failed with exit code {completed.returncode}")
        status["seeds"][str(seed)] = "complete"
        _write_json(status_path, status)


if __name__ == "__main__":
    main()
