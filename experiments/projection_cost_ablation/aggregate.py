"""Aggregate seed-level timings and generate the ablation table and figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .backbones import load_config


METRICS = (
    "compile_mean_seconds",
    "projection_mean_seconds_per_query",
    "projection_seconds_per_active_ellipse",
)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(input_root, seeds):
    raw = []
    for seed in seeds:
        path = Path(input_root) / f"seed_{seed}" / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed seed result: {path}")
        raw.extend(csv.DictReader(path.open(encoding="utf-8")))
    grouped = {}
    for row in raw:
        grouped.setdefault(row["backbone_key"], []).append(row)
    rows = []
    for key, members in grouped.items():
        first = members[0]
        result = {
            "backbone_key": key,
            "backbone": first["backbone"],
            "exact_parameter_count": int(first["exact_parameter_count"]),
            "experts": int(first["experts"]),
            "active_ellipsoids": int(first["active_ellipsoids"]),
            "completed_seeds": len(members),
        }
        for metric in METRICS:
            values = np.asarray([float(member[metric]) for member in members], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        rows.append(result)
    return rows


def write_table(path, rows):
    lines = [
        "| Backbone | Exact params | Q | Active ellipsoids | Compile time (ms) | Projection/query (ms) | Time/active ellipse (us) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['exact_parameter_count']:,} | {row['experts']} | {row['active_ellipsoids']} | "
            f"{1e3*row['compile_mean_seconds_mean']:.3f} ± {1e3*row['compile_mean_seconds_sample_sd']:.3f} | "
            f"{1e3*row['projection_mean_seconds_per_query_mean']:.3f} ± {1e3*row['projection_mean_seconds_per_query_sample_sd']:.3f} | "
            f"{1e6*row['projection_seconds_per_active_ellipse_mean']:.2f} ± {1e6*row['projection_seconds_per_active_ellipse_sample_sd']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path, rows):
    parameters = np.asarray([row["exact_parameter_count"] for row in rows])
    compile_ms = 1e3 * np.asarray([row["compile_mean_seconds_mean"] for row in rows])
    compile_sd = 1e3 * np.asarray([row["compile_mean_seconds_sample_sd"] for row in rows])
    projection_ms = 1e3 * np.asarray([row["projection_mean_seconds_per_query_mean"] for row in rows])
    projection_sd = 1e3 * np.asarray([row["projection_mean_seconds_per_query_sample_sd"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].errorbar(parameters, compile_ms, yerr=compile_sd, marker="o", capsize=3, color="#8b1a1a")
    axes[1].errorbar(parameters, projection_ms, yerr=projection_sd, marker="o", capsize=3, color="#17365d")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Trainable parameters")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Compilation time (ms)")
    axes[1].set_ylabel("Projection time/query (ms)")
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Aggregate controlled projection-cost timings")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--input", default="results/projection_cost_ablation")
    parser.add_argument("--output", default="results/projection_cost_ablation/aggregate")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = aggregate(args.input, config["seeds"])
    _write_csv(output / "summary.csv", rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_table(output / "table.md", rows)
    plot(output / "projection_cost_ablation.pdf", rows)
    plot(output / "projection_cost_ablation.png", rows)


if __name__ == "__main__":
    main()
