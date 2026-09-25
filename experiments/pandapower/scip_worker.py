"""Torch-free exact SCIP spatial-B&B worker for a frozen Tanh MLP hosting NLP."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from pyscipopt import Model, quicksum


def evaluate(layers: list[dict], point: np.ndarray) -> float:
    value = np.asarray(point, dtype=float)
    for index, layer in enumerate(layers):
        value = np.asarray(layer["weight"], dtype=float) @ value + np.asarray(layer["bias"], dtype=float)
        if index < len(layers) - 1:
            value = np.tanh(value)
    return float(value[0])


def main(source: str, destination: str) -> None:
    request = json.loads(open(source, encoding="utf-8").read())
    dimension, model = int(request["input_dim"]), Model("pandapower_tanh_global_hosting")
    lower = np.broadcast_to(np.asarray(request.get("lower", 0.0), dtype=float), (dimension,)).copy()
    upper = np.broadcast_to(np.asarray(request.get("upper", 1.0), dtype=float), (dimension,)).copy()
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("box bounds must be finite with lower <= upper")
    weights = np.asarray(request.get("objective_weights", np.ones(dimension)), dtype=float)
    if weights.shape != (dimension,) or not np.all(np.isfinite(weights)):
        raise ValueError("objective_weights must be a finite vector of input dimension")
    build_started = time.perf_counter()
    model.hideOutput(True)
    model.setParam("randomization/permutationseed", 0)
    model.setParam("randomization/randomseedshift", 0)
    model.setParam("limits/time", float(request["timeout_seconds"]))
    x = [model.addVar(lb=float(lower[i]), ub=float(upper[i]), name=f"u_{i}") for i in range(dimension)]
    previous = x
    for layer_index, layer in enumerate(request["layers"]):
        weight, bias = np.asarray(layer["weight"], dtype=float), np.asarray(layer["bias"], dtype=float)
        pre = [model.addVar(lb=None, ub=None, name=f"pre_{layer_index}_{j}") for j in range(len(bias))]
        for j in range(len(bias)):
            model.addCons(pre[j] == quicksum(float(weight[j, k]) * previous[k] for k in range(len(previous))) + float(bias[j]))
        if layer_index == len(request["layers"]) - 1:
            output = pre[0]
            break
        activation = [model.addVar(lb=-1.0, ub=1.0, name=f"tanh_{layer_index}_{j}") for j in range(len(bias))]
        for j in range(len(bias)):
            # Exact tanh identity: z(1 + exp(-2s)) = 1 - exp(-2s).
            exponential = (-2.0 * pre[j]).exp()
            model.addCons(activation[j] * (1.0 + exponential) == 1.0 - exponential)
        previous = activation
    model.addCons(output <= float(request["threshold"]), name="learned_feasibility")
    model.setObjective(quicksum(float(weights[i]) * x[i] for i in range(dimension)), "maximize")
    build_seconds = time.perf_counter() - build_started
    started = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - started
    status = str(model.getStatus())
    solution = model.getBestSol()
    feasible = solution is not None
    point = np.asarray([float(model.getSolVal(solution, v)) for v in x], dtype=float) if feasible else None
    try:
        gap = float(model.getGap())
    except Exception:
        gap = None
    result = {
        "status": status,
        "globally_certified": status == "optimal",
        "feasible": feasible,
        "timeout": status == "timelimit",
        "point_normalized": point.tolist() if point is not None else None,
        "objective_normalized_sum": float(np.asarray(point).sum()) if feasible else None,
        "objective_normalized_weighted": float(model.getObjVal()) if feasible else None,
        "objective_weights": weights.tolist(), "lower": lower.tolist(), "upper": upper.tolist(), "build_seconds": build_seconds,
        "primal_bound": float(model.getPrimalbound()),
        "dual_bound": float(model.getDualbound()),
        "optimality_gap": gap,
        "node_count": int(model.getNNodes()),
        "solve_seconds": solve_seconds,
        "learned_constraint_residual_numpy": None if point is None else max(evaluate(request["layers"], point) - float(request["threshold"]), 0.0),
    }
    open(destination, "w", encoding="utf-8").write(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

