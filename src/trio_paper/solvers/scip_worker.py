"""Torch-free SCIP worker for exact smooth-MLP global projection."""
from __future__ import annotations
import json, sys
import numpy as np
from pyscipopt import Model, quicksum


def evaluate_network(layers: list[dict], point: np.ndarray) -> float:
    """Independent NumPy check of the frozen tanh network."""
    value = np.asarray(point, dtype=float)
    for index, layer in enumerate(layers):
        value = np.asarray(layer["weight"], dtype=float) @ value + np.asarray(layer["bias"], dtype=float)
        if index < len(layers) - 1:
            value = np.tanh(value)
    return float(value[0])

def main(source: str, destination: str) -> None:
    data = json.loads(open(source).read()); x0 = np.asarray(data["x0"], float); model = Model("baseline_smooth_mlp_global"); model.hideOutput(True)
    model.setParam("randomization/permutationseed", 0); model.setParam("randomization/randomseedshift", 0)
    if data["timeout_seconds"] is not None: model.setParam("limits/time", float(data["timeout_seconds"]))
    x = [model.addVar(lb=-1., ub=1., name=f"x_{index}") for index in range(2)]; model.addCons(quicksum(item * item for item in x) <= 1.); previous = x
    for layer_index, layer in enumerate(data["layers"]):
        weight, bias = np.asarray(layer["weight"]), np.asarray(layer["bias"]); pre = [model.addVar(lb=None, ub=None, name=f"s_{layer_index}_{j}") for j in range(len(bias))]
        for j in range(len(bias)): model.addCons(pre[j] == quicksum(float(weight[j, k]) * previous[k] for k in range(len(previous))) + float(bias[j]))
        if layer_index == len(data["layers"]) - 1: output = pre[0]; break
        current = [model.addVar(lb=-1., ub=1., name=f"z_{layer_index}_{j}") for j in range(len(bias))]
        for j in range(len(bias)):
            e = (-2 * pre[j]).exp(); model.addCons(current[j] * (1 + e) == 1 - e)
        previous = current
    model.addCons(output <= float(data["threshold"])); objective = model.addVar(lb=0., name="projection_objective")
    model.addCons(objective >= .5 * quicksum((x[index] - float(x0[index])) * (x[index] - float(x0[index])) for index in range(2))); model.setObjective(objective, "minimize"); model.optimize()
    status = str(model.getStatus()); solution = model.getBestSol(); feasible = solution is not None
    try: gap = float(model.getGap())
    except Exception: gap = None
    point = np.asarray([float(model.getSolVal(solution, item)) for item in x]) if feasible else None
    violation = None if point is None else max(
        evaluate_network(data["layers"], point) - float(data["threshold"]),
        float(point @ point - 1.0),
        0.0,
    )
    result = {"status": status, "objective": max(float(model.getObjVal()), 0.0) if feasible else None, "lower_bound": float(model.getDualbound()) if feasible else None, "optimality_gap": gap, "point": point.tolist() if point is not None else None, "feasibility_violation": violation, "node_count": int(model.getNNodes()), "timeout": status == "timelimit", "globally_certified": status == "optimal"}
    open(destination, "w").write(json.dumps(result))
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])

