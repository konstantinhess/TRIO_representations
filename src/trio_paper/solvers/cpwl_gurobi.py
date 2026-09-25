"""Exact bounded-domain CPWL MIP encoding; backend use is strictly optional."""
from __future__ import annotations
import numpy as np
import torch
from trio_paper.baselines import CPWLMLP
from .availability import baseline_require_solver


def cpwl_propagated_bounds(model: CPWLMLP, lower=(-1., -1.), upper=(1., 1.)) -> list[dict]:
    """Valid interval bounds used as tight big-M constants, never arbitrary M."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float); rows = []
    for index, layer in enumerate(model.linears):
        weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
        pre_lo = weight.clip(min=0) @ lo + weight.clip(max=0) @ hi + bias
        pre_hi = weight.clip(min=0) @ hi + weight.clip(max=0) @ lo + bias
        rows.append({"layer": index, "pre_lower": pre_lo.tolist(), "pre_upper": pre_hi.tolist()})
        lo, hi = (np.maximum(pre_lo, 0), np.maximum(pre_hi, 0)) if index < len(model.linears) - 1 else (pre_lo, pre_hi)
    return rows


def cpwl_mip_project(model: CPWLMLP, x0: np.ndarray, threshold: float, timeout_seconds: float | None = None) -> dict:
    backend = baseline_require_solver("mip")
    if backend == "gurobi":
        return cpwl_gurobi_project(model, x0, threshold, timeout_seconds)
    if backend != "scip": raise RuntimeError(f"The installed MIP backend '{backend}' has no baseline adapter yet; no fallback is permitted.")
    from pyscipopt import Model, quicksum
    bounds = cpwl_propagated_bounds(model)
    problem = Model("baseline_cpwl_mlp_mip")
    problem.hideOutput(True)
    if timeout_seconds is not None: problem.setParam("limits/time", float(timeout_seconds))
    x0 = np.asarray(x0, float)
    inputs = [problem.addVar(lb=-1., ub=1., name=f"x_{index}") for index in range(2)]
    problem.addCons(quicksum(item * item for item in inputs) <= 1.)
    previous = inputs
    binaries = 0
    for layer_index, layer in enumerate(model.linears):
        weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
        lower, upper = np.asarray(bounds[layer_index]["pre_lower"]), np.asarray(bounds[layer_index]["pre_upper"])
        pre = [problem.addVar(lb=float(lower[j]), ub=float(upper[j]), name=f"s_{layer_index}_{j}") for j in range(len(lower))]
        for j in range(len(lower)):
            problem.addCons(pre[j] == quicksum(float(weight[j, k]) * previous[k] for k in range(len(previous))) + float(bias[j]))
        if layer_index == len(model.linears) - 1:
            output = pre[0]
            break
        current = []
        for j, (lo, hi) in enumerate(zip(lower, upper)):
            z = problem.addVar(lb=max(float(lo), 0.), ub=max(float(hi), 0.), name=f"z_{layer_index}_{j}")
            if hi <= 0: problem.addCons(z == 0.)
            elif lo >= 0: problem.addCons(z == pre[j])
            else:
                active = problem.addVar(vtype="B", name=f"a_{layer_index}_{j}"); binaries += 1
                problem.addCons(z >= pre[j]); problem.addCons(z >= 0.)
                problem.addCons(z <= float(hi) * active)
                problem.addCons(z <= pre[j] - float(lo) * (1 - active))
            current.append(z)
        previous = current
    problem.addCons(output <= float(threshold))
    # PySCIPOpt requires a linear objective.  This epigraph is exactly the
    # convex quadratic projection objective and remains a MIQCP.
    objective = problem.addVar(lb=0., name="projection_objective")
    problem.addCons(objective >= .5 * quicksum((inputs[index] - float(x0[index])) * (inputs[index] - float(x0[index])) for index in range(2)))
    problem.setObjective(objective, "minimize")
    problem.optimize()
    status = str(problem.getStatus())
    solution = problem.getBestSol()
    feasible = solution is not None and status not in {"infeasible", "unknown"}
    objective = float(problem.getObjVal()) if feasible else None
    point = [float(problem.getSolVal(solution, value)) for value in inputs] if feasible else None
    try: gap = float(problem.getGap())
    except Exception: gap = None
    return {"backend": "scip", "status": status, "objective": objective, "point": point, "feasibility_violation": 0.0 if feasible else None, "mip_gap": gap, "node_count": int(problem.getNNodes()), "timeout": status == "timelimit", "globally_certified": bool(status == "optimal"), "big_m_bounds": bounds, "binary_count": binaries, "exact_relative_to_learned_model": True}


def cpwl_gurobi_project(model: CPWLMLP, x0: np.ndarray, threshold: float, timeout_seconds: float | None = None) -> dict:
    """Exact MIQP encoding of the frozen CPWL network over the unit disk box."""
    import gurobipy as gp
    from gurobipy import GRB
    bounds = cpwl_propagated_bounds(model)
    problem = gp.Model("baseline_cpwl_mlp_mip"); problem.Params.OutputFlag = 0
    if timeout_seconds is not None: problem.Params.TimeLimit = float(timeout_seconds)
    x0 = np.asarray(x0, float); inputs = [problem.addVar(lb=-1., ub=1., name=f"x_{index}") for index in range(2)]; problem.addQConstr(gp.quicksum(item * item for item in inputs) <= 1.); previous = inputs; binaries = 0
    for layer_index, layer in enumerate(model.linears):
        weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy(); lower, upper = np.asarray(bounds[layer_index]["pre_lower"]), np.asarray(bounds[layer_index]["pre_upper"])
        # Keep each preactivation as an affine expression rather than a
        # redundant auxiliary variable.  This is algebraically identical to
        # the conventional big-M encoding, but avoids crossing the 200-var
        # limit of the installed academic Gurobi licence when all 66 CPWLs
        # are ambiguous for a frozen seed.
        pre = [gp.quicksum(float(weight[j, k]) * previous[k] for k in range(len(previous))) + float(bias[j]) for j in range(len(lower))]
        if layer_index == len(model.linears) - 1: output = pre[0]; break
        current = []
        for j, (lo, hi) in enumerate(zip(lower, upper)):
            z = problem.addVar(lb=max(float(lo), 0.), ub=max(float(hi), 0.), name=f"z_{layer_index}_{j}")
            if hi <= 0: problem.addConstr(z == 0.)
            elif lo >= 0: problem.addConstr(z == pre[j])
            else:
                active = problem.addVar(vtype=GRB.BINARY, name=f"a_{layer_index}_{j}"); binaries += 1
                problem.addConstr(z >= pre[j]); problem.addConstr(z >= 0.); problem.addConstr(z <= float(hi) * active); problem.addConstr(z <= pre[j] - float(lo) * (1 - active))
            current.append(z)
        previous = current
    problem.addConstr(output <= float(threshold)); objective = problem.addVar(lb=0., name="projection_objective")
    problem.addQConstr(gp.quicksum((inputs[index] - float(x0[index])) * (inputs[index] - float(x0[index])) for index in range(2)) <= 2 * objective)
    problem.setObjective(objective, GRB.MINIMIZE); problem.optimize()
    status = int(problem.Status); feasible = problem.SolCount > 0; optimal = status == GRB.OPTIMAL
    return {"backend": "gurobi", "status": status, "objective": float(problem.ObjVal) if feasible else None, "point": [float(value.X) for value in inputs] if feasible else None, "feasibility_violation": 0.0 if feasible else None, "mip_gap": float(problem.MIPGap) if feasible and not optimal else 0.0 if optimal else None, "node_count": float(problem.NodeCount), "timeout": status == GRB.TIME_LIMIT, "globally_certified": optimal, "big_m_bounds": bounds, "binary_count": binaries, "exact_relative_to_learned_model": True}


