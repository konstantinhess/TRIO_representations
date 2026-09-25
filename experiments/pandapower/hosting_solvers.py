"""Exact learned-model optimization utilities for the 5-D hosting pilot."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def relu_interval_bounds(model, lower: np.ndarray, upper: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Exact interval propagation over a box, used as valid ReLU big-M bounds."""
    lo, hi, rows = np.asarray(lower, float), np.asarray(upper, float), []
    for layer in model.linears:
        weight, bias = layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()
        pre_lo = weight.clip(min=0) @ lo + weight.clip(max=0) @ hi + bias
        pre_hi = weight.clip(min=0) @ hi + weight.clip(max=0) @ lo + bias
        rows.append({"pre_lower": pre_lo, "pre_upper": pre_hi})
        if layer is not model.linears[-1]: lo, hi = np.maximum(pre_lo, 0.), np.maximum(pre_hi, 0.)
    return rows


def _box_bounds(dimension: int, lower: np.ndarray | float = 0.0, upper: np.ndarray | float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (dimension,)).copy()
    upper = np.broadcast_to(np.asarray(upper, dtype=float), (dimension,)).copy()
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("box bounds must be finite with lower <= upper")
    return lower, upper


def build_relu_gurobi_hosting_model(
    model, threshold: float = 1.0, objective_weights: np.ndarray | None = None,
    lower: np.ndarray | float = 0.0, upper: np.ndarray | float = 1.0,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Build the exact frozen ReLU MILP on ``[0,1]^d`` using propagated bounds."""
    import gurobipy as gp
    from gurobipy import GRB

    started = time.perf_counter()
    dimension = int(model.input_dim)
    lower, upper = _box_bounds(dimension, lower, upper)
    weights = np.ones(dimension, dtype=float) if objective_weights is None else np.asarray(objective_weights, dtype=float)
    if weights.shape != (dimension,) or not np.all(np.isfinite(weights)):
        raise ValueError("objective_weights must be a finite vector of input dimension")
    bounds = relu_interval_bounds(model, lower, upper)
    problem = gp.Model("pandapower_relu_hosting_capacity")
    problem.Params.OutputFlag = 0
    inputs = [problem.addVar(lb=float(lower[index]), ub=float(upper[index]), name=f"u_{index}") for index in range(dimension)]
    previous, binary_count = inputs, 0
    for layer_index, layer in enumerate(model.linears):
        weight, bias = layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()
        lo, hi = bounds[layer_index]["pre_lower"], bounds[layer_index]["pre_upper"]
        pre = [gp.quicksum(float(weight[row, column]) * previous[column] for column in range(len(previous))) + float(bias[row]) for row in range(len(lo))]
        if layer_index == len(model.linears) - 1:
            output = pre[0]
            break
        current = []
        for index, (pre_lo, pre_hi) in enumerate(zip(lo, hi)):
            z = problem.addVar(lb=max(float(pre_lo), 0.), ub=max(float(pre_hi), 0.), name=f"z_{layer_index}_{index}")
            if pre_hi <= 0: problem.addConstr(z == 0.)
            elif pre_lo >= 0: problem.addConstr(z == pre[index])
            else:
                active = problem.addVar(vtype=GRB.BINARY, name=f"active_{layer_index}_{index}"); binary_count += 1
                problem.addConstr(z >= pre[index]); problem.addConstr(z >= 0.)
                problem.addConstr(z <= float(pre_hi) * active)
                problem.addConstr(z <= pre[index] - float(pre_lo) * (1 - active))
            current.append(z)
        previous = current
    problem.addConstr(output <= float(threshold), name="learned_feasibility")
    problem.setObjective(gp.quicksum(float(weights[index]) * inputs[index] for index in range(dimension)), GRB.MAXIMIZE)
    problem.update()
    return problem, inputs, {"build_seconds": time.perf_counter() - started, "binary_count": binary_count, "variable_count": int(problem.NumVars), "constraint_count": int(problem.NumConstrs), "objective_weights": weights.tolist(), "lower": lower.tolist(), "upper": upper.tolist(), "bounds": [{key: value.tolist() for key, value in row.items()} for row in bounds]}


def solve_gurobi_hosting_model(problem, inputs: list[Any]) -> dict[str, Any]:
    """Solve a prebuilt exact MILP; only Gurobi OPTIMAL is globally certified."""
    from gurobipy import GRB

    started = time.perf_counter(); problem.optimize(); elapsed = time.perf_counter() - started
    status = int(problem.Status); feasible = problem.SolCount > 0; optimal = status == GRB.OPTIMAL
    return {"status_code": status, "status": {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE"}.get(status, str(status)),
            "globally_certified": bool(optimal), "feasible": bool(feasible), "objective_normalized_sum": float(problem.ObjVal) if feasible else None,
            "point_normalized": [float(item.X) for item in inputs] if feasible else None,
            "mip_gap": float(problem.MIPGap) if feasible and not optimal else 0.0 if optimal else None,
            "node_count": float(problem.NodeCount), "solve_seconds": elapsed}


def solve_tanh_ipopt_hosting(
    model, threshold: float = 1.0, starts: int = 8, seed: int = 704, objective_weights: np.ndarray | None = None,
    lower: np.ndarray | float = 0.0, upper: np.ndarray | float = 1.0,
) -> dict[str, Any]:
    """Deterministic multistart local IPOPT solve for the frozen tanh MLP."""
    import casadi as ca

    dimension = int(model.input_dim)
    lower, upper = _box_bounds(dimension, lower, upper)
    weights = np.ones(dimension, dtype=float) if objective_weights is None else np.asarray(objective_weights, dtype=float)
    if weights.shape != (dimension,) or not np.all(np.isfinite(weights)):
        raise ValueError("objective_weights must be a finite vector of input dimension")
    build_started = time.perf_counter()
    x = ca.SX.sym("u", dimension); value = x
    layers = list(model.network)
    for index, layer in enumerate(layers):
        if not hasattr(layer, "weight"): continue
        weight, bias = layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()
        value = ca.DM(weight) @ value + ca.DM(bias)
        following = layers[index + 1] if index + 1 < len(layers) else None
        if following is not None and following.__class__.__name__ == "Tanh": value = ca.tanh(value)
    solver = ca.nlpsol(
        "pandapower_tanh_hosting", "ipopt",
        {"x": x, "f": -ca.dot(ca.DM(weights), x), "g": value[0] - float(threshold)},
        {
            "print_time": False, "ipopt.print_level": 0,
            # Do not let IPOPT relax the physical [0, 1]^d box.  This avoids
            # reporting an almost-boundary point that the frozen AC oracle
            # correctly rejects as out of domain.
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.tol": 1e-10,
            "ipopt.constr_viol_tol": 1e-10,
            "ipopt.acceptable_tol": 1e-10,
            "ipopt.acceptable_constr_viol_tol": 1e-10,
        },
    )
    build_seconds = time.perf_counter() - build_started
    rng = np.random.default_rng(seed)
    starts_array = [(lower + upper) / 2.0] + [rng.uniform(lower, upper, size=dimension) for _ in range(max(0, starts - 1))]
    def _run_starts(record_attempts: bool) -> tuple[list[np.ndarray], list[dict[str, Any],], float]:
        attempts, candidates = [], []
        solve_started = time.perf_counter()
        for start in starts_array:
            try:
                result = solver(x0=start, lbx=lower, ubx=upper, lbg=-ca.inf, ubg=0.)
                point, residual, status = np.asarray(result["x"]).reshape(-1), float(np.asarray(result["g"]).reshape(-1)[0]), str(solver.stats().get("return_status"))
                # IPOPT terminates within its feasibility tolerance rather than
                # necessarily returning a bitwise box-boundary point.  The pilot's
                # external feasibility convention is 1e-6, so accept such a point
                # at that tolerance but retain it unmodified for oracle evaluation.
                box_violation = float(max(0., float((lower - point).max()), float((point - upper).max())))
                valid = bool(residual <= 1e-6 and box_violation <= 1e-6)
                if record_attempts:
                    attempts.append({"start": start.tolist(), "status": status, "constraint_residual": residual,
                                     "box_violation": box_violation, "valid": valid, "point": point.tolist()})
                if valid: candidates.append(point)
            except Exception as error:
                if record_attempts:
                    attempts.append({"start": start.tolist(), "status": f"error:{type(error).__name__}", "valid": False, "error": str(error)})
        return candidates, attempts, time.perf_counter() - solve_started

    candidates, attempts, solve_seconds = _run_starts(record_attempts=True)
    # The initial solve fully warms the CasADi/IPOPT graph.  Time 20 further
    # deterministic eight-start solves without graph construction or logging.
    timing_samples = np.asarray([_run_starts(record_attempts=False)[2] for _ in range(20)], dtype=float)
    best = max(candidates, key=lambda point: float(weights @ point)) if candidates else None
    return {"status": "local_solution" if best is not None else "no_valid_local_solution", "globally_certified": False,
            "local_noncertified": True, "starts": len(starts_array), "successful_valid_starts": len(candidates), "attempts": attempts,
            "point_normalized": best.tolist() if best is not None else None, "objective_normalized_sum": float(best.sum()) if best is not None else None,
            "objective_normalized_weighted": float(weights @ best) if best is not None else None, "objective_weights": weights.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
            "build_seconds": build_seconds, "solve_seconds": solve_seconds, "total_one_shot_seconds": build_seconds + solve_seconds,
            "solve_mean_seconds_20": float(timing_samples.mean()), "solve_sd_seconds_20": float(timing_samples.std(ddof=1))}


def solve_icnn_clarabel_hosting(
    model, threshold: float = 1.0, objective_weights: np.ndarray | None = None,
    lower: np.ndarray | float = 0.0, upper: np.ndarray | float = 1.0,
) -> dict[str, Any]:
    """Exact convex epigraph formulation of the frozen ICNN hosting problem."""
    import cvxpy as cp

    dimension, width = int(model.input_dim), int(model.width)
    lower, upper = _box_bounds(dimension, lower, upper)
    weights = np.ones(dimension, dtype=float) if objective_weights is None else np.asarray(objective_weights, dtype=float)
    if weights.shape != (dimension,) or not np.all(np.isfinite(weights)):
        raise ValueError("objective_weights must be a finite vector of input dimension")
    build_started = time.perf_counter()
    x, z1, z2 = cp.Variable(dimension), cp.Variable(width), cp.Variable(width)
    first_w, first_b = model.input_first.weight.detach().cpu().numpy(), model.input_first.bias.detach().cpu().numpy()
    hin_w, hin_b = model.hidden_input.weight.detach().cpu().numpy(), model.hidden_input.bias.detach().cpu().numpy()
    hidden_bias = model.hidden_bias.detach().cpu().numpy(); wz = model.hidden_nonnegative.detach().cpu().numpy(); out_z = model.output_nonnegative.detach().cpu().numpy()
    out_w, out_b = model.output_input.weight.detach().cpu().numpy()[0], float(model.output_input.bias.detach())
    softplus = lambda affine: cp.log_sum_exp(cp.vstack([np.zeros(width), affine]), axis=0)
    constraints = [z1 >= softplus(first_w @ x + first_b), z2 >= softplus(wz @ z1 + hin_w @ x + hin_b + hidden_bias),
                   out_z @ z2 + out_w @ x + out_b <= float(threshold), x >= lower, x <= upper]
    problem = cp.Problem(cp.Maximize(weights @ x), constraints)
    build_seconds = time.perf_counter() - build_started
    def _solve_once() -> tuple[float, float]:
        started = time.perf_counter()
        value = problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9, warm_start=False)
        return float(value), time.perf_counter() - started

    value, solve_seconds = _solve_once()
    # The first call caches CVXPY canonicalization.  Retain it as the
    # one-shot solve and time 20 solver-only repetitions after warm-up.
    timing_samples = np.asarray([_solve_once()[1] for _ in range(20)], dtype=float)
    success = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None
    return {"status": problem.status, "globally_certified": bool(success and problem.status == cp.OPTIMAL), "feasible": bool(success),
            "point_normalized": np.asarray(x.value).reshape(-1).tolist() if success else None,
            "objective_normalized_sum": float(np.asarray(x.value).sum()) if success else None,
            "objective_normalized_weighted": float(value) if success else None, "objective_weights": weights.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
            "build_seconds": build_seconds, "solve_seconds": solve_seconds, "total_one_shot_seconds": build_seconds + solve_seconds,
            "solve_mean_seconds_20": float(timing_samples.mean()), "solve_sd_seconds_20": float(timing_samples.std(ddof=1))}

