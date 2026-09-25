"""Strict IPOPT entry point for the smooth baseline; no heuristic fallback."""
from __future__ import annotations
import numpy as np
from trio_paper.baselines import SmoothMLP
from .availability import baseline_require_solver


def baseline_smooth_mlp_ipopt_project(model: SmoothMLP, x0: np.ndarray, threshold: float, starts: int = 8) -> dict:
    backend = baseline_require_solver("ipopt")
    if backend != "casadi": raise RuntimeError(f"Unsupported IPOPT driver '{backend}'; no fallback is permitted.")
    import casadi as ca
    x0 = np.asarray(x0, float); x = ca.SX.sym("x", 2); value = x
    import torch
    layers = list(model.network)
    for index, layer in enumerate(layers):
        if not hasattr(layer, "weight"): continue
        weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
        value = ca.DM(weight) @ value + ca.DM(bias)
        # Tanh follows every non-final affine layer in SmoothMLP.
        following = layers[index + 1] if index + 1 < len(layers) else None
        if following is not None and following.__class__.__name__ == "Tanh": value = ca.tanh(value)
    output = value[0]; constraints = ca.vertcat(output - float(threshold), ca.dot(x, x) - 1.)
    solver = ca.nlpsol("baseline_ipopt", "ipopt", {"x": x, "f": .5 * ca.dot(x - ca.DM(x0), x - ca.DM(x0)), "g": constraints}, {"print_time": False, "ipopt.print_level": 0})
    rng = np.random.default_rng(704); starts_array = [np.clip(x0, -1, 1.)]
    for _ in range(max(0, starts - 1)):
        radius, angle = np.sqrt(rng.random()), 2 * np.pi * rng.random(); starts_array.append(radius * np.array([np.cos(angle), np.sin(angle)]))
    answers = []
    for initial in starts_array:
        try:
            result = solver(x0=initial, lbx=[-1., -1.], ubx=[1., 1.], lbg=[-ca.inf, -ca.inf], ubg=[0., 0.])
            # Do not call the PyTorch predictor here: the canonical timing
            # protocol performs candidate validation after the timed solver
            # interval.  IPOPT's own constraints still enforce feasibility.
            point = np.asarray(result["x"]).reshape(-1); objective = float(result["f"])
            answers.append((objective, point, solver.stats().get("return_status")))
        except Exception as error: answers.append((float("inf"), None, f"error:{error}"))
    candidates = [answer for answer in answers if answer[1] is not None]
    best = min(candidates, key=lambda answer: answer[0]) if candidates else None
    return {"backend": "casadi_ipopt", "status": best[2] if best else "no_local_solution", "objective": best[0] if best else None, "point": best[1].tolist() if best else None, "feasibility_violation": None, "starts": len(starts_array), "success": best is not None, "locally_certified": False, "global_optimality_certified": False, "note": "Local / not globally certified; derivatives are CasADi autodiff. Candidate validation occurs outside timing."}

