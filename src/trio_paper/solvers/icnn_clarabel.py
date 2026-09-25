"""Convex downstream projection for the conventional baseline ICNN."""
from __future__ import annotations
import numpy as np
import cvxpy as cp
from trio_paper.baselines import ICNN


def baseline_icnn_project(model: ICNN, x0: np.ndarray, threshold: float) -> dict:
    """Globally solve the ICNN sublevel projection when CLARABEL reports optimal."""
    x = cp.Variable(2); z1 = cp.Variable(model.width); z2 = cp.Variable(model.width)
    x0p = cp.Parameter(2, value=np.asarray(x0, float)); g = cp.Parameter(value=float(threshold))
    first_w, first_b = model.input_first.weight.detach().numpy(), model.input_first.bias.detach().numpy()
    hin_w, hin_b = model.hidden_input.weight.detach().numpy(), model.hidden_input.bias.detach().numpy()
    hidden_bias = model.hidden_bias.detach().numpy()
    wz = model.hidden_nonnegative.detach().numpy(); out_z = model.output_nonnegative.detach().numpy()
    out_w, out_b = model.output_input.weight.detach().numpy()[0], float(model.output_input.bias.detach())
    soft = lambda affine: cp.log_sum_exp(cp.vstack([np.zeros(model.width), affine]), axis=0)
    constraints = [z1 >= soft(first_w @ x + first_b), z2 >= soft(wz @ z1 + hin_w @ x + hin_b + hidden_bias), out_z @ z2 + out_w @ x + out_b <= g, cp.sum_squares(x) <= 1]
    problem = cp.Problem(cp.Minimize(.5 * cp.sum_squares(x - x0p)), constraints)
    value = problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    success = problem.status in ("optimal", "optimal_inaccurate") and x.value is not None
    return {"objective": float(value) if success else None, "point": np.asarray(x.value).tolist() if success else None, "feasibility_violation": 0.0 if success else None, "status": problem.status, "globally_certified": bool(success and problem.status == "optimal"), "local": False}

