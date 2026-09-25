"""Strict solver availability checks; requested methods never silently fall back."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def baseline_solver_availability() -> dict:
    modules = {name: importlib.util.find_spec(name) is not None for name in ("cyipopt", "casadi", "gurobipy", "pyscipopt", "cvxpy")}
    ipopt_plugin = False
    if modules["casadi"]:
        import casadi as ca
        ipopt_plugin = bool(ca.has_nlpsol("ipopt"))
    return {"python": sys.version, "modules": modules, "casadi_ipopt_plugin": ipopt_plugin, "executables": {name: shutil.which(name) for name in ("maingo", "baron", "scip")}, "local_smooth": "casadi" if ipopt_plugin else None, "mip": "gurobi" if modules["gurobipy"] else ("scip" if modules["pyscipopt"] else None), "global_smooth": "scip" if modules["pyscipopt"] else None}


def _baseline_package_version(distribution: str) -> str | None:
    """Return an installed distribution version without making it a requirement."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def baseline_solver_run_metadata(
    method: str,
    *,
    timeout_seconds: float | None = None,
    ipopt_starts: int = 8,
    premap_budget: int = 64,
) -> dict:
    """Reproducibility record for one completed baseline run.

    Explicit adapter settings are deliberately distinguished from untouched
    solver defaults, so a result never implies a default tolerance was set.
    """
    package_versions = {
        name: _baseline_package_version(name)
        for name in ("casadi", "gurobipy", "pyscipopt", "cvxpy", "clarabel", "torch", "numpy")
    }
    gurobi_version = None
    if importlib.util.find_spec("gurobipy") is not None:
        try:
            import gurobipy as gp
            gurobi_version = ".".join(str(item) for item in gp.gurobi.version())
        except Exception:
            pass
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "runtime": {"python": sys.version, "package_versions": package_versions, "gurobi_runtime_version": gurobi_version},
        "solver_selection": baseline_solver_availability(),
        "query_settings": {
            "timeout_seconds": timeout_seconds,
            "ipopt_multistart_count": ipopt_starts,
            "premap_refinement_budget": premap_budget,
        },
        "numerical_settings": {
            "model_dtype": "torch.float64",
            "common_domain_constraint": "x_1^2 + x_2^2 <= 1",
            "relu_mip": {
                "backend": "gurobi when installed, otherwise SCIP",
                "big_m": "valid interval-propagated preactivation bounds",
                "explicit_options": {"OutputFlag": 0, "TimeLimit": timeout_seconds},
                "tolerances": "solver defaults; no feasibility, integrality, or optimality tolerance override",
            },
            "smooth_ipopt": {
                "backend": "CasADi IPOPT",
                "explicit_options": {"print_time": False, "ipopt.print_level": 0},
                "wrapper_feasibility_acceptance_tolerance": 1e-7,
                "tolerances": "IPOPT defaults; no IPOPT tolerance override",
            },
            "smooth_global_scip": {
                "backend": "SCIP spatial branch-and-bound",
                "explicit_options": {"randomization/permutationseed": 0, "randomization/randomseedshift": 0, "limits/time": timeout_seconds},
                "tolerances": "SCIP defaults; no feasibility or optimality tolerance override",
            },
            "icnn": {
                "backend": "CVXPY/CLARABEL",
                "explicit_options": {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9},
            },
            "premap": {
                "backend": "CVXPY/CLARABEL relaxation bounds",
                "explicit_options": {},
                "tolerances": "CLARABEL defaults; no tolerance override",
            },
        },
    }


def baseline_write_solver_run_metadata(directory: Path, method: str, **settings) -> dict:
    """Persist the per-run solver record beside every future baseline artifact."""
    metadata = baseline_solver_run_metadata(method, **settings)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "baseline_solver_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class BaselineSolverUnavailable(RuntimeError):
    pass


def baseline_require_solver(kind: str) -> str:
    available = baseline_solver_availability()
    selected = available[{"ipopt": "local_smooth", "mip": "mip", "global": "global_smooth"}[kind]]
    if selected is None:
        raise BaselineSolverUnavailable(f"Requested {kind} baseline is unavailable: {available}")
    return selected

