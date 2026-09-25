"""Global SCIP spatial branch-and-bound for the frozen smooth-MLP baseline."""
from __future__ import annotations
import json
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
from trio_paper.baselines import SmoothMLP
from .availability import baseline_require_solver


def baseline_smooth_mlp_global_project(model: SmoothMLP, x0: np.ndarray, threshold: float, timeout_seconds: float | None = None) -> dict:
    if baseline_require_solver("global") != "scip": raise RuntimeError("SCIP global NLP backend is required.")
    layers = [{"weight": layer.weight.detach().numpy().tolist(), "bias": layer.bias.detach().numpy().tolist()} for layer in model.network if hasattr(layer, "weight")]
    request = {"layers": layers, "x0": np.asarray(x0, float).tolist(), "threshold": float(threshold), "timeout_seconds": timeout_seconds}
    folder = Path(tempfile.mkdtemp(prefix="trio_scip_"))
    try:
        source, destination = folder / "request.json", folder / "result.json"; source.write_text(json.dumps(request))
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name("scip_worker.py")), str(source), str(destination)], capture_output=True, text=True, timeout=(timeout_seconds + 30 if timeout_seconds else None))
        if completed.returncode != 0: return {"backend": "scip", "status": "worker_error", "stdout": completed.stdout[-1000:], "stderr": completed.stderr[-1000:], "globally_certified": False}
        result = json.loads(destination.read_text())
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    result.update({"backend": "scip_spatial_branch_and_bound", "exact_relative_to_learned_model": True, "local": False})
    return result
