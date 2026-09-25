"""Training-data-only deterministic centre initialization."""
from __future__ import annotations

import numpy as np
import torch


def _array(points: np.ndarray | torch.Tensor) -> np.ndarray:
    out = np.asarray(points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points, dtype=np.float64)
    if out.ndim != 2 or not len(out) or not np.isfinite(out).all():
        raise ValueError("points must be a nonempty finite [N,d] array")
    return out


def farthest_point_indices(points: np.ndarray | torch.Tensor, count: int) -> np.ndarray:
    points = _array(points)
    if count < 1 or count > len(points):
        raise ValueError("invalid FPS count")
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.square(points).sum(1).argmin())
    nearest = np.square(points - points[selected[0]]).sum(1)
    for index in range(1, count):
        selected[index] = int(nearest.argmax())
        nearest = np.minimum(nearest, np.square(points - points[selected[index]]).sum(1))
    return selected


def mixed_low_target_fps_indices(points, targets, count: int, low_target_fraction: float = .10) -> np.ndarray:
    points = _array(points)
    targets = np.asarray(targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets, dtype=np.float64).reshape(-1)
    if len(targets) != len(points) or not np.isfinite(targets).all():
        raise ValueError("targets must have one finite value per point")
    global_count = count // 2
    low_count = count - global_count
    global_rows = farthest_point_indices(points, global_count)
    low_pool = np.flatnonzero(targets <= np.quantile(targets, low_target_fraction))
    low_rows = low_pool[farthest_point_indices(points[low_pool], low_count)]
    return np.concatenate((global_rows, low_rows))


def initialize_from_training(model, points, targets, mode: str) -> np.ndarray:
    points = _array(points)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    count = len(model.beta)
    if mode == "standard":
        return np.arange(count, dtype=np.int64)
    if mode != "mixed_fps":
        raise ValueError(f"unknown initialization mode: {mode}")
    rows = mixed_low_target_fps_indices(points, targets, count)
    with torch.no_grad():
        model.centers.copy_(torch.as_tensor(points[rows], dtype=model.centers.dtype))
        model.beta.copy_(torch.as_tensor(targets[rows], dtype=model.beta.dtype))
    return rows
