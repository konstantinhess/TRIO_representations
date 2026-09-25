from __future__ import annotations

import numpy as np
from scipy import ndimage


def regression_metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    error = prediction - target
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "correlation": float(np.corrcoef(target, prediction)[0, 1])}


def mask_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    intersection = np.count_nonzero(truth & prediction)
    union = np.count_nonzero(truth | prediction)
    return {
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(intersection / np.count_nonzero(prediction)) if prediction.any() else float(not truth.any()),
        "recall": float(intersection / np.count_nonzero(truth)) if truth.any() else float(not prediction.any()),
    }


def component_count(mask, connectivity: int):
    rank = np.asarray(mask).ndim
    structure = ndimage.generate_binary_structure(rank, 1 if connectivity == 4 else rank)
    return int(ndimage.label(mask, structure=structure)[1])


def mean_sample_sd(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {"mean": float(values.mean()) if len(values) else None, "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None, "count": int(len(values))}
