from __future__ import annotations

import numpy as np


def uniform_unit_disk(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.random(count))
    angle = 2.0 * np.pi * rng.random(count)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
