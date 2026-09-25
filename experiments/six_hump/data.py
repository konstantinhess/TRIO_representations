from __future__ import annotations
from pathlib import Path
import numpy as np
from .dgp import raw_to_normalized, six_hump_camel

def _split(count, seed, config):
    rng = np.random.default_rng(seed); domain = config["raw_domain"]
    raw = np.column_stack((rng.uniform(*domain["x1"], count), rng.uniform(*domain["x2"], count)))
    return raw, raw_to_normalized(raw, config), six_hump_camel(raw)

def generate(config, destination):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True); values = {}
    for name in ("train", "validation", "test"):
        raw, normalized, target = _split(config["data"][f"{name}_size"], config["data"][f"{name}_seed"], config)
        values.update({f"{name}_raw": raw, f"{name}_normalized": normalized, f"{name}_target": target})
    np.savez_compressed(destination, **values)

def load(path):
    values = np.load(path)
    return {name: {key: values[f"{name}_{key}"] for key in ("raw", "normalized", "target")} for name in ("train", "validation", "test")}

def grid(config, resolution=None):
    n = int(resolution or config["geometry"]["grid_resolution"]); domain = config["raw_domain"]
    a, b = np.linspace(*domain["x1"], n), np.linspace(*domain["x2"], n)
    xx, yy = np.meshgrid(a, b, indexing="xy"); raw = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, raw, raw_to_normalized(raw, config)
