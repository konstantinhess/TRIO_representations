from __future__ import annotations
import numpy as np

def six_hump_camel(x):
    x = np.asarray(x, dtype=np.float64); x1, x2 = x[..., 0], x[..., 1]
    return (4 - 2.1*x1*x1 + x1**4/3)*x1*x1 + x1*x2 + (-4 + 4*x2*x2)*x2*x2

def raw_to_normalized(x, config):
    return (np.asarray(x) - np.asarray(config["normalization"]["center"])) * np.asarray(config["normalization"]["scale"])

def normalized_to_raw(x, config):
    return np.asarray(x) / np.asarray(config["normalization"]["scale"]) + np.asarray(config["normalization"]["center"])
