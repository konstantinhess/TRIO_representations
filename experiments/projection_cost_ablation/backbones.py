"""Deterministic radial backbones sharing one synthetic ellipsoid geometry."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trio_paper.models import BrokenPowerTRIO, build_trio


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    model: torch.nn.Module
    frozen: Any
    parameter_count: int


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _inverse_softplus_numpy(value):
    value = np.asarray(value, dtype=np.float64)
    return value + np.log(-np.expm1(-value))


def shared_spatial_parameters(config, seed):
    """Generate c, Cholesky factors, and target radii from one deterministic seed."""
    experts = int(config["experts"])
    generator = np.random.default_rng(int(seed))
    angles = generator.uniform(-np.pi, np.pi, size=experts)
    radial = np.sqrt(generator.uniform(0.0, .45 ** 2, size=experts))
    centers = np.column_stack((radial * np.cos(angles), radial * np.sin(angles)))
    diagonal = generator.uniform(.70, 1.35, size=(experts, 2))
    offdiag = generator.uniform(-.20, .20, size=experts)
    radii = generator.choice(np.asarray(config["desired_radius_values"], dtype=np.float64), size=experts, replace=True)
    epsilon = 1e-8
    return {
        "centers": centers,
        "raw_diag": _inverse_softplus_numpy(diagonal - epsilon),
        "offdiag": offdiag,
        "desired_radii": radii,
    }


def deterministic_queries(config, seed):
    generator = np.random.default_rng(int(seed) + 1)
    count = int(config["query_count"])
    radius = np.sqrt(generator.uniform(0.0, 1.0, size=count))
    angle = generator.uniform(-np.pi, np.pi, size=count)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def _make_model(spec, experts):
    return build_trio(
        spec["backbone"], experts, input_dim=2,
        radial_units=spec.get("units"), spline_bins=spec.get("bins", 6),
    )


def _copy_spatial(model, spatial):
    with torch.no_grad():
        model.centers.copy_(torch.as_tensor(spatial["centers"], dtype=torch.float64))
        model.raw_diag.copy_(torch.as_tensor(spatial["raw_diag"], dtype=torch.float64))
        model.offdiag.copy_(torch.as_tensor(spatial["offdiag"], dtype=torch.float64))
        model.beta.zero_()


def _set_target_matched_offsets(model, target, radii):
    distances = torch.as_tensor(radii[None], dtype=torch.float64)
    with torch.no_grad():
        if isinstance(model, BrokenPowerTRIO):
            p1, p2, kappa = model.p1, model.p2, model.kappa
            at_break = .5 * kappa.pow(p1)
            inner = .5 * distances.pow(p1)
            outer = at_break + .5 * p1 / p2 * kappa.pow(p1 - p2) * (distances.pow(p2) - kappa.pow(p2))
            radial_values = torch.where(distances <= kappa, inner, outer)[0]
        else:
            radial_values = model.radial(distances)[0]
        if not bool(torch.isfinite(radial_values).all()) or not bool((radial_values > 0).all()):
            raise AssertionError("The initialized radial laws must be finite and positive at every desired radius")
        model.beta.copy_(float(target) - radial_values)


def build_variants(config=None, seed=101):
    config = config or load_config()
    torch.manual_seed(int(seed))
    spatial = shared_spatial_parameters(config, seed)
    variants = {}
    for spec in config["variants"]:
        model = _make_model(spec, int(config["experts"])).eval()
        _copy_spatial(model, spatial)
        _set_target_matched_offsets(model, float(config["target"]), spatial["desired_radii"])
        count = parameter_count(model)
        expected = int(config["expected_parameter_counts"][spec["key"]])
        if count != expected:
            raise AssertionError(f"{spec['key']} has {count} parameters, expected {expected}")
        variants[spec["key"]] = Variant(spec["key"], spec["label"], model, model.freeze(), count)
    return variants, spatial
