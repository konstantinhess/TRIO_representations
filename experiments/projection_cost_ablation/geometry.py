"""Proof obligations for the identical compiled-union construction."""
from __future__ import annotations

import numpy as np


def compiled_radii(certificate):
    return np.sqrt(np.maximum(np.asarray(certificate.squared_radii, dtype=np.float64), 0))


def validate_geometry(variants, desired_radii, target, *, tolerance=2e-7, grid_resolution=101):
    reference_key = next(iter(variants))
    reference = variants[reference_key].frozen.compile(float(target))
    reference_radii = compiled_radii(reference)
    expected_active = np.ones_like(reference.active, dtype=bool)
    np.testing.assert_array_equal(reference.active, expected_active)
    np.testing.assert_allclose(reference_radii, desired_radii, rtol=0, atol=tolerance)

    axis = np.linspace(-1, 1, int(grid_resolution))
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid = grid[np.square(grid).sum(axis=1) <= 1]
    reference_membership = reference.membership(grid)
    checks = []
    for key, variant in variants.items():
        compiled = variant.frozen.compile(float(target))
        np.testing.assert_allclose(compiled.centers, reference.centers, rtol=0, atol=0)
        np.testing.assert_allclose(compiled.matrices, reference.matrices, rtol=0, atol=0)
        np.testing.assert_array_equal(compiled.active, reference.active)
        radii = compiled_radii(compiled)
        np.testing.assert_allclose(radii, desired_radii, rtol=0, atol=tolerance)
        np.testing.assert_allclose(radii, reference_radii, rtol=0, atol=tolerance)
        mismatches = int(np.count_nonzero(compiled.membership(grid) != reference_membership))
        if mismatches:
            raise AssertionError(f"{variant.label} disagrees on {mismatches} deterministic grid points")
        checks.append({
            "backbone_key": key,
            "backbone": variant.label,
            "exact_parameter_count": variant.parameter_count,
            "active_ellipsoids": int(compiled.active.sum()),
            "max_radius_error": float(np.max(np.abs(radii - desired_radii))),
            "membership_mismatches": mismatches,
        })
    return {
        "reference_backbone": variants[reference_key].label,
        "target": float(target),
        "grid_points": int(len(grid)),
        "radius_tolerance": float(tolerance),
        "checks": checks,
    }
