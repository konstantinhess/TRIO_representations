"""Cached exact 2-D projection onto a compiled union of ellipsoids."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq

EPS = 2e-8


class CachedEllipsoidProjector:
    def __init__(self, frozen_model):
        self.model = frozen_model
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(frozen_model.matrices)

    def _onto_ellipsoid(self, index, radius_squared, query):
        center = self.model.centers[index]; values = self.eigenvalues[index]; vectors = self.eigenvectors[index]
        rotated = vectors.T @ (query - center)
        if np.sum(values * rotated * rotated) <= radius_squared:
            return query.copy()
        equation = lambda mu: float(np.sum(values * np.square(rotated / (1 + mu * values))) - radius_squared)
        high = 1.0
        while equation(high) > 0: high *= 2
        mu = brentq(equation, 0., high, xtol=1e-13, rtol=1e-13)
        return center + vectors @ (rotated / (1 + mu * values))

    def _circle_intersections(self, index, radius_squared):
        center = self.model.centers[index]; matrix = self.model.matrices[index]
        b = matrix @ center; constant = float(center @ b - radius_squared)
        polynomial = np.polynomial.Polynomial
        one, t = polynomial([1]), polynomial([0, 1]); denominator = one + t * t
        ux, uy = one - t * t, 2 * t
        expression = matrix[0, 0] * ux * ux + 2 * matrix[0, 1] * ux * uy + matrix[1, 1] * uy * uy - 2 * (b[0] * ux + b[1] * uy) * denominator + constant * denominator * denominator
        roots = np.roots(expression.coef[::-1]); candidates = []
        for root in roots:
            if abs(root.imag) <= 2e-8 * max(1., abs(root.real)):
                angle = 2 * math.atan(float(root.real)); candidates.append(np.array([math.cos(angle), math.sin(angle)]))
        left = np.array([-1., 0.])
        if abs(float((left - center) @ matrix @ (left - center) - radius_squared)) <= 2e-8 * max(1., abs(radius_squared)):
            candidates.append(left)
        unique = []
        for candidate in candidates:
            residual = abs(float((candidate - center) @ matrix @ (candidate - center) - radius_squared))
            if residual <= 2e-6 * max(1., abs(radius_squared)) and not any(np.linalg.norm(candidate - old) < 1e-7 for old in unique):
                unique.append(candidate)
        return unique

    def project_compiled(self, compiled, query, expert_indices=None):
        query = np.asarray(query, dtype=np.float64)
        best, best_value = None, float("inf")
        indices = compiled.expert_indices if expert_indices is None else expert_indices
        for index in indices:
            point = self._onto_ellipsoid(index, compiled.squared_radii[index], query)
            if np.linalg.norm(point) > 1 + EPS:
                candidates = self._circle_intersections(index, compiled.squared_radii[index])
                point = min(candidates, key=lambda x: np.square(x - query).sum()) if candidates else None
            if point is not None:
                value = float(np.square(point - query).sum())
                if value < best_value: best, best_value = point, value
        return best, best_value
