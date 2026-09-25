"""Exact linear optimization over detached unions of SPD ellipsoids."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class EllipsoidLinearOptimum:
    """The exact minimizer of a linear form over a compiled ellipsoid union."""

    point: np.ndarray
    value: float
    expert_index: int
    active_position: int
    squared_radius: float


@dataclass(frozen=True)
class BoxEllipsoidLinearOptimum(EllipsoidLinearOptimum):
    """Exact linear optimum with the active box-face state per coordinate.

    Face entries are ``0=free``, ``1=lower`` and ``2=upper``.
    """

    face: tuple[int, ...]
    faces_examined: int
    feasible_face_candidates: int


@dataclass(frozen=True)
class _BoxFaceCache:
    """Detached per-free-mask factors for batched box-face evaluation."""

    free: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray
    faces: np.ndarray
    cholesky: np.ndarray | None
    reduced_vector: np.ndarray | None
    support: np.ndarray | None
    solve_fixed_block: np.ndarray | None
    schur: np.ndarray | None


class BoxedEllipsoidLinearOptimizer:
    """Exact cached optimizer for a fixed ellipsoid union, linear form and box.

    Cache construction factors every ``A_FF`` once for each free-coordinate
    mask.  A call to :meth:`solve` then evaluates every lower/upper assignment
    belonging to that mask in NumPy batches across all active ellipsoids.
    No inverse is formed; every cached quantity comes from a linear solve.
    """

    def __init__(
        self,
        certificate,
        vector: np.ndarray,
        lower: np.ndarray | float = 0.0,
        upper: np.ndarray | float = 1.0,
        tolerance: float = 1e-10,
    ) -> None:
        vector = np.asarray(vector, dtype=np.float64)
        centers = np.asarray(certificate.centers, dtype=np.float64)
        matrices = np.asarray(certificate.matrices, dtype=np.float64)
        if centers.ndim != 2 or matrices.shape != (len(centers), centers.shape[1], centers.shape[1]):
            raise ValueError("certificate centers/matrices have incompatible dimensions")
        dimension = centers.shape[1]
        if vector.shape != (dimension,):
            raise ValueError("linear vector dimension disagrees with certificate input dimension")
        lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (dimension,)).copy()
        upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (dimension,)).copy()
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower > upper):
            raise ValueError("box bounds must be finite with lower <= upper")
        positions = np.flatnonzero(np.asarray(certificate.active, dtype=bool))
        if not len(positions):
            raise ValueError("cannot optimize over an empty compiled ellipsoid union")
        radii2 = np.asarray(certificate.squared_radii, dtype=np.float64)[positions]
        if np.any(radii2 < -tolerance):
            raise ValueError("an active compiled ellipsoid has negative squared radius")
        expert_indices = np.asarray(certificate.expert_indices, dtype=int)
        # Preserve the certificate mapping convention exactly.
        if not np.array_equal(expert_indices, positions):
            expert_indices = positions

        self.vector = vector
        self.lower, self.upper, self.tolerance = lower, upper, float(tolerance)
        self.centers, self.matrices = centers[positions], matrices[positions]
        self.radii2 = np.maximum(radii2, 0.0)
        self.positions, self.expert_indices = positions, expert_indices
        self.dimension = dimension
        self._cache = self._build_cache()

    def _build_cache(self) -> tuple[_BoxFaceCache, ...]:
        caches: list[_BoxFaceCache] = []
        q, d = len(self.positions), self.dimension
        for free_mask in product((False, True), repeat=d):
            free = np.flatnonzero(np.asarray(free_mask, dtype=bool))
            fixed = np.flatnonzero(~np.asarray(free_mask, dtype=bool))
            # Assignment bit 0 is the lower face (state 1), 1 is upper (state 2).
            bits = np.asarray(list(product((0, 1), repeat=len(fixed))), dtype=np.int8)
            if len(fixed) == 0:
                bits = np.empty((1, 0), dtype=np.int8)
            faces = np.zeros((len(bits), d), dtype=np.int8)
            if len(fixed):
                faces[:, fixed] = bits + 1
                fixed_values = np.where(bits == 0, self.lower[fixed], self.upper[fixed]).astype(np.float64)
            else:
                fixed_values = np.empty((1, 0), dtype=np.float64)
            if not len(free):
                caches.append(_BoxFaceCache(free, fixed, fixed_values, faces, None, None, None, None, None))
                continue
            aff = self.matrices[:, free][:, :, free]
            cholesky = np.linalg.cholesky(aff)
            def _chol_solve(rhs: np.ndarray) -> np.ndarray:
                return np.linalg.solve(np.swapaxes(cholesky, -1, -2), np.linalg.solve(cholesky, rhs))
            rhs = np.broadcast_to(self.vector[free], (q, len(free)))[..., None]
            reduced_vector = _chol_solve(rhs)[..., 0]
            support = np.einsum("i,qi->q", self.vector[free], reduced_vector)
            if np.any(support < -1e-11):
                raise ValueError("compiled ellipsoid matrix is not positive definite")
            support = np.maximum(support, 0.0)
            if len(fixed):
                afb = self.matrices[:, free][:, :, fixed]
                solve_fixed_block = _chol_solve(afb)
                abf = self.matrices[:, fixed][:, :, free]
                abb = self.matrices[:, fixed][:, :, fixed]
                schur = abb - np.matmul(abf, solve_fixed_block)
            else:
                solve_fixed_block = schur = None
            caches.append(_BoxFaceCache(free, fixed, fixed_values, faces, cholesky, reduced_vector, support, solve_fixed_block, schur))
        return tuple(caches)

    @property
    def cache_entry_count(self) -> int:
        return len(self._cache)

    def solve(self) -> BoxEllipsoidLinearOptimum:
        """Return the globally exact box-constrained optimum of this cache."""
        q, d, tolerance = len(self.positions), self.dimension, self.tolerance
        best_value = np.inf
        best: tuple[np.ndarray, int, int, np.ndarray] | None = None
        feasible_total = 0
        for cached in self._cache:
            free, fixed, face_values = cached.free, cached.fixed, cached.fixed_values
            assignments = len(face_values)
            point = np.empty((q, assignments, d), dtype=np.float64)
            if len(fixed):
                point[:, :, fixed] = face_values[None]
                delta_fixed = face_values[None] - self.centers[:, None, fixed]
            else:
                delta_fixed = np.empty((q, assignments, 0), dtype=np.float64)
            if not len(free):
                reduced_radius2 = self.radii2[:, None] - np.einsum(
                    "qmi,qij,qmj->qm", delta_fixed, self.matrices, delta_fixed,
                )
                valid = reduced_radius2 >= -tolerance
            else:
                assert cached.reduced_vector is not None and cached.support is not None
                if len(fixed):
                    assert cached.solve_fixed_block is not None and cached.schur is not None
                    shift = np.einsum("qfb,qmb->qmf", cached.solve_fixed_block, delta_fixed)
                    reduced_center = self.centers[:, None, free] - shift
                    reduced_radius2 = self.radii2[:, None] - np.einsum(
                        "qmi,qij,qmj->qm", delta_fixed, cached.schur, delta_fixed,
                    )
                else:
                    reduced_center = np.broadcast_to(self.centers[:, None, free], (q, assignments, len(free)))
                    reduced_radius2 = np.broadcast_to(self.radii2[:, None], (q, assignments))
                valid = reduced_radius2 >= -tolerance
                scale = np.zeros((q, assignments), dtype=np.float64)
                nonzero = cached.support > 0.0
                if np.any(nonzero):
                    # Feasibility of the slice is assignment-specific.  In
                    # particular, the all-lower assignment can be empty while
                    # another assignment of the same free mask is feasible.
                    scale[nonzero] = np.sqrt(
                        np.maximum(reduced_radius2[nonzero], 0.0) / cached.support[nonzero, None],
                    )
                point[:, :, free] = reduced_center - scale[:, :, None] * cached.reduced_vector[:, None, :]
                valid &= np.all(point[:, :, free] >= self.lower[free] - tolerance, axis=-1)
                valid &= np.all(point[:, :, free] <= self.upper[free] + tolerance, axis=-1)
            delta = point - self.centers[:, None, :]
            quadratic = np.einsum("qmi,qij,qmj->qm", delta, self.matrices, delta)
            valid &= quadratic <= self.radii2[:, None] + 10.0 * tolerance
            feasible_total += int(valid.sum())
            if not np.any(valid):
                continue
            values = np.einsum("qmi,i->qm", point, self.vector)
            values = np.where(valid, values, np.inf)
            flat = int(np.argmin(values)); expert_ordinal, assignment = np.unravel_index(flat, values.shape)
            value = float(values[expert_ordinal, assignment])
            if value < best_value:
                best_value = value
                best = (point[expert_ordinal, assignment].copy(), expert_ordinal, assignment, cached.faces[assignment].copy())
        if best is None:
            raise ValueError("the compiled ellipsoid union has empty intersection with the requested box")
        point, ordinal, assignment, face = best
        return BoxEllipsoidLinearOptimum(
            point=point, value=best_value, expert_index=int(self.expert_indices[ordinal]),
            active_position=int(self.positions[ordinal]), squared_radius=float(self.radii2[ordinal]),
            face=tuple(int(item) for item in face), faces_examined=int(q * (3 ** d)),
            feasible_face_candidates=feasible_total,
        )


def minimize_linear_over_ellipsoid_union(certificate, vector: np.ndarray) -> EllipsoidLinearOptimum:
    """Compute ``min_x vector @ x`` over a detached compiled ellipsoid union.

    ``certificate`` must expose the common frozen-certificate fields
    ``centers``, ``matrices``, ``active``, ``expert_indices`` and
    ``squared_radii``.  The computation uses linear solves, never an explicit
    inverse.  It is exact up to the supplied float64 certificate arithmetic.
    """
    vector = np.asarray(vector, dtype=np.float64)
    centers = np.asarray(certificate.centers, dtype=np.float64)
    matrices = np.asarray(certificate.matrices, dtype=np.float64)
    active_positions = np.flatnonzero(np.asarray(certificate.active, dtype=bool))
    if centers.ndim != 2 or matrices.shape != (len(centers), centers.shape[1], centers.shape[1]):
        raise ValueError("certificate centers/matrices have incompatible dimensions")
    if vector.shape != (centers.shape[1],):
        raise ValueError("linear vector dimension disagrees with certificate input dimension")
    if not len(active_positions):
        raise ValueError("cannot optimize over an empty compiled ellipsoid union")
    radii2 = np.asarray(certificate.squared_radii, dtype=np.float64)
    if radii2.shape != (len(centers),):
        raise ValueError("certificate squared radii have an incompatible shape")
    if np.any(radii2[active_positions] < -1e-12):
        raise ValueError("an active compiled ellipsoid has negative squared radius")

    best: EllipsoidLinearOptimum | None = None
    expert_indices = np.asarray(certificate.expert_indices, dtype=int)
    if not np.array_equal(expert_indices, active_positions):
        # Some certificates can expose a nontrivial mapping, but their active
        # positions still address their center/matrix/radius arrays.
        expert_indices = active_positions
    for ordinal, position in enumerate(active_positions):
        solution = np.linalg.solve(matrices[position], vector)
        support = float(vector @ solution)
        if support < -1e-11:
            raise ValueError("compiled ellipsoid matrix is not positive definite")
        support = max(support, 0.0)
        radius = float(np.sqrt(max(radii2[position], 0.0)))
        if support == 0.0 or radius == 0.0:
            point = centers[position].copy()
        else:
            point = centers[position] - radius * solution / np.sqrt(support)
        value = float(vector @ point)
        candidate = EllipsoidLinearOptimum(point, value, int(expert_indices[ordinal]), int(position), float(radii2[position]))
        if best is None or candidate.value < best.value:
            best = candidate
    assert best is not None
    return best


def minimize_linear_over_boxed_ellipsoid_union_reference(
    certificate,
    vector: np.ndarray,
    lower: np.ndarray | float = 0.0,
    upper: np.ndarray | float = 1.0,
    tolerance: float = 1e-10,
) -> BoxEllipsoidLinearOptimum:
    """Slow reference implementation of exact boxed ellipsoid optimization.

    The method exhaustively enumerates the ``3**d`` coordinate faces of every
    active ellipsoid.  On each face it analytically solves the linear objective
    over the corresponding Schur-complement ellipsoid slice, then retains only
    candidates whose free coordinates are inside the box.  It is therefore a
    globally exact finite procedure for fixed dimension, subject only to
    float64 arithmetic and ``tolerance``.
    """
    vector = np.asarray(vector, dtype=np.float64)
    centers = np.asarray(certificate.centers, dtype=np.float64)
    matrices = np.asarray(certificate.matrices, dtype=np.float64)
    if centers.ndim != 2 or matrices.shape != (len(centers), centers.shape[1], centers.shape[1]):
        raise ValueError("certificate centers/matrices have incompatible dimensions")
    dimension = centers.shape[1]
    if vector.shape != (dimension,):
        raise ValueError("linear vector dimension disagrees with certificate input dimension")
    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (dimension,)).copy()
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (dimension,)).copy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("box bounds must be finite with lower <= upper")
    active_positions = np.flatnonzero(np.asarray(certificate.active, dtype=bool))
    if not len(active_positions):
        raise ValueError("cannot optimize over an empty compiled ellipsoid union")
    radii2 = np.asarray(certificate.squared_radii, dtype=np.float64)
    expert_indices = np.asarray(certificate.expert_indices, dtype=int)
    if not np.array_equal(expert_indices, active_positions):
        expert_indices = active_positions

    best: BoxEllipsoidLinearOptimum | None = None
    faces_examined, feasible_face_candidates = 0, 0
    for ordinal, position in enumerate(active_positions):
        center, matrix, radius2 = centers[position], matrices[position], float(radii2[position])
        if radius2 < -tolerance:
            continue
        radius2 = max(radius2, 0.0)
        for face in product((0, 1, 2), repeat=dimension):
            faces_examined += 1
            fixed = np.flatnonzero(np.asarray(face) != 0)
            free = np.flatnonzero(np.asarray(face) == 0)
            point = np.empty(dimension, dtype=np.float64)
            if len(fixed):
                fixed_values = np.where(np.asarray(face)[fixed] == 1, lower[fixed], upper[fixed])
                point[fixed] = fixed_values
                delta_fixed = fixed_values - center[fixed]
            else:
                fixed_values = np.empty(0, dtype=np.float64)
                delta_fixed = np.empty(0, dtype=np.float64)
            if not len(free):
                quadratic = float((point - center) @ matrix @ (point - center))
                if quadratic > radius2 + tolerance:
                    continue
            elif not len(fixed):
                reduced_center, reduced_matrix, reduced_radius2 = center[free], matrix[np.ix_(free, free)], radius2
                solution = np.linalg.solve(reduced_matrix, vector[free])
                support = max(float(vector[free] @ solution), 0.0)
                if support == 0.0 or reduced_radius2 == 0.0:
                    point[free] = reduced_center
                else:
                    point[free] = reduced_center - np.sqrt(reduced_radius2 / support) * solution
            else:
                aff = matrix[np.ix_(free, fixed)] @ delta_fixed
                reduced_matrix = matrix[np.ix_(free, free)]
                reduced_center = center[free] - np.linalg.solve(reduced_matrix, aff)
                schur = matrix[np.ix_(fixed, fixed)] - matrix[np.ix_(fixed, free)] @ np.linalg.solve(reduced_matrix, matrix[np.ix_(free, fixed)])
                reduced_radius2 = radius2 - float(delta_fixed @ schur @ delta_fixed)
                if reduced_radius2 < -tolerance:
                    continue
                reduced_radius2 = max(reduced_radius2, 0.0)
                solution = np.linalg.solve(reduced_matrix, vector[free])
                support = max(float(vector[free] @ solution), 0.0)
                if support == 0.0 or reduced_radius2 == 0.0:
                    point[free] = reduced_center
                else:
                    point[free] = reduced_center - np.sqrt(reduced_radius2 / support) * solution
            if np.any(point < lower - tolerance) or np.any(point > upper + tolerance):
                continue
            quadratic = float((point - center) @ matrix @ (point - center))
            if quadratic > radius2 + 10 * tolerance:
                continue
            feasible_face_candidates += 1
            candidate = BoxEllipsoidLinearOptimum(
                point=point,
                value=float(vector @ point),
                expert_index=int(expert_indices[ordinal]),
                active_position=int(position),
                squared_radius=radius2,
                face=tuple(int(item) for item in face),
                faces_examined=faces_examined,
                feasible_face_candidates=feasible_face_candidates,
            )
            if best is None or candidate.value < best.value:
                best = candidate
    if best is None:
        raise ValueError("the compiled ellipsoid union has empty intersection with the requested box")
    return BoxEllipsoidLinearOptimum(
        point=best.point,
        value=best.value,
        expert_index=best.expert_index,
        active_position=best.active_position,
        squared_radius=best.squared_radius,
        face=best.face,
        faces_examined=faces_examined,
        feasible_face_candidates=feasible_face_candidates,
    )


def minimize_linear_over_boxed_ellipsoid_union(
    certificate,
    vector: np.ndarray,
    lower: np.ndarray | float = 0.0,
    upper: np.ndarray | float = 1.0,
    tolerance: float = 1e-10,
) -> BoxEllipsoidLinearOptimum:
    """Exactly minimize a linear form over a boxed ellipsoid union, cached path.

    This convenience function constructs a cache for one query.  Repeated
    queries against one frozen certificate should construct
    :class:`BoxedEllipsoidLinearOptimizer` once and call :meth:`solve`.
    """
    return BoxedEllipsoidLinearOptimizer(certificate, vector, lower, upper, tolerance).solve()
