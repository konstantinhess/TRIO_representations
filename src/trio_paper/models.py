"""Shared TRIO envelope implementation.

The spatial score of expert ``r`` is an SPD Mahalanobis distance.  A strictly
increasing radial law turns every scalar sublevel set into an ellipsoid; the
outer hard minimum therefore compiles exactly to a finite union of ellipsoids.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import brentq


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def _register_spd(module, experts: int, input_dim: int):
    module.input_dim = int(input_dim)
    module.raw_diag = nn.Parameter(torch.empty(experts, input_dim))
    module.offdiag = nn.Parameter(torch.zeros(experts) if input_dim == 2 else torch.zeros(experts, input_dim * (input_dim - 1) // 2))


def _matrices(module):
    diagonal = F.softplus(module.raw_diag) + module.epsilon
    q, d = len(module.beta), module.input_dim
    lower = torch.zeros((q, d, d), dtype=diagonal.dtype, device=diagonal.device)
    idx = torch.arange(d, device=diagonal.device)
    lower[:, idx, idx] = diagonal
    if d == 2:
        lower[:, 1, 0] = module.offdiag
    else:
        row, col = torch.tril_indices(d, d, -1, device=diagonal.device)
        lower[:, row, col] = module.offdiag
    return lower @ lower.transpose(-1, -2) + module.epsilon * torch.eye(d, dtype=diagonal.dtype, device=diagonal.device)


@dataclass(frozen=True)
class EllipsoidUnion:
    centers: np.ndarray
    matrices: np.ndarray
    active: np.ndarray
    squared_radii: np.ndarray
    target: float

    @property
    def expert_indices(self):
        return np.flatnonzero(self.active)

    def membership(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.active.any():
            return np.zeros(len(x), dtype=bool)
        dx = x[:, None] - self.centers[self.active][None]
        values = np.einsum("nqi,qij,nqj->nq", dx, self.matrices[self.active], dx)
        return (values <= self.squared_radii[self.active][None]).any(-1)


class _TRIOBase(nn.Module):
    def hard_forward(self, x):
        return self.values(x).amin(-1)

    def soft_forward(self, x, temperature: float):
        return -temperature * torch.logsumexp(-self.values(x) / temperature, -1)

    def forward(self, x):
        return self.hard_forward(x)

    def matrices(self):
        return _matrices(self)


class BrokenPowerTRIO(_TRIOBase):
    """C1 two-regime radial powers; nine parameters per 2-D expert."""
    def __init__(self, experts=128, input_dim=2, epsilon=1e-8):
        super().__init__(); self.epsilon = epsilon
        self.centers = nn.Parameter(torch.empty(experts, input_dim)); self.beta = nn.Parameter(torch.zeros(experts))
        _register_spd(self, experts, input_dim)
        self.raw_p1 = nn.Parameter(torch.empty(experts)); self.raw_p2 = nn.Parameter(torch.empty(experts)); self.raw_kappa = nn.Parameter(torch.empty(experts))
        nn.init.normal_(self.centers, std=.25)
        with torch.no_grad():
            self.raw_diag.copy_(inverse_softplus(torch.full((experts, input_dim), 1 - epsilon)))
            self.raw_p1.copy_(inverse_softplus(torch.ones(experts))); self.raw_p2.copy_(inverse_softplus(torch.ones(experts)))
            self.raw_kappa.copy_(inverse_softplus(torch.ones(experts)))

    @property
    def p1(self): return 1 + F.softplus(self.raw_p1)
    @property
    def p2(self): return 1 + F.softplus(self.raw_p2)
    @property
    def kappa(self): return F.softplus(self.raw_kappa) + self.epsilon

    def values(self, x):
        dx = x[:, None] - self.centers[None]
        distance = torch.sqrt(torch.einsum("nqi,qij,nqj->nq", dx, self.matrices(), dx).clamp_min(0))
        p1, p2, k = self.p1, self.p2, self.kappa
        at_break = .5 * k.pow(p1)
        inner = .5 * distance.pow(p1)
        outer = at_break + .5 * p1 / p2 * k.pow(p1 - p2) * (distance.pow(p2) - k.pow(p2))
        return self.beta + torch.where(distance <= k, inner, outer)

    @torch.no_grad()
    def freeze(self):
        return FrozenBrokenPower(self.centers.cpu().numpy().copy(), self.beta.cpu().numpy().copy(), self.matrices().cpu().numpy().copy(), self.p1.cpu().numpy().copy(), self.p2.cpu().numpy().copy(), self.kappa.cpu().numpy().copy())


@dataclass(frozen=True)
class FrozenBrokenPower:
    centers: np.ndarray; beta: np.ndarray; matrices: np.ndarray; p1: np.ndarray; p2: np.ndarray; kappa: np.ndarray
    def values(self, x):
        dx = np.asarray(x)[:, None] - self.centers[None]
        d = np.sqrt(np.maximum(np.einsum("nqi,qij,nqj->nq", dx, self.matrices, dx), 0))
        uk = .5 * self.kappa ** self.p1
        inner = .5 * d ** self.p1
        outer = uk + .5 * self.p1 / self.p2 * self.kappa ** (self.p1 - self.p2) * (d ** self.p2 - self.kappa ** self.p2)
        return self.beta + np.where(d <= self.kappa, inner, outer)
    def forward(self, x): return self.values(x).min(-1)
    def compile(self, target):
        u = target - self.beta; active = u >= 0; uk = .5 * self.kappa ** self.p1; d = np.zeros_like(u)
        inside = active & (u <= uk); outside = active & ~inside
        d[inside] = (2 * u[inside]) ** (1 / self.p1[inside])
        term = self.kappa[outside] ** self.p2[outside] + (2 * self.p2[outside] / self.p1[outside]) * self.kappa[outside] ** (self.p2[outside] - self.p1[outside]) * (u[outside] - uk[outside])
        d[outside] = np.maximum(term, 0) ** (1 / self.p2[outside])
        return EllipsoidUnion(self.centers.copy(), self.matrices.copy(), active, d * d, float(target))


def _rqs_forward(x, widths, heights, derivatives, bound):
    """Vectorized anchored rational-quadratic spline with identity tails."""
    inside = (x >= 0) & (x <= bound)
    xx = x.clamp(0, bound)
    cumulative_widths = torch.cat([torch.zeros_like(widths[:, :1]), widths.cumsum(-1)], -1)
    cumulative_heights = torch.cat([torch.zeros_like(heights[:, :1]), heights.cumsum(-1)], -1)
    index = (xx[..., None] >= cumulative_widths[None]).sum(-1).sub(1).clamp(0, widths.shape[-1] - 1)

    def gather(values, offset=0):
        expanded = values[None].expand(len(x), -1, -1)
        return torch.gather(expanded, 2, (index + offset)[..., None]).squeeze(-1)

    width = gather(widths)
    height = gather(heights)
    x_knot = gather(cumulative_widths)
    y_knot = gather(cumulative_heights)
    derivative_0 = gather(derivatives)
    derivative_1 = gather(derivatives, 1)
    theta = (xx - x_knot) / width
    slope = height / width
    one_minus_theta = 1 - theta
    numerator = height * (slope * theta.square() + derivative_0 * theta * one_minus_theta)
    denominator = slope + (derivative_1 + derivative_0 - 2 * slope) * theta * one_minus_theta
    return torch.where(inside, y_knot + numerator / denominator, x)


def _rqs_inverse_numpy(y, widths, heights, derivatives, bound):
    """Closed-form inverse of the anchored rational-quadratic spline."""
    y = np.asarray(y, dtype=np.float64)
    output = y.copy()
    inside = (y >= 0) & (y <= bound)
    cumulative_widths = np.concatenate([np.zeros((len(widths), 1)), np.cumsum(widths, axis=-1)], axis=-1)
    cumulative_heights = np.concatenate([np.zeros((len(heights), 1)), np.cumsum(heights, axis=-1)], axis=-1)
    for expert in range(y.shape[1]):
        rows = np.flatnonzero(inside[:, expert])
        if not len(rows):
            continue
        values = y[rows, expert]
        bins = np.clip(np.searchsorted(cumulative_heights[expert], values, side="right") - 1, 0, widths.shape[1] - 1)
        width = widths[expert, bins]
        height = heights[expert, bins]
        x_knot = cumulative_widths[expert, bins]
        y_knot = cumulative_heights[expert, bins]
        derivative_0 = derivatives[expert, bins]
        derivative_1 = derivatives[expert, bins + 1]
        slope = height / width
        shifted = values - y_knot
        a = shifted * (derivative_1 + derivative_0 - 2 * slope) + height * (slope - derivative_0)
        b = height * derivative_0 - shifted * (derivative_1 + derivative_0 - 2 * slope)
        c = -slope * shifted
        discriminant = np.maximum(b * b - 4 * a * c, 0)
        theta = np.where(np.abs(a) > 1e-14, (-b + np.sqrt(discriminant)) / (2 * a), -c / np.maximum(b, 1e-14))
        output[rows, expert] = x_knot + width * np.clip(theta, 0, 1)
    return output


class SplineTRIO(_TRIOBase):
    """Independent monotone rational-quadratic radial spline per expert."""
    def __init__(self, experts=128, bins=6, bound=4.0, input_dim=2, epsilon=1e-8):
        super().__init__()
        self.epsilon, self.bins, self.bound = epsilon, bins, bound
        self.centers = nn.Parameter(torch.empty(experts, input_dim))
        self.beta = nn.Parameter(torch.zeros(experts))
        _register_spd(self, experts, input_dim)
        self.raw_widths = nn.Parameter(torch.zeros(experts, bins))
        self.raw_heights = nn.Parameter(torch.zeros(experts, bins))
        self.raw_derivatives = nn.Parameter(inverse_softplus(torch.ones(experts, bins + 1)))
        nn.init.normal_(self.centers, std=.25)
        with torch.no_grad():
            self.raw_diag.copy_(inverse_softplus(torch.full((experts, input_dim), 1 - epsilon)))

    def spline_parameters(self):
        widths = F.softmax(self.raw_widths, -1) * self.bound
        heights = F.softmax(self.raw_heights, -1) * self.bound
        derivatives = F.softplus(self.raw_derivatives) + self.epsilon
        return widths, heights, derivatives

    def radial(self, distance):
        return _rqs_forward(distance, *self.spline_parameters(), self.bound)

    def values(self, x):
        dx = x[:, None] - self.centers[None]
        distance = torch.sqrt(torch.einsum("nqi,qij,nqj->nq", dx, self.matrices(), dx).clamp_min(0))
        return self.beta + self.radial(distance)

    @torch.no_grad()
    def freeze(self):
        widths, heights, derivatives = self.spline_parameters()
        return FrozenSpline(
            self.centers.cpu().numpy().copy(), self.beta.cpu().numpy().copy(), self.matrices().cpu().numpy().copy(),
            widths.cpu().numpy().copy(), heights.cpu().numpy().copy(), derivatives.cpu().numpy().copy(), self.bound,
        )


@dataclass(frozen=True)
class FrozenSpline:
    centers: np.ndarray; beta: np.ndarray; matrices: np.ndarray
    widths: np.ndarray; heights: np.ndarray; derivatives: np.ndarray; bound: float

    def values(self, x):
        dx = np.asarray(x)[:, None] - self.centers[None]
        distance = np.sqrt(np.maximum(np.einsum("nqi,qij,nqj->nq", dx, self.matrices, dx), 0))
        return self.beta + _rqs_forward(
            torch.as_tensor(distance), torch.as_tensor(self.widths), torch.as_tensor(self.heights),
            torch.as_tensor(self.derivatives), self.bound,
        ).numpy()

    def forward(self, x):
        return self.values(x).min(-1)

    def compile(self, target):
        residual = target - self.beta
        active = residual >= 0
        distance = np.zeros_like(residual)
        distance[active] = _rqs_inverse_numpy(
            residual[None], self.widths, self.heights, self.derivatives, self.bound,
        )[0, active]
        return EllipsoidUnion(self.centers.copy(), self.matrices.copy(), active, distance * distance, float(target))


class WideSoftplusTRIO(_TRIOBase):
    """Strictly monotone shallow Softplus radial network per expert."""
    def __init__(self, experts=128, units=8, input_dim=2, epsilon=1e-8):
        super().__init__()
        self.epsilon, self.units = epsilon, units
        self.centers = nn.Parameter(torch.empty(experts, input_dim))
        self.beta = nn.Parameter(torch.zeros(experts))
        _register_spd(self, experts, input_dim)
        self.raw_a = nn.Parameter(inverse_softplus(torch.full((experts,), .25)))
        self.raw_v = nn.Parameter(inverse_softplus(torch.full((experts, units), .02)))
        self.raw_w = nn.Parameter(inverse_softplus(torch.full((experts, units), 2.0)))
        self.raw_kappa = nn.Parameter(torch.empty(experts, units))
        nn.init.normal_(self.centers, std=.25)
        with torch.no_grad():
            self.raw_diag.copy_(inverse_softplus(torch.full((experts, input_dim), 1 - epsilon)))
            knots = torch.linspace(.05, 1.8, units)[None].repeat(experts, 1)
            self.raw_kappa.copy_(inverse_softplus(knots))

    @property
    def a(self): return F.softplus(self.raw_a) + self.epsilon
    @property
    def v(self): return F.softplus(self.raw_v)
    @property
    def w(self): return F.softplus(self.raw_w) + self.epsilon
    @property
    def kappa(self): return F.softplus(self.raw_kappa) + self.epsilon

    def radial(self, distance):
        z = self.w[None] * (distance[..., None] - self.kappa[None])
        z0 = -self.w * self.kappa
        return self.a[None] * distance + (self.v[None] * (F.softplus(z) - F.softplus(z0)[None])).sum(-1)

    def values(self, x):
        dx = x[:, None] - self.centers[None]
        distance = torch.sqrt(torch.einsum("nqi,qij,nqj->nq", dx, self.matrices(), dx).clamp_min(0))
        return self.beta + self.radial(distance)

    @torch.no_grad()
    def freeze(self):
        return FrozenWideSoftplus(
            self.centers.cpu().numpy().copy(), self.beta.cpu().numpy().copy(), self.matrices().cpu().numpy().copy(),
            self.a.cpu().numpy().copy(), self.v.cpu().numpy().copy(), self.w.cpu().numpy().copy(), self.kappa.cpu().numpy().copy(),
        )


@dataclass(frozen=True)
class FrozenWideSoftplus:
    centers: np.ndarray; beta: np.ndarray; matrices: np.ndarray
    a: np.ndarray; v: np.ndarray; w: np.ndarray; kappa: np.ndarray

    def radial(self, distance):
        distance = np.asarray(distance)
        z = self.w[None] * (distance[..., None] - self.kappa[None])
        z0 = -self.w * self.kappa
        return self.a[None] * distance + (self.v[None] * (np.logaddexp(0, z) - np.logaddexp(0, z0)[None])).sum(-1)

    def values(self, x):
        dx = np.asarray(x)[:, None] - self.centers[None]
        distance = np.sqrt(np.maximum(np.einsum("nqi,qij,nqj->nq", dx, self.matrices, dx), 0))
        return self.beta + self.radial(distance)

    def forward(self, x):
        return self.values(x).min(-1)

    def compile(self, target):
        active = self.beta <= target
        squared_radii = np.zeros_like(self.beta)
        for expert in np.flatnonzero(active):
            def equation(distance):
                value = self.a[expert] * distance + (
                    self.v[expert] * (
                        np.logaddexp(0, self.w[expert] * (distance - self.kappa[expert]))
                        - np.logaddexp(0, -self.w[expert] * self.kappa[expert])
                    )
                ).sum()
                return float(value - (target - self.beta[expert]))
            high = 1.0
            while equation(high) < 0:
                high *= 2
            distance = brentq(equation, 0., high, xtol=1e-12, rtol=1e-12)
            squared_radii[expert] = distance * distance
        return EllipsoidUnion(self.centers.copy(), self.matrices.copy(), active, squared_radii, float(target))


class WideTanhTRIO(_TRIOBase):
    """Strictly monotone neural radial law over ``log(1+d^2)``."""
    def __init__(self, experts=128, units=32, input_dim=2, derivative_floor=1e-4, epsilon=1e-8):
        super().__init__(); self.epsilon = epsilon; self.units = units; self.derivative_floor = derivative_floor
        self.centers = nn.Parameter(torch.empty(experts, input_dim)); self.beta = nn.Parameter(torch.zeros(experts)); _register_spd(self, experts, input_dim)
        self.raw_a = nn.Parameter(inverse_softplus(torch.full((experts,), .25)))
        self.raw_w = nn.Parameter(inverse_softplus(torch.full((experts, units), .01)))
        self.raw_s = nn.Parameter(inverse_softplus(torch.ones(experts, units)))
        self.t = nn.Parameter(torch.zeros(experts, units)); nn.init.normal_(self.centers, std=.25)
        with torch.no_grad(): self.raw_diag.copy_(inverse_softplus(torch.full((experts, input_dim), 1 - epsilon)))
    @property
    def a(self): return F.softplus(self.raw_a) + self.derivative_floor
    @property
    def w(self): return F.softplus(self.raw_w)
    @property
    def s(self): return F.softplus(self.raw_s) + self.epsilon
    def radial(self, d2):
        u = torch.log1p(d2.clamp_min(0)); z = self.s[None] * u[..., None] + self.t[None]
        return self.a[None] * u + (self.w[None] * (torch.tanh(z) - torch.tanh(self.t)[None])).sum(-1)
    def values(self, x):
        dx = x[:, None] - self.centers[None]
        d2 = torch.einsum("nqi,qij,nqj->nq", dx, self.matrices(), dx).clamp_min(0)
        return self.beta + self.radial(d2)
    @torch.no_grad()
    def freeze(self):
        return FrozenWideTanh(self.centers.cpu().numpy().copy(), self.beta.cpu().numpy().copy(), self.matrices().cpu().numpy().copy(), self.a.cpu().numpy().copy(), self.w.cpu().numpy().copy(), self.s.cpu().numpy().copy(), self.t.cpu().numpy().copy())


@dataclass(frozen=True)
class FrozenWideTanh:
    centers: np.ndarray; beta: np.ndarray; matrices: np.ndarray; a: np.ndarray; w: np.ndarray; s: np.ndarray; t: np.ndarray
    def _radial_one(self, index, u):
        return self.a[index] * u + np.sum(self.w[index] * (np.tanh(self.s[index] * u + self.t[index]) - np.tanh(self.t[index])))
    def values(self, x):
        dx = np.asarray(x)[:, None] - self.centers[None]
        d2 = np.maximum(np.einsum("nqi,qij,nqj->nq", dx, self.matrices, dx), 0)
        u = np.log1p(d2); z = self.s[None] * u[..., None] + self.t[None]
        radial = self.a[None] * u + (self.w[None] * (np.tanh(z) - np.tanh(self.t)[None])).sum(-1)
        return self.beta + radial
    def forward(self, x): return self.values(x).min(-1)
    def compile(self, target):
        active = self.beta <= target; rhs = np.zeros_like(self.beta)
        for i in np.flatnonzero(active):
            fun = lambda u: float(self._radial_one(i, u) - (target - self.beta[i]))
            high = 1.0
            while fun(high) < 0: high *= 2
            rhs[i] = np.expm1(brentq(fun, 0., high, xtol=1e-12, rtol=1e-12))
        return EllipsoidUnion(self.centers.copy(), self.matrices.copy(), active, rhs, float(target))


def build_trio(backbone: str, experts: int, input_dim: int, radial_units: int | None = None, spline_bins: int = 6):
    if backbone == "broken_power":
        return BrokenPowerTRIO(experts, input_dim).double()
    if backbone == "spline":
        return SplineTRIO(experts, spline_bins, input_dim=input_dim).double()
    if backbone == "wide_softplus":
        if radial_units is None: raise ValueError("wide_softplus requires radial_units")
        return WideSoftplusTRIO(experts, radial_units, input_dim).double()
    if backbone == "wide_tanh":
        if radial_units is None: raise ValueError("wide_tanh requires radial_units")
        return WideTanhTRIO(experts, radial_units, input_dim).double()
    raise ValueError(f"unknown TRIO radial backbone: {backbone}")
