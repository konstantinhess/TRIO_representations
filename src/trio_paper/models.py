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


def build_trio(backbone: str, experts: int, input_dim: int, radial_units: int | None = None):
    if backbone == "broken_power":
        return BrokenPowerTRIO(experts, input_dim).double()
    if backbone == "wide_tanh":
        if radial_units is None: raise ValueError("wide_tanh requires radial_units")
        return WideTanhTRIO(experts, radial_units, input_dim).double()
    raise ValueError(f"unknown TRIO radial backbone: {backbone}")
