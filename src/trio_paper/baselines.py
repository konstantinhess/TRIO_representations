from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class SmoothMLP(nn.Module):
    def __init__(self, input_dim: int, widths: tuple[int, ...]):
        super().__init__()
        dims = (input_dim, *widths, 1)
        layers = []
        for i, (left, right) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.widths = tuple(widths)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class CPWLMLP(nn.Module):
    def __init__(self, input_dim: int, widths: tuple[int, ...]):
        super().__init__()
        dims = (input_dim, *widths, 1)
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.widths = tuple(widths)

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        return self.linears[-1](x).squeeze(-1)


class ICNN(nn.Module):
    """Two-hidden-layer scalar input-convex network."""
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.input_dim, self.width = input_dim, width
        self.input_first = nn.Linear(input_dim, width)
        self.hidden_input = nn.Linear(input_dim, width)
        self.hidden_raw_nonnegative = nn.Parameter(torch.empty(width, width))
        self.hidden_bias = nn.Parameter(torch.zeros(width))
        self.output_raw_nonnegative = nn.Parameter(torch.empty(width))
        self.output_input = nn.Linear(input_dim, 1)
        nn.init.normal_(self.hidden_raw_nonnegative, std=.05)
        nn.init.normal_(self.output_raw_nonnegative, std=.05)

    @property
    def hidden_nonnegative(self):
        return F.softplus(self.hidden_raw_nonnegative)

    @property
    def output_nonnegative(self):
        return F.softplus(self.output_raw_nonnegative)

    def forward(self, x):
        first = F.softplus(self.input_first(x))
        second = F.softplus(first @ self.hidden_nonnegative.T + self.hidden_input(x) + self.hidden_bias)
        return second @ self.output_nonnegative + self.output_input(x).squeeze(-1)
