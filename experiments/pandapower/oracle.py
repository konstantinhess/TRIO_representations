"""Physical data/oracle pipeline for the pandapower renewable-integration study.

This module intentionally contains no learned predictor, inverse procedure, or optimizer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as ppt
from scipy.stats import qmc


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value)!r} as JSON")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG_PATH).open(encoding="utf-8") as handle:
        return json.load(handle)


def package_metadata() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "pandapower": pp.__version__,
    }


def base_network() -> pp.pandapowerNet:
    """Return the canonical pandapower IEEE 30-bus case without modifications."""
    return pn.case30()


def require_finite_bus_voltage_limits(net: pp.pandapowerNet) -> None:
    active = net.bus.index[net.bus.in_service.astype(bool)]
    limits = net.bus.loc[active, ["min_vm_pu", "max_vm_pu"]]
    valid = np.isfinite(limits.to_numpy(dtype=float)).all(axis=1) & (limits["min_vm_pu"] > 0).to_numpy() & (
        limits["max_vm_pu"] > limits["min_vm_pu"]
    ).to_numpy()
    if not bool(np.all(valid)):
        bad = limits.index[~valid].tolist()
        raise ValueError(f"Frozen network lacks valid finite voltage limits at buses {bad}")


def select_renewable_buses(net: pp.pandapowerNet, count: int = 5) -> list[int]:
    """Choose spatially distributed, non-slack existing-load buses deterministically."""
    slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service.astype(bool), "bus"].astype(int).tolist())
    candidates = sorted(set(net.load.loc[net.load.in_service.astype(bool), "bus"].astype(int).tolist()) - slack_buses)
    if len(candidates) < count:
        raise ValueError(f"Need {count} renewable candidates, only found {len(candidates)}")
    graph = ppt.create_nxgraph(net, respect_switches=True, include_lines=True, include_trafos=True)
    # Start at the highest-demand candidate: it is a deterministic network/load property and makes
    # renewable integration physically relevant to the existing demand pattern. Subsequent greedy
    # graph farthest-point choices preserve spatial distribution.
    demand_by_bus = net.load.loc[net.load.in_service.astype(bool)].groupby("bus")["p_mw"].sum()
    max_demand = float(demand_by_bus.loc[candidates].max())
    selected = [min(bus for bus in candidates if float(demand_by_bus.loc[bus]) == max_demand)]
    while len(selected) < count:
        scores: dict[int, int] = {}
        for candidate in candidates:
            if candidate in selected:
                continue
            scores[candidate] = min(nx.shortest_path_length(graph, candidate, previous) for previous in selected)
        best_score = max(scores.values())
        selected.append(min(bus for bus, score in scores.items() if score == best_score))
    return selected


def add_renewable_sgens(net: pp.pandapowerNet, buses: list[int], q_mvar: float = 0.0) -> list[int]:
    """Add one controllable renewable static generator at each frozen selected bus."""
    indices: list[int] = []
    for ordinal, bus in enumerate(buses):
        indices.append(
            int(
                pp.create_sgen(
                    net,
                    bus=bus,
                    p_mw=0.0,
                    q_mvar=q_mvar,
                    name=f"renewable_control_{ordinal}_bus_{bus}",
                    controllable=False,
                    in_service=True,
                )
            )
        )
    return indices


@dataclass(frozen=True)
class OracleResult:
    converged: bool
    f: float | None
    feasible: bool
    total_renewable_mw: float
    line_loading_ratio: float | None
    transformer_loading_ratio: float | None
    voltage_upper_ratio: float | None
    voltage_lower_ratio: float | None
    limiting_constraint_type: str | None
    limiting_constraint_id: int | None
    error: str | None = None
    min_bus_voltage_pu: float | None = None
    max_bus_voltage_pu: float | None = None


class RenewableOracle:
    """Deterministic AC power-flow oracle on a frozen IEEE-30 network instance."""

    def __init__(self, net: pp.pandapowerNet, renewable_buses: list[int], bounds_mw: np.ndarray, config: dict[str, Any]):
        require_finite_bus_voltage_limits(net)
        self.net = copy.deepcopy(net)
        self.renewable_buses = [int(bus) for bus in renewable_buses]
        self.bounds_mw = np.asarray(bounds_mw, dtype=float)
        if self.bounds_mw.shape != (len(renewable_buses), 2):
            raise ValueError("bounds_mw must be a d-by-2 array")
        self.config = config
        self.sgen_indices = add_renewable_sgens(self.net, self.renewable_buses, config["renewables"]["q_mvar"])

    @property
    def dimension(self) -> int:
        return len(self.renewable_buses)

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"Expected an input of shape ({self.dimension},), received {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Renewable injections must be finite")
        lower, upper = self.bounds_mw[:, 0], self.bounds_mw[:, 1]
        if np.any(x < lower - 1e-12) or np.any(x > upper + 1e-12):
            raise ValueError("Renewable injections fall outside frozen physical bounds")
        return x

    def evaluate(self, x: np.ndarray) -> OracleResult:
        x = self._validate_input(x)
        self.net.sgen.loc[self.sgen_indices, "p_mw"] = x
        pf = self.config["network"]["power_flow"]
        try:
            pp.runpp(
                self.net,
                algorithm=pf["algorithm"],
                init=pf["init"],
                max_iteration=pf["max_iteration"],
                numba=pf["numba"],
                recycle=None,
            )
        except Exception as error:  # pandapower may raise several convergence exceptions.
            return OracleResult(False, None, False, float(x.sum()), None, None, None, None, None, None, f"{type(error).__name__}: {error}")
        if not bool(self.net.converged):
            return OracleResult(False, None, False, float(x.sum()), None, None, None, None, None, None, "pandapower returned unconverged")

        candidates: list[tuple[float, str, int]] = []
        line_ratio: float | None = None
        trafo_ratio: float | None = None
        if len(self.net.res_line):
            lines = self.net.line.index[self.net.line.in_service.astype(bool)]
            values = self.net.res_line.loc[lines, "loading_percent"].astype(float) / 100.0
            if len(values):
                line_ratio = float(values.max())
                candidates.append((line_ratio, "line", int(values.idxmax())))
        if len(self.net.res_trafo):
            trafos = self.net.trafo.index[self.net.trafo.in_service.astype(bool)]
            values = self.net.res_trafo.loc[trafos, "loading_percent"].astype(float) / 100.0
            if len(values):
                trafo_ratio = float(values.max())
                candidates.append((trafo_ratio, "transformer", int(values.idxmax())))
        buses = self.net.bus.index[self.net.bus.in_service.astype(bool)]
        voltage = self.net.res_bus.loc[buses, "vm_pu"].astype(float)
        upper = self.net.bus.loc[buses, "max_vm_pu"].astype(float)
        lower = self.net.bus.loc[buses, "min_vm_pu"].astype(float)
        upper_values = voltage / upper
        lower_values = lower / voltage
        upper_ratio = float(upper_values.max())
        lower_ratio = float(lower_values.max())
        candidates.extend([(upper_ratio, "voltage_upper", int(upper_values.idxmax())), (lower_ratio, "voltage_lower", int(lower_values.idxmax()))])
        f, kind, identifier = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        return OracleResult(
            converged=True,
            f=float(f),
            feasible=bool(f <= 1.0),
            total_renewable_mw=float(x.sum()),
            line_loading_ratio=line_ratio,
            transformer_loading_ratio=trafo_ratio,
            voltage_upper_ratio=upper_ratio,
            voltage_lower_ratio=lower_ratio,
            limiting_constraint_type=kind,
            limiting_constraint_id=identifier,
            min_bus_voltage_pu=float(voltage.min()),
            max_bus_voltage_pu=float(voltage.max()),
        )


def sobol_box(bounds_mw: np.ndarray, n: int, seed: int) -> np.ndarray:
    bounds_mw = np.asarray(bounds_mw, dtype=float)
    sampler = qmc.Sobol(d=bounds_mw.shape[0], scramble=True, seed=seed)
    raw = sampler.random(n)
    return qmc.scale(raw, bounds_mw[:, 0], bounds_mw[:, 1])


def assign_splits(n: int, config: dict[str, Any]) -> np.ndarray:
    sizes = config["dataset"]["split_sizes"]
    if sum(sizes.values()) != n:
        raise ValueError("Configured split sizes must equal the dataset size")
    rng = np.random.default_rng(config["dataset"]["split_seed"])
    permutation = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    start = 0
    for label, size in sizes.items():
        labels[permutation[start : start + size]] = label
        start += size
    return labels


def normalization_metadata(bounds_mw: np.ndarray) -> dict[str, Any]:
    lower, upper = bounds_mw[:, 0], bounds_mw[:, 1]
    return {
        "kind": "x_normalized = (x - lower) / (upper - lower)",
        "lower_mw": lower.tolist(),
        "upper_mw": upper.tolist(),
        "inverse": "x = lower + x_normalized * (upper - lower)",
    }


def freeze_network(net: pp.pandapowerNet, output_dir: Path) -> dict[str, Any]:
    network_dir = output_dir / "network"
    network_dir.mkdir(parents=True, exist_ok=True)
    require_finite_bus_voltage_limits(net)
    pickle_path = network_dir / "case30_frozen.p"
    # pandapower's serializer currently expects a string rather than a Path.
    pp.to_pickle(net, str(pickle_path))
    summary = {
        "source": "pandapower.networks.case30",
        "source_label": "pandapower canonical IEEE 30-bus case",
        "tables": {name: int(len(getattr(net, name))) for name in ["bus", "line", "trafo", "load", "gen", "sgen", "ext_grid"]},
        "slack_buses": net.ext_grid.loc[net.ext_grid.in_service.astype(bool), "bus"].astype(int).tolist(),
        "total_load_p_mw": float(net.load.loc[net.load.in_service.astype(bool), "p_mw"].sum()),
        "total_load_q_mvar": float(net.load.loc[net.load.in_service.astype(bool), "q_mvar"].sum()),
        "line_max_i_ka": net.line["max_i_ka"].astype(float).tolist(),
        "bus_voltage_limits": net.bus[["min_vm_pu", "max_vm_pu"]].to_dict(orient="index"),
        "serialized_sha256": sha256_file(pickle_path),
    }
    write_json(network_dir / "network_summary.json", summary)
    return summary
