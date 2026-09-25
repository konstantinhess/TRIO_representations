from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
import hashlib
import platform
from . import models as trio_models
import numpy as np
import torch


def temperature(step: int, soft_steps: int, start: float, end: float) -> float:
    fraction = min(max(step / max(soft_steps, 1), 0.0), 1.0)
    return end + .5 * (start - end) * (1.0 + math.cos(math.pi * fraction))


def _atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def train(model, train_x, train_y, validation_x, validation_y, output_dir, config, *, trio=False, seed=101):
    """Float64 Adam loop with shielded checkpoint selection and patience.

    Expensive runs are restart-safe at the completed-run level.  A completed
    ``metrics.json`` is never overwritten unless ``force`` is set in config.
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "seed": int(seed), "training_config": config,
        "training_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {
            "training": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "models": hashlib.sha256(Path(trio_models.__file__).read_bytes()).hexdigest(),
        },
    }
    _atomic_json(output_dir / "run_metadata.json", metadata)
    done = output_dir / "metrics.json"
    if done.exists() and not config.get("force", False):
        return json.loads(done.read_text())
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.double(); optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    train_x, train_y = train_x.double(), train_y.double(); validation_x, validation_y = validation_x.double(), validation_y.double()
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf"); best_step = None; last_qualifying = int(config["shield_steps"]); history = []
    epoch = cursor = 0
    order = np.random.default_rng(seed * 1_000_003 + epoch).permutation(len(train_x))
    start = time.perf_counter()
    max_steps, validation_every = int(config["max_steps"]), int(config["validation_every"])
    for step in range(max_steps + 1):
        if step % validation_every == 0:
            with torch.no_grad():
                val_loss = float(torch.mean((model(validation_x) - validation_y) ** 2))
            row = {"step": step, "validation_rmse": math.sqrt(val_loss), "elapsed_seconds": time.perf_counter() - start}
            history.append(row)
            if step >= int(config.get("checkpoint_start_step", config["shield_steps"])):
                if val_loss < best_loss:
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "config": config}, output_dir / "best.pt")
                    if val_loss <= best_loss * (1 - float(config["relative_min_delta"])):
                        last_qualifying = step
                    best_loss, best_step = val_loss, step
                if step - last_qualifying >= int(config["patience_steps"]):
                    break
        if step == max_steps:
            break
        if config.get("sampling", "with_replacement") == "epoch_shuffle":
            if cursor >= len(train_x):
                epoch += 1; cursor = 0; order = np.random.default_rng(seed * 1_000_003 + epoch).permutation(len(train_x))
            end = min(cursor + int(config["batch_size"]), len(train_x)); rows = torch.as_tensor(order[cursor:end], dtype=torch.long); cursor = end
        else:
            rows = torch.randint(len(train_x), (int(config["batch_size"]),), generator=generator)
        if trio and step < int(config["soft_steps"]):
            pred = model.soft_forward(train_x[rows], temperature(step, int(config["soft_steps"]), float(config["temperature_start"]), float(config["temperature_end"])))
        else:
            pred = model(train_x[rows])
        loss = torch.mean((pred - train_y[rows]) ** 2)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    torch.save({"model_state_dict": model.state_dict(), "step": step, "config": config}, output_dir / "final.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    result = {"best_step": best_step, "stopping_step": step, "best_validation_rmse": math.sqrt(best_loss), "training_seconds": time.perf_counter() - start}
    _atomic_json(done, result)
    return result
