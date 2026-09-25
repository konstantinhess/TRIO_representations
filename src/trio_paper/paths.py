from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve_from_root(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value
