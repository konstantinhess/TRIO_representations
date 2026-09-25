from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_config(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
