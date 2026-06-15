from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a JSON object: {path}")
    return dict(data)


def expand_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(os.path.expandvars(str(value))).expanduser()


def resolve_fp16(value) -> bool:
    if isinstance(value, str) and value.lower() == "auto":
        import torch

        return torch.cuda.is_available()
    return bool(value)
