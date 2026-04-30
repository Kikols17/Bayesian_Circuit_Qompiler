from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value)]
    return str(value)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    safe_payload = _json_safe(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2, sort_keys=True)


def write_yaml(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
