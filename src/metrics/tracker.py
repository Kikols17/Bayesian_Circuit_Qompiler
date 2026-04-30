from __future__ import annotations

from typing import Any, Dict

from src.utils.io import ensure_dir, write_json


def save_metrics(output_dir: str, payload: Dict[str, Any]) -> str:
    ensure_dir(output_dir)
    path = f"{output_dir}/metrics.json"
    write_json(path, payload)
    return path
