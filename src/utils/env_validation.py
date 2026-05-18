from __future__ import annotations

import os
from typing import Dict, Iterable


class MissingEnvError(RuntimeError):
    """Raised when one or more required environment variables are missing."""


def require_env(keys: Iterable[str]) -> Dict[str, str]:
    keys = list(keys)
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise MissingEnvError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return {k: os.environ[k] for k in keys}
