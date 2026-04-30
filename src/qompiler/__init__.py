"""Qompiler implementations."""

from typing import Type

from .base import QompilerBase
from .dcm import DCMQompiler
from .qaa import QAAQompiler


def get_qompiler(name: str) -> QompilerBase:
    name_upper = name.strip().upper()
    registry: dict[str, Type[QompilerBase]] = {
        "DCM": DCMQompiler,
        "QAA": QAAQompiler,
    }
    if name_upper not in registry:
        raise ValueError(f"Unknown Qompiler: {name}")
    return registry[name_upper]()
