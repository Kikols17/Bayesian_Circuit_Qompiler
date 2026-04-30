"""Baseline inference components."""

from typing import Type

from .base import BaselineBase
from .gibbs_sampling import GibbsSamplingBaseline
from .variable_elimination import VariableEliminationBaseline


def get_baseline(name: str) -> BaselineBase:
    name_lower = name.strip().lower()
    registry: dict[str, Type[BaselineBase]] = {
        "variable_elimination": VariableEliminationBaseline,
        "gibbs_sampling": GibbsSamplingBaseline,
        "gibbs": GibbsSamplingBaseline,
        "ve": VariableEliminationBaseline,
    }
    if name_lower not in registry:
        raise ValueError(f"Unknown baseline method: {name}")
    return registry[name_lower]()
