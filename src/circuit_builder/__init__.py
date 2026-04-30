"""Circuit builder implementations."""

from typing import Type

from .base import CircuitBuilderBase
from .pennylane_builder import PennyLaneCircuitBuilder
from .qiskit_builder import QiskitCircuitBuilder


def get_circuit_builder(name: str) -> CircuitBuilderBase:
    name_lower = name.strip().lower()
    registry: dict[str, Type[CircuitBuilderBase]] = {
        "pennylane": PennyLaneCircuitBuilder,
        "qiskit": QiskitCircuitBuilder,
    }
    if name_lower not in registry:
        raise ValueError(f"Unknown circuit builder: {name}")
    return registry[name_lower]()
