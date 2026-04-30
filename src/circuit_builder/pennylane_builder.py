from __future__ import annotations

from typing import Dict, List

import pennylane as qml

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec

from .base import CircuitBuilderBase


class PennyLaneCircuitBuilder(CircuitBuilderBase):
    name = "pennylane"

    def build(
        self,
        spec: CircuitSpec,
        evidence: Dict[str, int],
        query: List[str],
        circuit_config: CircuitConfig,
        backend_config: BackendConfig,
    ):
        device_name = backend_config.device or "default.qubit"
        device = qml.device(
            device_name,
            wires=backend_config.wires or spec.num_wires,
            shots=circuit_config.shots,
        )

        def _circuit():
            return spec.circuit_fn(spec.wires_map, evidence, query)

        return qml.QNode(_circuit, device)
