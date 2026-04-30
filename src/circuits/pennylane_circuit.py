from __future__ import annotations

from typing import Any, Dict, List

import pennylane as qml

from src.qompiler.base import CircuitSpec


def build_qnode(
    spec: CircuitSpec,
    evidence: Dict[str, int],
    query: List[str],
    device_name: str,
    shots: int,
    wires: int | None = None,
):
    device = qml.device(device_name, wires=wires or spec.num_wires, shots=shots)

    def _circuit():
        return spec.circuit_fn(spec.wires_map, evidence, query)

    return qml.QNode(_circuit, device)
