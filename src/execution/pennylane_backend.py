from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pennylane as qml

from src.circuit_builder import get_circuit_builder
from src.circuits.visualize import save_pennylane_circuit_image
from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec


def run_pennylane(
    output_dir: str,
    spec: CircuitSpec,
    circuit_config: CircuitConfig,
    backend_config: BackendConfig,
    evidence: Dict[str, int],
    query: List[str],
    save_circuit_image: bool = False,
) -> Dict[str, Any]:
    device_name = backend_config.device or "default.qubit"
    builder = get_circuit_builder("pennylane")
    qnode = builder.build(
        spec=spec,
        evidence=evidence,
        query=query,
        circuit_config=circuit_config,
        backend_config=backend_config,
    )
    result = qnode()

    def _decode_binary_state(sample, wires):
        if not wires:
            return 0
        bits = [str(int(sample[w])) for w in wires]
        return int("".join(bits), 2)

    def _samples_to_distribution(samples, wires_map, query, evidence):
        if samples.ndim != 2:
            raise ValueError("Expected sample output with shape (shots, num_wires)")

        query_nodes = query if query else list(wires_map.keys())
        query_wires = [wire for node in query_nodes for wire in wires_map[node]]
        evidence_nodes = list(evidence.keys())

        counts = {}
        evidence_matches = 0
        for sample in samples:
            matches = True
            for node in evidence_nodes:
                node_value = _decode_binary_state(sample, wires_map[node])
                if node_value != evidence[node]:
                    matches = False
                    break
            if not matches:
                continue

            evidence_matches += 1
            idx = _decode_binary_state(sample, query_wires)
            counts[idx] = counts.get(idx, 0) + 1

        if evidence_matches == 0:
            return [0.0] * (2 ** len(query_wires))

        total = float(evidence_matches)
        max_index = 2 ** len(query_wires)
        distribution = [0.0] * max_index
        for idx, count in counts.items():
            if 0 <= idx < max_index:
                distribution[idx] = count / total
        return distribution

    probs = None
    if isinstance(result, list):
        probs = result
    elif isinstance(result, np.ndarray) and result.ndim == 2:
        probs = _samples_to_distribution(result, spec.wires_map, query, evidence)
    else:
        probs = result.tolist() if hasattr(result, "tolist") else list(result)

    device = qnode.device
    if hasattr(device, "wires"):
        num_wires = len(device.wires)
    elif hasattr(device, "_wires"):
        num_wires = len(device._wires)
    else:
        num_wires = None

    tape = getattr(qnode, "qtape", None)
    if tape is None:
        tape = getattr(qnode, "_tape", None)
    if tape is None and hasattr(qnode, "tape"):
        tape = getattr(qnode, "tape")

    operations = list(tape.operations) if tape is not None else []
    circuit_stats = {
        "num_wires": num_wires,
        "num_operations": len(operations),
        "operation_counts": {
            op.name: sum(1 for o in operations if o.name == op.name)
            for op in operations
        },
    }

    circuit_image = None
    if save_circuit_image:
        circuit_image = save_pennylane_circuit_image(output_dir, qnode)

    return {
        "probs": probs,
        "raw": result,
        "device": device_name,
        "circuit_stats": circuit_stats,
        "circuit_image": circuit_image,
    }
