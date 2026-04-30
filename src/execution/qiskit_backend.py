from __future__ import annotations

from typing import Any, Dict, List

from qiskit_aer import Aer

from src.circuit_builder import get_circuit_builder
from src.circuits.visualize import save_qiskit_circuit_image
from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec


def run_qiskit(
    output_dir: str,
    spec: CircuitSpec,
    circuit_config: CircuitConfig,
    backend_config: BackendConfig,
    evidence: Dict[str, int],
    query: List[str],
    save_circuit_image: bool = False,
) -> Dict[str, Any]:
    builder = get_circuit_builder("qiskit")
    circuit = builder.build(
        spec=spec,
        evidence=evidence,
        query=query,
        circuit_config=circuit_config,
        backend_config=backend_config,
    )
    backend_name = backend_config.device or "aer_simulator"
    backend = Aer.get_backend(backend_name)
    circuit = circuit.copy()
    circuit = circuit.decompose(reps=1)
    circuit = circuit.bind_parameters({})

    circuit_stats = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "operation_counts": dict(circuit.count_ops()),
        "total_operations": sum(circuit.count_ops().values()),
    }
    circuit_image = None
    if save_circuit_image:
        circuit_image = save_qiskit_circuit_image(output_dir, circuit)

    job = backend.run(circuit, shots=circuit_config.shots)
    result = job.result()
    counts = result.get_counts()
    total_shots = sum(counts.values())
    probs = {key: val / total_shots for key, val in counts.items()}

    return {
        "counts": counts,
        "probs": probs,
        "backend": backend_name,
        "circuit_stats": circuit_stats,
        "circuit_image": circuit_image,
    }
