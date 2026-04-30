from __future__ import annotations

from typing import Dict, List

import pennylane as qml
from qiskit import QuantumCircuit

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec

from .base import CircuitBuilderBase


class QiskitCircuitBuilder(CircuitBuilderBase):
    name = "qiskit"

    def build(
        self,
        spec: CircuitSpec,
        evidence: Dict[str, int],
        query: List[str],
        circuit_config: CircuitConfig,
        backend_config: BackendConfig,
    ) -> QuantumCircuit:
        with qml.tape.QuantumTape() as tape:
            spec.circuit_fn(spec.wires_map, evidence, query)

        query_wires = spec.metadata.get("query_wires")
        if not query_wires:
            query_wires = list(range(spec.num_wires))

        qc = QuantumCircuit(spec.num_wires, len(query_wires))
        for op in tape.operations:
            name = op.name
            wires = list(op.wires)

            if name == "PauliX":
                qc.x(wires[0])
            elif name == "PauliZ":
                qc.z(wires[0])
            elif name == "Hadamard":
                qc.h(wires[0])
            elif name == "PhaseShift":
                qc.p(float(op.parameters[0]), wires[0])
            elif name == "RY":
                qc.ry(float(op.parameters[0]), wires[0])
            elif name == "CRY":
                qc.cry(float(op.parameters[0]), wires[0], wires[1])
            elif name.startswith("C("):
                # PennyLane sometimes represents controlled ops as 'C(RY)'
                # with the wrapped op available as `op.base` and control
                # wires in `op.control_wires`.
                base = getattr(op, "base", None)
                if base is None:
                    raise ValueError(f"Unsupported controlled op structure for Qiskit conversion: {name}")

                if base.name != "RY":
                    raise ValueError(
                        f"Unsupported controlled base op for Qiskit conversion: {base.name}"
                    )
                control_wires = list(op.control_wires) if hasattr(op, "control_wires") else []
                target_wire = list(base.wires)[0]
                angle = float(base.parameters[0])
                if len(control_wires) == 1:
                    qc.cry(angle, control_wires[0], target_wire)
                else:
                    qc.mcry(angle, control_wires, target_wire)
            elif name == "MultiControlledX":
                qc.mcx(wires[:-1], wires[-1])
            elif name == "Controlled":
                base = op.base
                if base.name != "RY":
                    raise ValueError(
                        f"Unsupported controlled base op for Qiskit conversion: {base.name}"
                    )
                control_wires = list(op.control_wires)
                target_wire = list(base.wires)[0]
                angle = float(base.parameters[0])
                if len(control_wires) == 1:
                    qc.cry(angle, control_wires[0], target_wire)
                else:
                    qc.mcry(angle, control_wires, target_wire)
            else:
                raise ValueError(f"Unsupported PennyLane op for Qiskit conversion: {name}")

        for idx, wire in enumerate(query_wires):
            qc.measure(wire, idx)

        return qc
