from __future__ import annotations

from typing import Dict, List

import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec

from .base import CircuitBuilderBase


def _apply_state_prep(
    qc: QuantumCircuit,
    amps,
    target_wires: list,
    control_wires: list,
) -> None:
    """Encode amplitudes via Qiskit StatePreparation.

    Qiskit uses LSB-first qubit ordering; our convention is MSB-first.
    Reversing target_wires on append aligns the two conventions so that
    amplitude index idx maps correctly to the measured outcome idx.
    """
    arr = [float(abs(a)) for a in amps]
    total = sum(x ** 2 for x in arr)
    if total <= 0:
        return
    normalized = [x / total ** 0.5 for x in arr]
    sp = StatePreparation(normalized)
    if control_wires:
        sp = sp.control(len(control_wires))
        qc.append(sp, list(control_wires) + list(target_wires)[::-1])
    else:
        qc.append(sp, list(target_wires)[::-1])


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

        qc = QuantumCircuit(spec.num_wires, spec.num_wires)
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
            elif name == "AmplitudeEmbedding":
                _apply_state_prep(qc, op.parameters[0], wires, [])
            elif name.startswith("C("):
                base = getattr(op, "base", None)
                if base is None:
                    raise ValueError(f"Unsupported controlled op structure for Qiskit conversion: {name}")
                control_wires = list(op.control_wires) if hasattr(op, "control_wires") else []
                if base.name == "RY":
                    angle = float(base.parameters[0])
                    target_wire = list(base.wires)[0]
                    if len(control_wires) == 1:
                        qc.cry(angle, control_wires[0], target_wire)
                    else:
                        qc.mcry(angle, control_wires, target_wire)
                elif base.name == "AmplitudeEmbedding":
                    _apply_state_prep(qc, base.parameters[0], list(base.wires), control_wires)
                else:
                    raise ValueError(f"Unsupported controlled base op for Qiskit conversion: {base.name}")
            elif name == "MultiControlledX":
                qc.mcx(wires[:-1], wires[-1])
            elif name == "Controlled":
                base = op.base
                control_wires = list(op.control_wires)
                if base.name == "RY":
                    angle = float(base.parameters[0])
                    target_wire = list(base.wires)[0]
                    if len(control_wires) == 1:
                        qc.cry(angle, control_wires[0], target_wire)
                    else:
                        qc.mcry(angle, control_wires, target_wire)
                elif base.name == "AmplitudeEmbedding":
                    _apply_state_prep(qc, base.parameters[0], list(base.wires), control_wires)
                else:
                    raise ValueError(f"Unsupported controlled base op for Qiskit conversion: {base.name}")
            else:
                raise ValueError(f"Unsupported PennyLane op for Qiskit conversion: {name}")

        for i in range(spec.num_wires):
            qc.measure(i, i)

        return qc
