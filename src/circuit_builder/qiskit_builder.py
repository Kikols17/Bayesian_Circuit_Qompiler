from __future__ import annotations

from typing import Dict, List

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec

from .base import CircuitBuilderBase


def _encode_amplitudes(
    qc: QuantumCircuit,
    amplitudes,
    target_wires: list,
    control_wires: list,
) -> None:
    """
    Encode an amplitude state using a recursive RY/CRY/MCRY decomposition.

    Replaces StatePreparation + .control() to avoid Qiskit's transpiler hanging
    on deep controlled state preparations. For n target qubits this emits
    O(2^n) multi-controlled RY gates which the transpiler handles efficiently.

    The X-MCRY-X sandwich used for 0-valued conditions is the same pattern
    already used by the 1- and 2-qubit analytical cases in the DCM compiler.
    """
    probs = [float(abs(a)) ** 2 for a in amplitudes]
    total = sum(probs)
    if total <= 0:
        return
    probs = [p / total for p in probs]
    _recursive_ry(qc, probs, list(target_wires), list(control_wires))


def _recursive_ry(qc: QuantumCircuit, probs: list, wires: list, ctrls: list) -> None:
    n = len(wires)
    if n == 0:
        return

    half = len(probs) // 2
    p_low = sum(probs[:half])
    p_high = sum(probs[half:])
    total = p_low + p_high
    if total <= 0:
        return

    angle = 2.0 * float(np.arcsin(np.sqrt(np.clip(p_high / total, 0.0, 1.0))))

    if abs(angle) > 1e-12:
        if not ctrls:
            qc.ry(angle, wires[0])
        elif len(ctrls) == 1:
            qc.cry(angle, ctrls[0], wires[0])
        else:
            qc.mcry(angle, ctrls, wires[0])

    if n == 1:
        return

    if p_low > 1e-12:
        cond_low = [p / p_low for p in probs[:half]]
        qc.x(wires[0])
        _recursive_ry(qc, cond_low, wires[1:], ctrls + [wires[0]])
        qc.x(wires[0])

    if p_high > 1e-12:
        cond_high = [p / p_high for p in probs[half:]]
        _recursive_ry(qc, cond_high, wires[1:], ctrls + [wires[0]])


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
                _encode_amplitudes(qc, op.parameters[0], wires, [])
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
                    _encode_amplitudes(qc, base.parameters[0], list(base.wires), control_wires)
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
                    _encode_amplitudes(qc, base.parameters[0], list(base.wires), control_wires)
                else:
                    raise ValueError(f"Unsupported controlled base op for Qiskit conversion: {base.name}")
            else:
                raise ValueError(f"Unsupported PennyLane op for Qiskit conversion: {name}")

        for i in range(spec.num_wires):
            qc.measure(i, i)

        return qc
