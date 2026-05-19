from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

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


def _greedy_distinguishing_bits(
    target_idx: int,
    others: Sequence[int],
    n: int,
    excluded_bit: int,
) -> List[int]:
    """Smallest set of bit positions (excluding `excluded_bit`) that, for each
    state in `others`, contains at least one bit where it differs from `target_idx`.

    Greedy set-cover: pick the bit that disambiguates the most remaining states,
    add it to the set, drop those states, repeat. Approximation ratio O(log k).
    """
    differ_sets: List[set] = []
    for s in others:
        bits = {
            i for i in range(n)
            if i != excluded_bit
            and ((s >> (n - 1 - i)) & 1) != ((target_idx >> (n - 1 - i)) & 1)
        }
        if bits:
            differ_sets.append(bits)

    chosen: List[int] = []
    while differ_sets:
        counts: Dict[int, int] = {}
        for ds in differ_sets:
            for b in ds:
                counts[b] = counts.get(b, 0) + 1
        best = max(counts, key=counts.get)
        chosen.append(best)
        differ_sets = [ds for ds in differ_sets if best not in ds]
    return chosen


def _apply_sparse_state_prep(
    qc: QuantumCircuit,
    indices: Sequence[int],
    amps: Sequence[float],
    target_wires: Sequence[int],
) -> None:
    """Prepare sum_j amps[j] |indices[j]> on target_wires from |0...0>.

    Convention: bit (n-1) of each index is interpreted as the value on
    target_wires[0] (MSB), bit 0 on target_wires[-1] (LSB).

    Algorithm: pairwise merging in U^† direction. At each step take the two
    active nodes with smallest amplitudes, find a differing bit position, and
    apply a multi-controlled Ry that maps the pair into a single basis state.
    Controls are chosen by greedy set-cover to be the smallest set that
    distinguishes the pair from all *other* active states — this keeps the
    multi-controlled rotation count from blowing up via Qiskit's noancilla
    expansion. After k-1 merges, X-gates map the surviving state to |0>. U is
    the reverse of those steps applied to |0>.
    """
    n = len(target_wires)
    if not indices:
        return

    nodes: List[Tuple[int, float]] = [(int(idx), float(amp)) for idx, amp in zip(indices, amps)]

    operations: List[Dict[str, object]] = []

    while len(nodes) > 1:
        nodes.sort(key=lambda t: abs(t[1]))
        idx_a, amp_a = nodes[0]
        idx_b, amp_b = nodes[1]
        others = [idx for idx, _ in nodes[2:]]

        diff_pos = next(
            i for i in range(n)
            if ((idx_a >> (n - 1 - i)) & 1) != ((idx_b >> (n - 1 - i)) & 1)
        )

        if ((idx_a >> (n - 1 - diff_pos)) & 1) == 0:
            idx_a, idx_b = idx_b, idx_a
            amp_a, amp_b = amp_b, amp_a

        chosen_bits = _greedy_distinguishing_bits(idx_a, others, n, excluded_bit=diff_pos)

        controls = [
            (target_wires[c], (idx_a >> (n - 1 - c)) & 1)
            for c in chosen_bits
        ]

        theta_forward = 2.0 * math.atan2(amp_a, amp_b)

        operations.append({
            "target_wire": target_wires[diff_pos],
            "theta_forward": theta_forward,
            "controls": controls,
        })

        nodes.pop(0)
        nodes.pop(0)
        merged_amp = math.sqrt(amp_a * amp_a + amp_b * amp_b)
        nodes.append((idx_b, merged_amp))

    final_idx = nodes[0][0]

    for i in range(n):
        if (final_idx >> (n - 1 - i)) & 1:
            qc.x(target_wires[i])

    for op in reversed(operations):
        controls = op["controls"]  # type: ignore[assignment]
        zero_ctrl_wires = [w for w, bit in controls if bit == 0]
        ctrl_wires = [w for w, _ in controls]
        target_wire = op["target_wire"]
        theta = float(op["theta_forward"])  # type: ignore[arg-type]
        for w in zero_ctrl_wires:
            qc.x(w)
        if ctrl_wires:
            qc.mcry(theta, ctrl_wires, target_wire)
        else:
            qc.ry(theta, target_wire)
        for w in zero_ctrl_wires:
            qc.x(w)


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

        sparse_state = spec.metadata.get("sparse_state")
        if sparse_state is not None:
            query_wires = list(spec.metadata["query_wires"])
            _apply_sparse_state_prep(
                qc,
                indices=sparse_state["indices"],
                amps=sparse_state["amps"],
                target_wires=query_wires,
            )

        for op in tape.operations:
            name = op.name
            wires = list(op.wires)

            if sparse_state is not None and name == "AmplitudeEmbedding":
                continue

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
