from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate

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
    excluded_bits: Sequence[int],
) -> List[int]:
    """Smallest set of bit positions (excluding `excluded_bits`) that, for each
    state in `others`, contains at least one bit where it differs from `target_idx`.

    Greedy set-cover: pick the bit that disambiguates the most remaining states,
    add it to the set, drop those states, repeat. Approximation ratio O(log k).
    Returns None if some `other` cannot be distinguished using the allowed bits.
    """
    excluded = set(excluded_bits)
    differ_sets: List[set] = []
    for s in others:
        bits = {
            i for i in range(n)
            if i not in excluded
            and ((s >> (n - 1 - i)) & 1) != ((target_idx >> (n - 1 - i)) & 1)
        }
        if not bits:
            return None  # type: ignore[return-value]
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

    Algorithm: pairwise merging in U^† direction. At each step pick the
    pair (a, b) with the smallest Hamming distance, ties broken by smaller
    amplitude sum; min-HD first because each unit of Hamming distance adds
    two bare CNOTs to the merge, so HD=1 pairs are cheapest. Let S be the
    bit positions where a and b differ and q = S[0] the pivot. The merge
    sandwiches a controlled Ry on q between two layers of *bare* CNOTs
    CX(q -> i) for i in S \\ {q}. The pre-CX layer flips bits in S \\ {q}
    on every state with q=1, compressing the pair's difference to bit q
    alone (and temporarily transforming other q=1 states). The Ry, gated
    by a control set that distinguishes the post-CX pair pattern (= b on
    all non-q bits) from every post-CX other, merges the pair into b. The
    post-CX layer flips bits back; for any state not touched by the Ry the
    two CNOT layers cancel, so other states are restored. Control selection
    uses greedy set-cover on the post-CX bit patterns, which lets bits in
    S \\ {q} be used as controls without colliding with the CNOT targets
    (the CXs are bare, not controlled, so target/control collision never
    arises).
    """
    n = len(target_wires)
    if not indices:
        return

    nodes: List[Tuple[int, float]] = [(int(idx), float(amp)) for idx, amp in zip(indices, amps)]

    operations: List[Dict[str, object]] = []

    while len(nodes) > 1:
        best_i = best_j = -1
        best_key = None
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                hd = bin(nodes[i][0] ^ nodes[j][0]).count("1")
                amp_sum = abs(nodes[i][1]) + abs(nodes[j][1])
                key = (hd, amp_sum)
                if best_key is None or key < best_key:
                    best_key = key
                    best_i, best_j = i, j

        if abs(nodes[best_j][1]) < abs(nodes[best_i][1]):
            best_i, best_j = best_j, best_i

        idx_a, amp_a = nodes[best_i]
        idx_b, amp_b = nodes[best_j]
        others = [t[0] for k, t in enumerate(nodes) if k != best_i and k != best_j]

        diff_bits = [
            i for i in range(n)
            if ((idx_a >> (n - 1 - i)) & 1) != ((idx_b >> (n - 1 - i)) & 1)
        ]
        diff_pos = diff_bits[0]

        if ((idx_a >> (n - 1 - diff_pos)) & 1) == 0:
            idx_a, idx_b = idx_b, idx_a
            amp_a, amp_b = amp_b, amp_a

        other_diff_bits = diff_bits[1:]

        q_mask = 1 << (n - 1 - diff_pos)
        s_minus_q_mask = 0
        for i in other_diff_bits:
            s_minus_q_mask |= 1 << (n - 1 - i)

        others_post_cx = [
            (c ^ s_minus_q_mask) if (c & q_mask) else c
            for c in others
        ]

        chosen_bits = _greedy_distinguishing_bits(
            idx_b, others_post_cx, n, excluded_bits=[diff_pos]
        )
        if chosen_bits is None:
            raise NotImplementedError(
                "Sparse SP merge: post-CX image of some other active state "
                "matches the pair on every non-pivot bit. The pair-merge "
                "rotation cannot fire selectively without an ancilla qubit. "
                f"Pair=({idx_a:0{n}b}, {idx_b:0{n}b}), q={diff_pos}, "
                f"others_post_cx={[f'{o:0{n}b}' for o in others_post_cx]}."
            )

        controls = [
            (target_wires[c], (idx_b >> (n - 1 - c)) & 1)
            for c in chosen_bits
        ]

        theta_forward = 2.0 * math.atan2(amp_a, amp_b)

        operations.append({
            "target_wire": target_wires[diff_pos],
            "theta_forward": theta_forward,
            "controls": controls,
            "other_diff_wires": [target_wires[i] for i in other_diff_bits],
        })

        hi, lo = (best_i, best_j) if best_i > best_j else (best_j, best_i)
        nodes.pop(hi)
        nodes.pop(lo)
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
        other_diff_wires = op["other_diff_wires"]  # type: ignore[assignment]

        for ow in other_diff_wires:
            qc.cx(target_wire, ow)

        for w in zero_ctrl_wires:
            qc.x(w)

        if ctrl_wires:
            qc.mcry(theta, ctrl_wires, target_wire)
        else:
            qc.ry(theta, target_wire)

        for w in zero_ctrl_wires:
            qc.x(w)

        for ow in other_diff_wires:
            qc.cx(target_wire, ow)


def _site_unitary(tensor: np.ndarray, b: int) -> np.ndarray:
    """Unitary on (1 physical + b ancilla) qubits implementing one MPS site.

    Sequential preparation (Schon 2005): the ancilla register carries the right
    bond. Acting on input basis (physical=|0>, ancilla=|beta>, beta<R) the gate
    outputs sum_{s,alpha<L} A[alpha,s,beta] |s>|alpha>. Columns are the
    left-canonical isometry (orthonormal); remaining columns are an arbitrary
    orthonormal completion (those inputs never occur during the run). Basis index
    is big-endian (physical = top bit, ancilla = low b bits).
    """
    L, _, R = tensor.shape
    dim = 2 * (2 ** b)
    filled = np.zeros((dim, R))
    used_cols = []
    for beta in range(R):
        col = beta  # input: physical=0, ancilla=beta
        used_cols.append(col)
        for s in range(2):
            for alpha in range(L):
                filled[(s << b) | alpha, beta] = tensor[alpha, s, beta]

    u_full, _, _ = np.linalg.svd(filled, full_matrices=True)
    completion = u_full[:, R:]

    unitary = np.zeros((dim, dim))
    for i, col in enumerate(used_cols):
        unitary[:, col] = filled[:, i]
    rem = [c for c in range(dim) if c not in used_cols]
    for j, col in enumerate(rem):
        unitary[:, col] = completion[:, j]
    return unitary


def _apply_mps_state_prep(
    qc: QuantumCircuit,
    tensors: List[np.ndarray],
    phys_wires: Sequence[int],
    anc_wires: Sequence[int],
) -> None:
    """Prepare a row-major MPS on phys_wires using anc_wires as the bond register.

    Sites are applied right-to-left so the ancilla, initialised |0>, threads the
    bond and returns to |0> after site 0 (left boundary dim 1). phys_wires[k] is
    site k, the k-th most significant index bit.
    """
    b = len(anc_wires)
    n = len(tensors)
    for k in range(n - 1, -1, -1):
        unitary = _site_unitary(tensors[k], b)
        big_endian = [phys_wires[k]] + list(anc_wires)
        qc.append(UnitaryGate(unitary), big_endian[::-1])


def _build_mps_circuit(spec: "CircuitSpec", mps_state: dict) -> QuantumCircuit:
    if mps_state.get("snake_perm") is not None:
        raise NotImplementedError(
            "MPS circuit synthesis supports order='row_major' only; 'snake' needs an "
            "in-circuit index permutation. row_major is also the better order empirically."
        )
    n = int(mps_state["num_qubits"])
    tensors = [np.asarray(t, dtype=float) for t in mps_state["tensors"]]
    max_bond = max([1] + [t.shape[0] for t in tensors] + [t.shape[2] for t in tensors])
    b = math.ceil(math.log2(max_bond)) if max_bond > 1 else 0
    qc = QuantumCircuit(n + b, n)
    _apply_mps_state_prep(qc, tensors, list(range(n)), list(range(n, n + b)))
    for i in range(n):
        qc.measure(i, i)
    return qc


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
        mps_state = spec.metadata.get("mps_state")
        if mps_state is not None:
            return _build_mps_circuit(spec, mps_state)

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
