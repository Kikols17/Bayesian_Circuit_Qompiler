from __future__ import annotations

import math
from itertools import product as iproduct
from typing import Any, Dict, List, Tuple

import numpy as np
import pennylane as qml
from pgmpy.models import DiscreteBayesianNetwork

from .base import CircuitSpec, QompilerBase, build_wires_map, query_wires


def _build_conditional_probs(
    model: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: List[str],
) -> List[float]:
    """Compute P(query | evidence) from the model's CPD tables.

    Enumerates all query variable combinations, computes the unnormalized joint
    P(query_vals) * prod(P(e_k | query_vals)) for each, then normalizes.
    """
    query_cards = [int(model.get_cpds(node).variable_card) for node in query]
    num_outcomes = 1
    for c in query_cards:
        num_outcomes *= c

    unnorm: List[float] = [0.0] * num_outcomes

    for query_vals in iproduct(*[range(c) for c in query_cards]):
        full_state = {**dict(zip(query, query_vals)), **evidence}

        prob = 1.0
        for node in model.nodes():
            val = full_state.get(node)
            if val is None:
                continue
            cpd = model.get_cpds(node)
            parents = cpd.variables[1:]
            values = cpd.get_values()

            if not parents:
                prob *= float(values[val, 0])
            else:
                col = 0
                for parent in parents:
                    col = col * int(model.get_cpds(parent).variable_card) + full_state[parent]
                prob *= float(values[val, col])

        idx = 0
        for v, c in zip(query_vals, query_cards):
            idx = idx * c + v
        unnorm[idx] = prob

    total = sum(unnorm)
    if total == 0:
        return [1.0 / num_outcomes] * num_outcomes
    return [p / total for p in unnorm]


def _binary_qubit_count(card: int) -> int:
    return max(1, int(math.ceil(math.log2(card))))


def _marginalize_out(
    probs: List[float],
    query: List[str],
    query_cards: List[int],
    marginalize: List[str],
) -> Tuple[List[float], List[str], List[int]]:
    """Sum out marginalised variables from a row-major flat distribution.

    Returns (reduced_probs, kept_query, kept_cards) where the reduced_probs
    layout matches kept_query in the same MSB-first row-major order as the
    binary index built in `_build_conditional_probs`.
    """
    if not marginalize:
        return probs, query, query_cards

    kept = [v for v in query if v not in marginalize]
    kept_cards = [query_cards[query.index(v)] for v in kept]
    kept_size = 1
    for c in kept_cards:
        kept_size *= c

    reduced = [0.0] * kept_size
    for flat_idx, p in enumerate(probs):
        coords: List[int] = []
        rem = flat_idx
        for c in reversed(query_cards):
            coords.insert(0, rem % c)
            rem //= c

        kept_coords = [coords[query.index(v)] for v in kept]
        kept_flat = 0
        for v, c in zip(kept_coords, kept_cards):
            kept_flat = kept_flat * c + v
        reduced[kept_flat] += p

    return reduced, kept, kept_cards


def _binary_index_to_padded_index(
    binary_idx: int, kept_cards: List[int], padded_bits_per_var: List[int]
) -> int:
    """Map a row-major index over kept_cards to the row-major qubit-padded index.

    Each kept variable occupies ceil(log2(card)) qubits. The padded index is the
    integer whose binary representation is the concatenation of each variable's
    binary value, in query order, MSB-first.
    """
    coords: List[int] = []
    rem = binary_idx
    for c in reversed(kept_cards):
        coords.insert(0, rem % c)
        rem //= c

    padded = 0
    for value, bits in zip(coords, padded_bits_per_var):
        padded = (padded << bits) | value
    return padded


class QAAQompiler(QompilerBase):
    name = "QAA"

    def compile(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        encoding: str,
        encoding_params: Dict[str, Any],
    ) -> CircuitSpec:
        full_wires_map = build_wires_map(model, encoding=encoding)

        if evidence:
            # Pre-conditioned mode: compute P(query | evidence) classically and
            # encode it as amplitudes on the query qubits only. No evidence qubits
            # are allocated.
            #
            # Binary qubit layout is used regardless of which amplitude-encoding
            # variant the user requested. The encoding flag controls *how* the
            # amplitude vector is prepared (dense StatePreparation vs. sparse
            # tree-walk SP), not the qubit-to-state mapping. One-hot would
            # allocate sum(card) qubits and require a 2^sum-sized amplitude
            # vector — infeasible.
            if encoding == "one_hot":
                raise ValueError(
                    "QAA preconditioned mode does not support encoding='one_hot' "
                    "(would allocate sum(card) qubits with a 2^sum amplitude vector; "
                    "infeasible at scale). Use 'binary' or 'sparse_topk' instead."
                )

            query_cards = [int(model.get_cpds(node).variable_card) for node in query]

            if encoding == "sparse_topk":
                k = int(encoding_params["k"])
                marginalize = list(encoding_params["marginalize"])
            elif encoding == "binary":
                k = None
                marginalize = []
            else:
                raise ValueError(
                    f"QAA preconditioned mode does not support encoding={encoding!r}"
                )

            cond_probs = _build_conditional_probs(model, evidence, query)
            cond_probs, effective_query, effective_cards = _marginalize_out(
                cond_probs, query, query_cards, marginalize
            )

            _bin_map = build_wires_map(model, encoding="binary")
            q_map: Dict[str, List[int]] = {}
            w = 0
            for node in effective_query:
                bits = len(_bin_map[node])
                q_map[node] = list(range(w, w + bits))
                w += bits
            num_wires = w
            query_wire_list = [wire for node in effective_query for wire in q_map[node]]
            padded_bits_per_var = [len(_bin_map[node]) for node in effective_query]

            padded_size = 2 ** num_wires
            padded_probs = [0.0] * padded_size
            for binary_idx, p in enumerate(cond_probs):
                padded_idx = _binary_index_to_padded_index(
                    binary_idx, effective_cards, padded_bits_per_var
                )
                padded_probs[padded_idx] = p

            metadata: Dict[str, Any] = {
                "query_wires": query_wire_list,
                "encoding": encoding,
                "encoding_params": dict(encoding_params),
                "mode": "preconditioned",
                "effective_query": effective_query,
                "marginalized": marginalize,
            }

            if encoding == "sparse_topk":
                indexed = sorted(
                    enumerate(padded_probs), key=lambda t: t[1], reverse=True
                )
                top = indexed[:k]
                top = [(idx, p) for idx, p in top if p > 0.0]
                if not top:
                    raise ValueError(
                        "sparse_topk truncation produced an empty support set; "
                        "the conditional posterior is identically zero."
                    )
                total = sum(p for _, p in top)
                indices = [idx for idx, _ in top]
                amps = [math.sqrt(p / total) for _, p in top]

                metadata["sparse_state"] = {
                    "indices": indices,
                    "amps": amps,
                    "num_qubits": num_wires,
                    "k_requested": k,
                    "k_effective": len(top),
                    "truncated_mass": float(total),
                }

                truncated_padded = [0.0] * padded_size
                for idx, p in top:
                    truncated_padded[idx] = p / total

                def circuit_fn(
                    wires: Dict[str, List[int]],
                    evidence_map: Dict[str, int],
                    query_nodes: List[str],
                ):
                    amps_pl = [math.sqrt(p) for p in truncated_padded]
                    qml.AmplitudeEmbedding(amps_pl, wires=query_wire_list, normalize=True)
                    return qml.probs(wires=query_wire_list)

            else:
                def circuit_fn(
                    wires: Dict[str, List[int]],
                    evidence_map: Dict[str, int],
                    query_nodes: List[str],
                ):
                    amps_pl = [math.sqrt(p) for p in padded_probs]
                    qml.AmplitudeEmbedding(amps_pl, wires=query_wire_list, normalize=True)
                    return qml.probs(wires=query_wire_list)

            return CircuitSpec(
                name=self.name,
                num_wires=num_wires,
                wires_map=q_map,
                circuit_fn=circuit_fn,
                metadata=metadata,
            )

        if encoding == "sparse_topk":
            raise ValueError(
                "QAA oracle mode (no evidence) does not support encoding='sparse_topk'. "
                "Sparse top-k truncation only applies to the preconditioned posterior. "
                "Use 'binary' or 'one_hot' for oracle mode."
            )

        # Oracle mode (no evidence): prepare full joint state + Grover iterations.
        wires_map = full_wires_map
        num_wires = max(w for ws in wires_map.values() for w in ws) + 1
        query_wire_list = query_wires(wires_map, query)
        use_one_hot = encoding == "one_hot"
        num_iterations = max(1, round(math.pi / 4 * math.sqrt(2 ** num_wires)))
        metadata = {
            "query_wires": query_wire_list,
            "num_iterations": num_iterations,
            "encoding": encoding,
            "encoding_params": dict(encoding_params),
            "mode": "oracle",
        }

        def _apply_unconditional(probs: List[float], node_wires: List[int]) -> None:
            num_node_qubits = len(node_wires)
            if use_one_hot:
                total_prob = sum(probs)
                if total_prob <= 0:
                    return
                normalized = [p / total_prob for p in probs]
                vector = [0.0] * (2 ** num_node_qubits)
                for idx, p in enumerate(normalized):
                    vector[1 << idx] = np.sqrt(p)
                qml.AmplitudeEmbedding(vector, wires=node_wires, normalize=True)
                return

            padded_size = 2 ** num_node_qubits
            padded_probs = probs + [0.0] * (padded_size - len(probs))
            total_prob = sum(padded_probs)

            if num_node_qubits == 1:
                if total_prob > 0:
                    p1 = padded_probs[1] / total_prob
                    angle = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, p1))))
                    qml.RY(angle, wires=node_wires[0])
                return

            if num_node_qubits == 2:
                if total_prob <= 0:
                    return
                p = [x / total_prob for x in padded_probs]
                p_high = p[2] + p[3]
                if p_high > 0:
                    angle1 = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, p_high))))
                    qml.RY(angle1, wires=node_wires[0])
                p_low = p[0] + p[1]
                if p_low > 0:
                    p01_given_low = p[1] / p_low
                    angle2_low = 2 * np.arcsin(
                        np.sqrt(max(0.0, min(1.0, p01_given_low)))
                    )
                    qml.PauliX(wires=node_wires[0])
                    qml.CRY(angle2_low, wires=[node_wires[0], node_wires[1]])
                    qml.PauliX(wires=node_wires[0])
                if p_high > 0:
                    p11_given_high = p[3] / p_high
                    angle2_high = 2 * np.arcsin(
                        np.sqrt(max(0.0, min(1.0, p11_given_high)))
                    )
                    qml.CRY(angle2_high, wires=[node_wires[0], node_wires[1]])
                return

            metadata["approximation"] = "uniform_for_nodes_gt_2_qubits"
            for wire in node_wires:
                qml.Hadamard(wires=wire)

        def _apply_controlled_ry(angle: float, controls: List[int], target: int) -> None:
            if len(controls) == 1:
                qml.CRY(angle, wires=[controls[0], target])
                return
            controlled = qml.ctrl(qml.RY, control=controls)
            controlled(angle, wires=target)

        def _apply_controlled_encoding(
            probs: List[float], node_wires: List[int], ctrl_qubits: List[int]
        ) -> None:
            num_node_qubits = len(node_wires)
            if use_one_hot and num_node_qubits > 1:
                total_prob = sum(probs)
                if total_prob <= 0 or not ctrl_qubits:
                    return
                normalized = [p / total_prob for p in probs]
                vector = [0.0] * (2 ** num_node_qubits)
                for idx, p in enumerate(normalized):
                    vector[1 << idx] = np.sqrt(p)
                controlled = qml.ctrl(qml.AmplitudeEmbedding, control=ctrl_qubits)
                controlled(vector, wires=node_wires, normalize=True)
                return

            padded_size = 2 ** num_node_qubits
            padded_probs = probs + [0.0] * (padded_size - len(probs))
            total_prob = sum(padded_probs)

            if num_node_qubits == 1:
                if total_prob > 0 and ctrl_qubits:
                    p1 = padded_probs[1] / total_prob
                    angle = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, p1))))
                    _apply_controlled_ry(angle, ctrl_qubits, node_wires[0])
                return

            if num_node_qubits == 2:
                if total_prob <= 0 or not ctrl_qubits:
                    return
                p = [x / total_prob for x in padded_probs]
                p_high = p[2] + p[3]
                if p_high > 0:
                    angle1 = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, p_high))))
                    _apply_controlled_ry(angle1, ctrl_qubits, node_wires[0])
                p_low = p[0] + p[1]
                if p_low > 0:
                    p01_given_low = p[1] / p_low
                    angle2_low = 2 * np.arcsin(
                        np.sqrt(max(0.0, min(1.0, p01_given_low)))
                    )
                    qml.PauliX(wires=node_wires[0])
                    _apply_controlled_ry(
                        angle2_low, ctrl_qubits + [node_wires[0]], node_wires[1]
                    )
                    qml.PauliX(wires=node_wires[0])
                if p_high > 0:
                    p11_given_high = p[3] / p_high
                    angle2_high = 2 * np.arcsin(
                        np.sqrt(max(0.0, min(1.0, p11_given_high)))
                    )
                    _apply_controlled_ry(
                        angle2_high, ctrl_qubits + [node_wires[0]], node_wires[1]
                    )
                return

            metadata["approximation"] = "uniform_for_nodes_gt_2_qubits"
            for wire in node_wires:
                if ctrl_qubits:
                    _apply_controlled_ry(np.pi / 2, ctrl_qubits, wire)

        def _state_preparation(wires: Dict[str, List[int]]) -> None:
            nodes = list(model.nodes())
            for node in nodes:
                cpd = model.get_cpds(node)
                parents = cpd.variables[1:]
                vals = cpd.get_values()
                variable_card, num_cols = vals.shape
                node_wires = wires[node]

                if len(parents) == 0:
                    probs = [float(vals[state, 0]) for state in range(variable_card)]
                    _apply_unconditional(probs, node_wires)
                    continue

                parent_card_list = [model.get_cpds(p).variable_card for p in parents]

                for col in range(num_cols):
                    parent_state_indices = []
                    temp_col = col
                    for card in reversed(parent_card_list):
                        parent_state_indices.insert(0, temp_col % card)
                        temp_col //= card

                    parent_conditions = []
                    for parent, parent_state_idx in zip(parents, parent_state_indices):
                        parent_qubits = wires[parent]
                        if use_one_hot:
                            for idx, qubit in enumerate(parent_qubits):
                                bit = "1" if idx == parent_state_idx else "0"
                                parent_conditions.append((qubit, bit))
                        else:
                            binary_rep = format(parent_state_idx, f"0{len(parent_qubits)}b")
                            for idx, bit in enumerate(binary_rep):
                                parent_conditions.append((parent_qubits[idx], bit))

                    ctrl_qubits = []
                    for qubit, bit in parent_conditions:
                        ctrl_qubits.append(qubit)
                        if bit == "0":
                            qml.PauliX(wires=qubit)

                    probs = [float(vals[state, col]) for state in range(variable_card)]
                    _apply_controlled_encoding(probs, node_wires, ctrl_qubits)

                    for qubit, bit in parent_conditions:
                        if bit == "0":
                            qml.PauliX(wires=qubit)

        def _apply_oracle(wires: Dict[str, List[int]]) -> None:
            for node, state in evidence.items():
                node_qubits = wires[node]
                if use_one_hot:
                    for idx, qubit in enumerate(node_qubits):
                        if idx != state:
                            qml.PauliX(wires=qubit)
                    qml.MultiControlledX(wires=node_qubits)
                    for idx, qubit in enumerate(node_qubits):
                        if idx != state:
                            qml.PauliX(wires=qubit)
                    continue

                binary_state = format(state, f"0{len(node_qubits)}b")
                for idx, bit in enumerate(binary_state):
                    if bit == "0":
                        qml.PauliX(wires=node_qubits[idx])

                if len(node_qubits) == 1:
                    qml.PauliZ(wires=node_qubits[0])
                else:
                    target = node_qubits[-1]
                    ctrls = node_qubits[:-1]
                    qml.Hadamard(wires=target)
                    qml.MultiControlledX(wires=ctrls + [target])
                    qml.Hadamard(wires=target)

                for idx, bit in enumerate(binary_state):
                    if bit == "0":
                        qml.PauliX(wires=node_qubits[idx])

        def _apply_diffuser(wires: Dict[str, List[int]]) -> None:
            qml.adjoint(lambda: _state_preparation(wires))()
            all_wires = [wire for wires_list in wires.values() for wire in wires_list]
            for wire in all_wires:
                qml.PauliX(wires=wire)
            target = all_wires[-1]
            qml.Hadamard(wires=target)
            if len(all_wires) == 1:
                qml.PauliZ(wires=target)
            else:
                qml.MultiControlledX(wires=all_wires[:-1] + [target])
            qml.Hadamard(wires=target)
            for wire in all_wires:
                qml.PauliX(wires=wire)
            _state_preparation(wires)

        def circuit_fn(
            wires: Dict[str, List[int]],
            evidence_map: Dict[str, int],
            query_nodes: List[str],
        ):
            _state_preparation(wires)

            for _ in range(num_iterations):
                _apply_oracle(wires)
                _apply_diffuser(wires)

            wires_to_measure = query_wires(wires, query_nodes)
            return qml.probs(wires=wires_to_measure)

        return CircuitSpec(
            name=self.name,
            num_wires=num_wires,
            wires_map=wires_map,
            circuit_fn=circuit_fn,
            metadata=metadata,
        )
