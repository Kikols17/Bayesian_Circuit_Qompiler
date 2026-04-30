from __future__ import annotations

from typing import Dict, List

import numpy as np
import pennylane as qml
from pgmpy.models import DiscreteBayesianNetwork

from .base import CircuitSpec, QompilerBase, build_wires_map, query_wires


class QAAQompiler(QompilerBase):
    name = "QAA"

    def compile(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        encoding: str = "binary",
    ) -> CircuitSpec:
        wires_map = build_wires_map(model, encoding=encoding)
        num_wires = max(w for wires in wires_map.values() for w in wires) + 1
        query_wire_list = query_wires(wires_map, query)
        use_one_hot = encoding == "one_hot"
        num_iterations = 1
        metadata: Dict[str, object] = {
            "query_wires": query_wire_list,
            "num_iterations": num_iterations,
            "encoding": encoding,
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

            if evidence:
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
