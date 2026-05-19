from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import math
from pgmpy.models import DiscreteBayesianNetwork


@dataclass
class CircuitSpec:
    name: str
    num_wires: int
    wires_map: Dict[str, List[int]]
    circuit_fn: Callable[[Dict[str, List[int]], Dict[str, int], List[str]], Any]
    metadata: Dict[str, Any]


class QompilerBase(ABC):
    name: str = "BASE"

    @abstractmethod
    def compile(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        encoding: str,
        encoding_params: Dict[str, Any],
    ) -> CircuitSpec:
        raise NotImplementedError


def qubit_count_per_card(card: int, encoding: str) -> int:
    """Number of qubits needed to encode a discrete variable of given cardinality.

    `sparse_topk` shares the binary wire layout — the encoding only changes how
    amplitudes are prepared, not how basis states are addressed.
    """
    if encoding == "one_hot":
        return card
    if encoding in ("binary", "sparse_topk"):
        return max(1, int(math.ceil(math.log2(card))))
    raise ValueError(f"Unknown encoding: {encoding!r}")


def build_wires_map(
    model: DiscreteBayesianNetwork,
    encoding: str,
) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    next_wire = 0
    for node in model.nodes():
        cpd = model.get_cpds(node)
        card = int(cpd.variable_card)
        num_qubits = qubit_count_per_card(card, encoding)
        mapping[node] = list(range(next_wire, next_wire + num_qubits))
        next_wire += num_qubits
    return mapping


def query_wires(wires_map: Dict[str, List[int]], query: List[str]) -> List[int]:
    if not query:
        wires = []
        for node_wires in wires_map.values():
            wires.extend(node_wires)
        return wires
    wires = []
    for node in query:
        if node not in wires_map:
            raise ValueError(f"Query variable not in wires map: {node}")
        wires.extend(wires_map[node])
    return wires
