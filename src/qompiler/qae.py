"""QAE compiler.

Allocates the joint+flag qubit layout and packs the metadata the runner needs
to dispatch to `src.execution.qae_runner.run_qae`. No circuit is built here —
QAE is inherently a *multi-circuit* algorithm (one MLAE round per estimated
amplitude), so the runner drives circuit construction itself.

The `circuit_fn` is a hard-error stub: QAE bypasses the standard
builder/backend pipeline, and the runner's dispatch must catch this before
`circuit_fn` is ever called.

See [problems/qae_prototype.md](../../problems/qae_prototype.md) for the
algorithm; see [problems/quantum_advantage.md](../../problems/quantum_advantage.md)
for why this is the only path in the codebase with a genuine asymptotic
advantage over classical sampling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from pgmpy.models import DiscreteBayesianNetwork

from src.execution.qae_runner import joint_qubit_layout

from .base import CircuitSpec, QompilerBase, query_wires


class QAEQompiler(QompilerBase):
    name = "QAE"

    def compile(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        encoding: str,
        encoding_params: Dict[str, Any],
    ) -> CircuitSpec:
        if encoding != "binary":
            raise ValueError(
                f"QAE compiler requires encoding='binary'; got {encoding!r}. "
                "QAE estimates amplitudes on the joint register; sparse/redundant/"
                "one_hot are amplitude-encoding tricks for *known* distributions, "
                "which QAE specifically avoids."
            )
        if encoding_params:
            raise ValueError(
                "QAE accepts no encoding_params (the algorithm has no encoding "
                "knobs). Use '{}'. MLAE schedule, shots-per-round, and driver "
                "live in backend.params: "
                "{mlae_schedule: [..], shots_per_round: int, driver: 'mlae'}."
            )
        if not evidence:
            raise ValueError(
                "QAE requires non-empty evidence; the algorithm estimates a "
                "posterior conditioned on observed values."
            )
        if not query:
            raise ValueError("QAE requires a non-empty query list.")
        for node in list(evidence) + list(query):
            if node not in model.nodes():
                raise ValueError(f"Node {node!r} is not in the network.")

        layout, num_state_qubits = joint_qubit_layout(model)
        flag_qubit = num_state_qubits
        num_wires = num_state_qubits + 1

        wires_map = {node: list(qubits) for node, qubits in layout.items()}
        query_wire_list = query_wires(wires_map, query)
        query_cards = [int(model.get_cpds(node).variable_card) for node in query]

        metadata: Dict[str, Any] = {
            "mode": "qae",
            "encoding": encoding,
            "encoding_params": dict(encoding_params),
            "query_wires": query_wire_list,
            "effective_query": list(query),
            "layout": {n: list(qs) for n, qs in layout.items()},
            "flag_qubit": flag_qubit,
            "num_state_qubits": num_state_qubits,
            "query_cards": query_cards,
            "query_bits": [max(1, int(math.ceil(math.log2(c)))) for c in query_cards],
        }

        def circuit_fn(*_args, **_kwargs):
            raise RuntimeError(
                "QAE bypasses the standard builder/backend pipeline. The "
                "pipeline runner must dispatch to src.execution.qae_runner.run_qae "
                "when compiler == 'QAE'; circuit_fn should never be invoked."
            )

        return CircuitSpec(
            name=self.name,
            num_wires=num_wires,
            wires_map=wires_map,
            circuit_fn=circuit_fn,
            metadata=metadata,
        )
