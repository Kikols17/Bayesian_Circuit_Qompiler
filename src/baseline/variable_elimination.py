from __future__ import annotations

from typing import Any, Dict, List

from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .base import BaselineBase


class VariableEliminationBaseline(BaselineBase):
    name = "variable_elimination"

    def run(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        infer = VariableElimination(model)
        distribution = infer.query(variables=query, evidence=evidence)

        result: Dict[str, Any] = {
            "method": self.name,
            "query": query,
            "evidence": evidence,
            "distribution": {},
        }

        if len(query) == 1:
            variable = query[0]
            values = distribution.values
            result["distribution"] = {str(i): float(values[i]) for i in range(len(values))}
        else:
            result["distribution"] = self._flatten_distribution(distribution)

        return result

    def _flatten_distribution(self, distribution: Any) -> Dict[str, float]:
        values = distribution.values
        shape = values.shape
        query_vars = distribution.variables
        result: Dict[str, float] = {}

        for index, prob in enumerate(values.flatten()):
            coords = []
            remainder = index
            for size in reversed(shape):
                coords.insert(0, remainder % size)
                remainder //= size
            key = ",".join(f"{var}={coord}" for var, coord in zip(query_vars, coords))
            result[key] = float(prob)

        return result
