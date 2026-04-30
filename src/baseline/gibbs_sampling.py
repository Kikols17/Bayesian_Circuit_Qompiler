from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .base import BaselineBase


class GibbsSamplingBaseline(BaselineBase):
    name = "gibbs_sampling"

    def run(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        num_samples = int(params.get("num_samples", 1000))
        burn_in = int(params.get("burn_in", 200))
        sample_interval = int(params.get("sample_interval", 1))

        nodes = list(model.nodes())
        non_evidence = [n for n in nodes if n not in evidence]
        if not non_evidence:
            return {
                "method": self.name,
                "query": query,
                "evidence": evidence,
                "distribution": {},
                "note": "All variables are evidence variables.",
            }

        inference = VariableElimination(model)
        current_state = {
            node: evidence[node]
            if node in evidence
            else int(model.get_cpds(node).variable_card / 2)
            for node in nodes
        }

        samples: List[Dict[str, int]] = []
        for iteration in range(num_samples * sample_interval + burn_in):
            for node in non_evidence:
                q = inference.query(variables=[node], evidence={
                    **{n: current_state[n] for n in nodes if n != node},
                    **evidence,
                })
                dist = q.values
                dist = np.array(dist, dtype=float)
                if dist.sum() == 0:
                    prob = np.ones_like(dist) / len(dist)
                else:
                    prob = dist / dist.sum()
                current_state[node] = int(np.random.choice(len(prob), p=prob))

            if iteration >= burn_in and (iteration - burn_in) % sample_interval == 0:
                samples.append({k: current_state[k] for k in query})

        counts: Dict[str, int] = {}
        for sample in samples:
            key = ",".join(f"{k}={sample[k]}" for k in query)
            counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values())
        distribution = {key: value / total for key, value in counts.items()}

        return {
            "method": self.name,
            "query": query,
            "evidence": evidence,
            "distribution": distribution,
            "samples": len(samples),
            "params": {
                "num_samples": num_samples,
                "burn_in": burn_in,
                "sample_interval": sample_interval,
            },
        }
