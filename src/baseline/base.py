from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pgmpy.models import DiscreteBayesianNetwork


class BaselineBase(ABC):
    name: str = "BASE"

    @abstractmethod
    def run(
        self,
        model: DiscreteBayesianNetwork,
        evidence: Dict[str, int],
        query: List[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError
