from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec


class CircuitBuilderBase(ABC):
    name: str = "BASE"

    @abstractmethod
    def build(
        self,
        spec: CircuitSpec,
        evidence: Dict[str, int],
        query: List[str],
        circuit_config: CircuitConfig,
        backend_config: BackendConfig,
    ) -> Any:
        raise NotImplementedError
