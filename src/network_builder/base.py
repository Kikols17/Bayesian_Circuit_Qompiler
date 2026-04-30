from __future__ import annotations

from abc import ABC, abstractmethod

from pgmpy.models import DiscreteBayesianNetwork

from src.config.types import NetworkConfig


class NetworkBuilderBase(ABC):
    name: str = "BASE"

    @abstractmethod
    def build(self, config: NetworkConfig) -> DiscreteBayesianNetwork:
        raise NotImplementedError
