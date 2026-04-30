from __future__ import annotations

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import BIFReader

from src.config.types import NetworkConfig

from .base import NetworkBuilderBase


class BIFNetworkBuilder(NetworkBuilderBase):
    name = "bif"

    def build(self, config: NetworkConfig) -> DiscreteBayesianNetwork:
        path = config.params.get("path")
        if not path:
            raise ValueError("BIF network builder requires params.path")
        reader = BIFReader(path)
        model = reader.get_model()
        model.check_model()
        return model
