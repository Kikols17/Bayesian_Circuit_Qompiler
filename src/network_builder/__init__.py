"""Network builder implementations."""

from typing import Type

from .base import NetworkBuilderBase
from .bif_builder import BIFNetworkBuilder
from .generator_builder import GeneratorNetworkBuilder


def get_network_builder(name: str) -> NetworkBuilderBase:
    name_lower = name.strip().lower()
    registry: dict[str, Type[NetworkBuilderBase]] = {
        "generator": GeneratorNetworkBuilder,
        "bif": BIFNetworkBuilder,
    }
    if name_lower not in registry:
        raise ValueError(f"Unknown network builder: {name}")
    return registry[name_lower]()
