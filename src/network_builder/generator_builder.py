from __future__ import annotations

from pgmpy.models import DiscreteBayesianNetwork

from src.config.types import NetworkConfig
from src.networks.generators import generate_random_bn, generate_ship_localization_bn

from .base import NetworkBuilderBase


class GeneratorNetworkBuilder(NetworkBuilderBase):
    name = "generator"

    def build(self, config: NetworkConfig) -> DiscreteBayesianNetwork:
        if config.type == "ship_localization":
            grid_size = int(config.params.get("grid_size", 16))
            lighthouses = config.params.get("lighthouses")
            if lighthouses is not None:
                lighthouses = [tuple(map(int, lh)) for lh in lighthouses]
            return generate_ship_localization_bn(
                grid_size=grid_size,
                lighthouses=lighthouses,
                num_dist_bins=int(config.params.get("num_dist_bins", 20)),
                num_bear_bins=int(config.params.get("num_bear_bins", 16)),
                dist_noise=tuple(config.params.get("dist_noise", [0.6, 0.15, 0.05, 0.01])),
                bear_noise=tuple(config.params.get("bear_noise", [0.6, 0.15, 0.05, 0.01])),
            )
        if config.type == "random":
            return generate_random_bn(
                num_nodes=int(config.params.get("num_nodes", 5)),
                max_parents=int(config.params.get("max_parents", 2)),
                max_states=int(config.params.get("max_states", 2)),
                seed=config.seed if config.seed is not None else config.params.get("seed"),
            )
        raise ValueError(f"Unsupported generated network type: {config.type}")
