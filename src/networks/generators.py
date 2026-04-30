from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def generate_ship_localization_bn(
    grid_size: int = 16,
    lighthouses: List[Tuple[int, int]] | None = None,
    num_dist_bins: int = 20,
    num_bear_bins: int = 16,
    dist_noise: Tuple[float, float, float, float] = (0.6, 0.15, 0.05, 0.01),
    bear_noise: Tuple[float, float, float, float] = (0.6, 0.15, 0.05, 0.01),
) -> DiscreteBayesianNetwork:
    """Create a ship localization model with distance and bearing observations.

    The ship's position is represented by discrete X and Y coordinates.
    Observations are conditionally independent given the ship position.

    Variables:
    - X, Y: ship position on a grid (uniform prior)
    - Dist{i}: discretized distance observation to lighthouse i
    - Bear{i}: discretized bearing observation to lighthouse i

    Args:
        grid_size: Number of states for X and Y.
        lighthouses: Optional positions of lighthouses. If None, two default locations are used.
        num_dist_bins: Number of discrete bins for each distance observation.
        num_bear_bins: Number of discrete bins for each bearing observation.
        dist_noise: Noise weights for distance observation bins [true, adj1, adj2, other].
        bear_noise: Noise weights for bearing observation bins [true, adj1, adj2, other].
    """

    if lighthouses is None:
        lighthouses = [
            (max(1, grid_size // 6), max(1, grid_size // 6)),
            (max(1, 5 * grid_size // 6), max(1, grid_size // 4)),
        ]

    nodes = ["X", "Y"]
    for i in range(1, len(lighthouses) + 1):
        nodes.extend([f"Dist{i}", f"Bear{i}"])

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)

    edges = []
    for i in range(1, len(lighthouses) + 1):
        edges.extend([("X", f"Dist{i}"), ("Y", f"Dist{i}"), ("X", f"Bear{i}"), ("Y", f"Bear{i}")])
    model.add_edges_from(edges)

    def _discretize_distance(dist: float) -> int:
        max_dist = np.hypot(grid_size - 1, grid_size - 1)
        bin_idx = int(min(num_dist_bins - 1, dist * num_dist_bins / max_dist))
        return bin_idx

    def _discretize_bearing(bearing_rad: float) -> int:
        bearing_deg = (bearing_rad * 180.0 / np.pi) % 360.0
        bin_idx = int(bearing_deg * num_bear_bins / 360.0)
        return min(bin_idx, num_bear_bins - 1)

    def _normalize_columns(values: List[List[float]]) -> None:
        num_rows = len(values)
        num_cols = len(values[0]) if values else 0
        for col in range(num_cols):
            col_sum = sum(values[row][col] for row in range(num_rows))
            if col_sum > 0.0:
                for row in range(num_rows):
                    values[row][col] /= col_sum
            else:
                for row in range(num_rows):
                    values[row][col] = 1.0 / num_rows

    x_values = [[1.0 / grid_size] for _ in range(grid_size)]
    y_values = [[1.0 / grid_size] for _ in range(grid_size)]
    model.add_cpds(TabularCPD(variable="X", variable_card=grid_size, values=x_values))
    model.add_cpds(TabularCPD(variable="Y", variable_card=grid_size, values=y_values))

    for lh_idx, (lh_x, lh_y) in enumerate(lighthouses, 1):
        dist_values: List[List[float]] = []
        for dist_bin in range(num_dist_bins):
            row: List[float] = []
            for y in range(grid_size):
                for x in range(grid_size):
                    true_dist = np.hypot(x - lh_x, y - lh_y)
                    true_bin = _discretize_distance(true_dist)
                    bin_diff = abs(dist_bin - true_bin)

                    if bin_diff == 0:
                        prob = dist_noise[0]
                    elif bin_diff == 1:
                        prob = dist_noise[1]
                    elif bin_diff == 2:
                        prob = dist_noise[2]
                    else:
                        prob = dist_noise[3]
                    row.append(prob)
            dist_values.append(row)
        _normalize_columns(dist_values)
        model.add_cpds(
            TabularCPD(
                variable=f"Dist{lh_idx}",
                variable_card=num_dist_bins,
                values=dist_values,
                evidence=["X", "Y"],
                evidence_card=[grid_size, grid_size],
            )
        )

        bear_values: List[List[float]] = []
        for bear_bin in range(num_bear_bins):
            row: List[float] = []
            for y in range(grid_size):
                for x in range(grid_size):
                    dx = lh_x - x
                    dy = lh_y - y
                    true_bearing = 0.0 if dx == 0 and dy == 0 else np.arctan2(dy, dx)
                    true_bin = _discretize_bearing(true_bearing)
                    wrapped_diff = min(
                        abs(bear_bin - true_bin),
                        abs(bear_bin - true_bin + num_bear_bins),
                        abs(bear_bin - true_bin - num_bear_bins),
                    )

                    if wrapped_diff == 0:
                        prob = bear_noise[0]
                    elif wrapped_diff == 1:
                        prob = bear_noise[1]
                    elif wrapped_diff == 2:
                        prob = bear_noise[2]
                    else:
                        prob = bear_noise[3]
                    row.append(prob)
            bear_values.append(row)
        _normalize_columns(bear_values)
        model.add_cpds(
            TabularCPD(
                variable=f"Bear{lh_idx}",
                variable_card=num_bear_bins,
                values=bear_values,
                evidence=["X", "Y"],
                evidence_card=[grid_size, grid_size],
            )
        )

    model.check_model()
    return model


def generate_random_bn(
    num_nodes: int = 5,
    max_parents: int = 2,
    max_states: int = 2,
    seed: int | None = None,
) -> DiscreteBayesianNetwork:
    if seed is not None:
        random.seed(seed)
    nodes = [f"X{i}" for i in range(num_nodes)]
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)

    edges = []
    for child in nodes:
        possible_parents = nodes[: nodes.index(child)]
        num_par = random.randint(0, min(max_parents, len(possible_parents)))
        parents = random.sample(possible_parents, num_par)
        for parent in parents:
            edges.append((parent, child))
    model.add_edges_from(edges)

    for node in nodes:
        parents = model.get_parents(node)
        parent_card = [model.get_cpds(p).variable_card for p in parents] if parents else []
        state_count = random.randint(2, max_states)
        num_columns = int(np.prod(parent_card)) if parents else 1
        raw = [[random.random() for _ in range(num_columns)] for _ in range(state_count)]

        columns = list(zip(*raw))
        norm_cols = []
        for col in columns:
            s = sum(col)
            norm_cols.append([v / s for v in col])
        values = list(zip(*norm_cols))
        values = [list(v) for v in values]

        cpd = TabularCPD(
            variable=node,
            variable_card=state_count,
            values=values,
            evidence=parents if parents else None,
            evidence_card=parent_card if parents else None,
        )
        model.add_cpds(cpd)

    model.check_model()
    return model
