from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import ensure_dir


def _query_probs_to_matrix(
    probs: List[float], query: List[str], wires_map: Dict[str, List[int]]
) -> tuple[list[list[float]], list[int], list[int]]:
    x_state_count = 2 ** len(wires_map[query[0]])
    y_state_count = 2 ** len(wires_map[query[1]])
    expected_length = x_state_count * y_state_count
    if len(probs) != expected_length:
        raise ValueError(
            f"Expected {expected_length} probability values for query {query}, got {len(probs)}"
        )

    matrix = [[0.0 for _ in range(x_state_count)] for _ in range(y_state_count)]
    total_bits = len(wires_map[query[0]]) + len(wires_map[query[1]])

    for idx, prob in enumerate(probs):
        bitstr = format(idx, f"0{total_bits}b")
        x_bits = bitstr[: len(wires_map[query[0]])]
        y_bits = bitstr[len(wires_map[query[0]]) :]
        x_state = int(x_bits, 2)
        y_state = int(y_bits, 2)
        matrix[y_state][x_state] = prob

    return matrix, list(range(x_state_count)), list(range(y_state_count))


def _parse_dict_distribution(
    distribution: Dict[str, float], query: List[str], wires_map: Dict[str, List[int]]
) -> tuple[list[float] | list[list[float]], list[int], list[int]]:
    if len(query) == 1:
        state_count = 2 ** len(wires_map[query[0]])
        probs = [0.0] * state_count
        for key, value in distribution.items():
            if "=" in key:
                _, state_str = key.split("=", 1)
            else:
                state_str = key
            state = int(state_str)
            if 0 <= state < state_count:
                probs[state] = float(value)
        return probs, list(range(state_count)), []

    if len(query) == 2:
        x_state_count = 2 ** len(wires_map[query[0]])
        y_state_count = 2 ** len(wires_map[query[1]])
        matrix = [[0.0 for _ in range(x_state_count)] for _ in range(y_state_count)]
        for key, value in distribution.items():
            coords = []
            for coord_str in key.split(","):
                if "=" in coord_str:
                    _, state_str = coord_str.split("=", 1)
                else:
                    state_str = coord_str
                coords.append(int(state_str))
            if len(coords) == 2:
                x_state, y_state = coords[0], coords[1]
                if 0 <= x_state < x_state_count and 0 <= y_state < y_state_count:
                    matrix[y_state][x_state] = float(value)
        return matrix, list(range(x_state_count)), list(range(y_state_count))

    raise ValueError("Unsupported query length for distribution parsing.")


def _plot_single_distribution(
    ax: plt.Axes,
    probs: List[float],
    title: str,
    label: str,
) -> None:
    x = list(range(len(probs)))
    ax.bar(x, probs, label=label, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.legend()


def _plot_heatmap(
    ax: plt.Axes,
    matrix: list[list[float]],
    title: str,
    x_label: str,
    y_label: str,
    x_ticks: list[int],
    y_ticks: list[int],
) -> None:
    im = ax.imshow(matrix, cmap="hot", origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(ticks=y_ticks)
    ax.set_yticklabels(y_ticks)
    return im


def save_probabilities_plot(
    output_dir: str,
    probs: List[float],
    title: str,
    filename: str = "probs.png",
    query: Optional[List[str]] = None,
    wires_map: Optional[Dict[str, List[int]]] = None,
    baseline_distribution: Optional[Dict[str, float]] = None,
) -> str:
    ensure_dir(output_dir)
    baseline_available = baseline_distribution is not None and query is not None and wires_map is not None

    if baseline_available and len(query) == 2:
        quantum_matrix, x_ticks, y_ticks = _query_probs_to_matrix(probs, query, wires_map)
        baseline_matrix, _, _ = _parse_dict_distribution(baseline_distribution, query, wires_map)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im0 = _plot_heatmap(
            axes[0], baseline_matrix, f"Baseline {title}", query[0], query[1], x_ticks, y_ticks
        )
        im1 = _plot_heatmap(
            axes[1], quantum_matrix, f"Quantum {title}", query[0], query[1], x_ticks, y_ticks
        )
        fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.04, pad=0.04)
        plt.tight_layout()
        path = f"{output_dir}/{filename}"
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    if baseline_available and len(query) == 1:
        quantum_probs = [float(val) for val in probs]
        baseline_probs, x_ticks, _ = _parse_dict_distribution(baseline_distribution, query, wires_map)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        x = list(range(len(quantum_probs)))
        width = 0.35
        ax.bar([xi - width / 2 for xi in x], quantum_probs, width=width, label="Quantum")
        ax.bar([xi + width / 2 for xi in x], baseline_probs, width=width, label="Baseline")
        ax.set_title(title)
        ax.set_xlabel(query[0])
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(x_ticks)
        ax.legend()
        plt.tight_layout()
        path = f"{output_dir}/{filename}"
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    if query is not None and wires_map is not None and len(query) == 2:
        try:
            matrix, x_ticks, y_ticks = _query_probs_to_matrix(probs, query, wires_map)
            plt.figure(figsize=(8, 6))
            im = plt.imshow(matrix, cmap="hot", origin="lower", aspect="auto")
            plt.colorbar(im, label="Probability")
            plt.title(f"{title} (heatmap)")
            plt.xlabel(query[0])
            plt.ylabel(query[1])
            plt.xticks(ticks=x_ticks, labels=x_ticks)
            plt.yticks(ticks=y_ticks, labels=y_ticks)
            plt.tight_layout()
            path = f"{output_dir}/{filename}"
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except ValueError:
            pass

    x = list(range(len(probs)))
    plt.figure(figsize=(8, 4))
    plt.bar(x, probs, color="#2b6cb0")
    plt.title(title)
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    plt.tight_layout()
    path = f"{output_dir}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    return path
