from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.qompiler.base import card_per_node, decode_node_value
from src.utils.io import ensure_dir


def _query_probs_to_matrix(
    probs: List[float], query: List[str], wires_map: Dict[str, List[int]],
    encoding: str = "binary",
) -> tuple[list[list[float]], list[int], list[int]]:
    x_state_count = card_per_node(len(wires_map[query[0]]), encoding)
    y_state_count = card_per_node(len(wires_map[query[1]]), encoding)

    matrix = [[0.0 for _ in range(x_state_count)] for _ in range(y_state_count)]
    total_bits = len(wires_map[query[0]]) + len(wires_map[query[1]])

    if len(probs) != 2 ** total_bits:
        raise ValueError(
            f"Expected {2 ** total_bits} probability values for query {query} "
            f"({encoding} encoding), got {len(probs)}"
        )

    for idx, prob in enumerate(probs):
        if prob <= 0.0:
            continue
        bitstr = format(idx, f"0{total_bits}b")
        x_bits = bitstr[: len(wires_map[query[0]])]
        y_bits = bitstr[len(wires_map[query[0]]) :]
        x_state = decode_node_value(x_bits, encoding)
        y_state = decode_node_value(y_bits, encoding)
        if x_state is None or y_state is None:
            continue
        matrix[y_state][x_state] += prob

    if encoding == "one_hot":
        total = sum(p for row in matrix for p in row)
        if total > 0:
            matrix = [[p / total for p in row] for row in matrix]

    return matrix, list(range(x_state_count)), list(range(y_state_count))


def _parse_dict_distribution(
    distribution: Dict[str, float], query: List[str], wires_map: Dict[str, List[int]],
    encoding: str = "binary",
) -> tuple[list[float] | list[list[float]], list[int], list[int]]:
    if len(query) == 1:
        state_count = card_per_node(len(wires_map[query[0]]), encoding)
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
        x_state_count = card_per_node(len(wires_map[query[0]]), encoding)
        y_state_count = card_per_node(len(wires_map[query[1]]), encoding)
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
    encoding: str = "binary",
) -> str:
    ensure_dir(output_dir)
    baseline_available = baseline_distribution is not None and query is not None and wires_map is not None

    if baseline_available and len(query) == 2:
        quantum_matrix, x_ticks, y_ticks = _query_probs_to_matrix(probs, query, wires_map, encoding)
        baseline_matrix, _, _ = _parse_dict_distribution(baseline_distribution, query, wires_map, encoding)

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
        baseline_probs, x_ticks, _ = _parse_dict_distribution(baseline_distribution, query, wires_map, encoding)
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
            matrix, x_ticks, y_ticks = _query_probs_to_matrix(probs, query, wires_map, encoding)
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


def compute_prob_history_from_samples(
    samples: Any,
    wires_map: Dict[str, List[int]],
    query: Optional[List[str]] = None,
    evidence: Optional[Dict[str, int]] = None,
) -> tuple[List[List[float]], int]:
    """Compute a per-shot probability history for the specified `query`.

    - `samples` may be a numpy array of shape `(shots, num_wires)` or a list of
      bitstrings (as returned by Qiskit `result.get_memory()`).
    - Returns `(history, matched_count)` where `history` has one entry per total
      shot (1:1), carrying forward the last distribution for non-matching shots.
      `matched_count` is the number of shots that satisfied the evidence constraint.
    """
    if samples is None:
        return [], 0

    query_nodes = query if query else list(wires_map.keys())
    query_wires = [w for node in query_nodes for w in wires_map[node]]
    evidence_nodes = list(evidence.keys()) if evidence else []

    num_outcomes = 2 ** len(query_wires)
    uniform = [1.0 / num_outcomes] * num_outcomes
    counts = np.zeros(num_outcomes, dtype=float)
    matched = 0
    history: List[List[float]] = []
    last_dist: List[float] = uniform

    wire_to_pos: Dict[int, int] = {w: i for i, w in enumerate(query_wires)}

    def _decode_from_array(sample) -> int:
        bits = [str(int(sample[w])) for w in query_wires]
        return int("".join(bits), 2) if bits else 0

    def _decode_node_from_array(sample, node_wires: List[int]) -> int:
        if not node_wires:
            return 0
        bits = [str(int(sample[w])) for w in node_wires]
        return int("".join(bits), 2)

    def _decode_from_bitstring(s: str) -> int:
        s_clean = s.strip()
        # Qiskit memory strings are printed MSB..LSB; reverse so index -> classical bit
        s_rev = s_clean[::-1]
        bits: List[str] = []
        for node in query_nodes:
            for w in wires_map[node]:
                if w in wire_to_pos and wire_to_pos[w] < len(s_rev):
                    bits.append(s_rev[wire_to_pos[w]])
                else:
                    bits.append("0")
        return int("".join(bits), 2) if bits else 0

    def _decode_node_from_bitstring(s: str, node_wires: List[int]) -> int:
        s_clean = s.strip()
        s_rev = s_clean[::-1]
        bits = [s_rev[wire_to_pos[w]] if w in wire_to_pos and wire_to_pos[w] < len(s_rev) else "0" for w in node_wires]
        return int("".join(bits), 2) if bits else 0

    if isinstance(samples, np.ndarray):
        if samples.ndim != 2:
            return [], 0
        for sample in samples:
            matches = True
            for node in evidence_nodes:
                if _decode_node_from_array(sample, wires_map[node]) != evidence[node]:
                    matches = False
                    break
            if matches:
                matched += 1
                idx = _decode_from_array(sample)
                if 0 <= idx < num_outcomes:
                    counts[idx] += 1
                last_dist = (counts / float(matched)).tolist()
            history.append(last_dist)
        return history, matched

    try:
        for s in samples:
            if not isinstance(s, str):
                s = str(s)
            matches = True
            for node in evidence_nodes:
                if any(w in wire_to_pos for w in wires_map[node]):
                    if _decode_node_from_bitstring(s, wires_map[node]) != evidence[node]:
                        matches = False
                        break
                else:
                    # evidence wires not measured; cannot post-select
                    matches = False
                    break
            if matches:
                matched += 1
                idx = _decode_from_bitstring(s)
                if 0 <= idx < num_outcomes:
                    counts[idx] += 1
                last_dist = (counts / float(matched)).tolist()
            history.append(last_dist)
        return history, matched
    except Exception:
        return [], 0


def save_probability_evolution_plot(
    output_dir: str,
    prob_history: List[List[float]],
    title: str,
    filename: str = "prob_evolution.png",
    query: Optional[List[str]] = None,
    wires_map: Optional[Dict[str, List[int]]] = None,
    top_n: int = 10,
    encoding: str = "binary",
) -> str:
    """Save a plot showing evolution of probabilities over matched shots.

    The plot shows the top `top_n` outcomes sorted by final probability.
    For one_hot encoding, only valid one-hot indices are eligible for top-N.
    """
    ensure_dir(output_dir)
    if not prob_history:
        return ""

    hist = np.array(prob_history)
    if hist.size == 0:
        return ""

    final = hist[-1]
    if encoding == "one_hot" and query is not None and wires_map is not None:
        node_sizes = [len(wires_map[node]) for node in query]
        total_bits = sum(node_sizes)
        valid_mask = np.zeros(final.size, dtype=bool)
        for idx in range(final.size):
            bitstr = format(idx, f"0{total_bits}b")
            pos = 0
            ok = True
            for size in node_sizes:
                seg = bitstr[pos:pos + size]
                pos += size
                if seg.count("1") != 1:
                    ok = False
                    break
            valid_mask[idx] = ok
        scored = np.where(valid_mask, final, -1.0)
    else:
        scored = final
    max_k = min(top_n, scored.size)
    order = np.argsort(-scored)
    top_idx = order[:max_k]

    def _index_label(idx: int) -> str:
        if query is None or wires_map is None or len(query) <= 1:
            return str(idx)
        node_sizes = [len(wires_map[node]) for node in query]
        total_bits = sum(node_sizes)
        bitstr = format(idx, f"0{total_bits}b")
        pos = 0
        coords = []
        for node, size in zip(query, node_sizes):
            seg = bitstr[pos : pos + size]
            pos += size
            val = decode_node_value(seg, encoding)
            coords.append(str(val) if val is not None else "?")
        return ",".join(f"{n}={v}" for n, v in zip(query, coords))

    shots = np.arange(1, hist.shape[0] + 1)
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, idx in enumerate(top_idx):
        color = cmap(i % 10)
        plt.plot(shots, hist[:, idx], label=f"{_index_label(int(idx))} ({final[int(idx)]:.3f})", color=color)

    plt.xlabel("Shots")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    path = f"{output_dir}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    return path
