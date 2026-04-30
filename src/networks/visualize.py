from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.utils.io import ensure_dir


def save_bayesian_network_plot(
    output_dir: str,
    model: Any,
    filename: str = "bayesian_network.png",
    seed: int = 42,
) -> str:
    ensure_dir(output_dir)
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())
    pos = nx.spring_layout(graph, seed=seed, k=1.5, iterations=300)

    fig, (ax_graph, ax_table) = plt.subplots(
        1,
        2,
        figsize=(18, 10),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax_graph,
        node_size=1600,
        node_color="#4c72b0",
        edgecolors="#222222",
        linewidths=1.5,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax_graph,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="#555555",
        width=1.4,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=15,
        min_target_margin=15,
    )
    labels = nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax_graph,
        font_size=10,
        font_color="white",
        font_weight="bold",
    )

    nodes.set_zorder(2)
    for text in labels.values():
        text.set_zorder(3)

    ax_graph.set_axis_off()

    ax_table.set_axis_off()
    node_list = list(model.nodes())
    table_height = 0.85 / max(len(node_list), 1)
    y_top = 0.95

    for node in node_list:
        cpd = model.get_cpds(node)
        evidence_vars = cpd.get_evidence()
        values = cpd.get_values()

        if values.ndim == 1:
            col_labels = ["State", "Prob"]
            cell_text = [[str(i), f"{float(values[i]):.3f}"] for i in range(values.shape[0])]
        else:
            max_cols = min(values.shape[1], 4)
            evidence_labels = []
            parent_cards = [model.get_cpds(ev).variable_card for ev in evidence_vars]
            for col in range(max_cols):
                if evidence_vars:
                    temp = col
                    state_indexes = []
                    for card in reversed(parent_cards):
                        state_indexes.insert(0, temp % card)
                        temp //= card
                    evidence_labels.append(
                        ",".join(f"{ev}={idx}" for ev, idx in zip(evidence_vars, state_indexes))
                    )
                else:
                    evidence_labels.append(f"col{col}")
            if values.shape[1] > max_cols:
                evidence_labels.append("...")
            col_labels = ["State"] + evidence_labels
            cell_text = []
            for state_idx in range(values.shape[0]):
                row = [str(state_idx)]
                row.extend(f"{float(values[state_idx, col]):.3f}" for col in range(max_cols))
                if values.shape[1] > max_cols:
                    row.append("...")
                cell_text.append(row)

        title = f"{cpd.variable} (card={cpd.variable_card})"
        if evidence_vars:
            title += f" | evidence={','.join(evidence_vars)}"

        ax_table.text(
            0.01,
            y_top,
            title,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper left",
            bbox=[0.01, y_top - table_height + 0.005, 0.98, table_height - 0.03],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        y_top -= table_height

        if y_top < 0.1:
            break

    plt.tight_layout()
    path = f"{output_dir}/{filename}"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
