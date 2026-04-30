from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml

from src.utils.io import ensure_dir


def save_qiskit_circuit_image(
    output_dir: str,
    circuit: Any,
    filename: str = "circuit.png",
) -> str:
    ensure_dir(output_dir)
    try:
        fig = circuit.draw(output="mpl", fold=100)
        path = f"{output_dir}/{filename}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        path = f"{output_dir}/circuit.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(circuit.draw(output="text"))
        return path


def save_pennylane_circuit_image(
    output_dir: str,
    qnode: Any,
    filename: str = "circuit.png",
) -> str:
    ensure_dir(output_dir)
    path = f"{output_dir}/{filename}"
    try:
        drawer = qml.draw_mpl(qnode)
        fig = drawer()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        text = qml.draw(qnode)()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.0, 1.0, text, fontfamily="monospace", fontsize=8, va="top")
        ax.axis("off")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path
