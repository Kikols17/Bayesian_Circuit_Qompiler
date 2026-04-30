from __future__ import annotations

from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import ensure_dir


def save_probabilities_plot(
    output_dir: str, probs: List[float], title: str, filename: str = "probs.png"
) -> str:
    ensure_dir(output_dir)
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
