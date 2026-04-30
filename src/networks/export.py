from __future__ import annotations

from typing import Any

from src.utils.io import ensure_dir


def save_cpd_markdown(
    output_dir: str,
    model: Any,
    filename: str = "cpd.md",
) -> str:
    ensure_dir(output_dir)
    path = f"{output_dir}/{filename}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# CPD Tables\n\n")
        for node in model.nodes():
            cpd = model.get_cpds(node)
            evidence_vars = cpd.get_evidence()
            f.write(f"## {cpd.variable} (card={cpd.variable_card})\n\n")
            if evidence_vars:
                f.write(f"**Evidence:** {', '.join(evidence_vars)}\n\n")

            values = cpd.get_values()
            if values.ndim == 1:
                f.write("| State | Probability |\n")
                f.write("|---|---:|\n")
                for state_idx, prob in enumerate(values):
                    f.write(f"| {state_idx} | {float(prob):.6f} |\n")
            else:
                header = ["State"]
                if evidence_vars:
                    parent_cards = [model.get_cpds(ev).variable_card for ev in evidence_vars]
                    for col in range(values.shape[1]):
                        temp = col
                        state_indexes = []
                        for card in reversed(parent_cards):
                            state_indexes.insert(0, temp % card)
                            temp //= card
                        header.append(
                            ",".join(f"{ev}={idx}" for ev, idx in zip(evidence_vars, state_indexes))
                        )
                else:
                    header.extend(f"col{col}" for col in range(values.shape[1]))
                f.write("| " + " | ".join(header) + " |\n")
                f.write("|" + "---|" * len(header) + "\n")
                for state_idx in range(values.shape[0]):
                    row = [str(state_idx)]
                    row.extend(f"{float(values[state_idx, col]):.6f}" for col in range(values.shape[1]))
                    f.write("| " + " | ".join(row) + " |\n")
            f.write("\n---\n\n")

    return path
