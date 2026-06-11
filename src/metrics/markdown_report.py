from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata as _metadata
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.utils.io import ensure_dir


_REPORT_FILENAME = "report.md"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if np.isnan(value):
            return "—"
        if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value) if value else "—"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items()) if value else "—"
    return str(value)


def _table(rows: Iterable[tuple]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for label, value in rows:
        lines.append(f"| {label} | {_fmt(value)} |")
    return "\n".join(lines)


def _gate_table(gate_counts: Optional[Dict[str, int]]) -> str:
    if not gate_counts:
        return "_No gates recorded._"
    lines = ["| Gate | Count |", "| --- | ---: |"]
    for name, count in sorted(gate_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        return _metadata.version(name)
    except Exception:
        return None


def _system_info() -> Dict[str, Optional[str]]:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "git_commit": _git_commit(),
        "pennylane": _pkg_version("pennylane"),
        "qiskit": _pkg_version("qiskit"),
        "qiskit_aer": _pkg_version("qiskit-aer"),
        "pgmpy": _pkg_version("pgmpy"),
        "numpy": _pkg_version("numpy"),
    }


def _network_stats(network) -> Dict[str, Any]:
    try:
        nodes = list(network.nodes())
        edges = list(network.edges())
        cpds = network.get_cpds()
        cardinalities = [int(c.variable_card) for c in cpds]
        in_degrees = [network.in_degree(n) for n in nodes]
        total_cpt = sum(c.values.size for c in cpds)
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
            "max_cardinality": max(cardinalities) if cardinalities else 0,
            "avg_cardinality": float(np.mean(cardinalities)) if cardinalities else 0.0,
            "total_cpt_entries": int(total_cpt),
        }
    except Exception:
        return {}


def _distribution_metrics(
    baseline: Optional[Dict[str, float]], quantum: Optional[Dict[str, float]]
) -> Dict[str, Optional[float]]:
    if not baseline or not quantum:
        return {
            "kl_divergence": None,
            "tv_distance": None,
            "hellinger_distance": None,
            "js_divergence": None,
            "l1_error": None,
            "l2_error": None,
            "top1_rank_correct": None,
            "max_prob_error": None,
        }

    eps = 1e-12
    keys = set(baseline.keys()) | set(quantum.keys())
    p = np.array([baseline.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([quantum.get(k, 0.0) for k in keys], dtype=float)

    kl = float(np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0.0)))
    tv = float(0.5 * np.sum(np.abs(p - q)))
    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    m = 0.5 * (p + q)
    js = 0.5 * float(
        np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
        + np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
    )
    l1 = float(np.sum(np.abs(p - q)))
    l2 = float(np.sqrt(np.sum((p - q) ** 2)))

    baseline_top = max(baseline, key=lambda k: baseline[k])
    quantum_top = max(quantum, key=lambda k: quantum[k])
    return {
        "kl_divergence": kl,
        "tv_distance": tv,
        "hellinger_distance": hellinger,
        "js_divergence": js,
        "l1_error": l1,
        "l2_error": l2,
        "top1_rank_correct": int(baseline_top == quantum_top),
        "max_prob_error": float(abs(baseline[baseline_top] - quantum.get(baseline_top, 0.0))),
    }


def _classify_gates(gate_counts: Dict[str, int], gate_sizes: Optional[Dict[int, int]]) -> Dict[str, int]:
    """Bucket gate counts by qubit arity.

    Prefers PennyLane's `gate_sizes` (arity -> count) when present; otherwise
    classifies by Qiskit gate names.
    """
    if gate_sizes:
        return {
            "single_qubit": int(gate_sizes.get(1, 0)),
            "two_qubit": int(gate_sizes.get(2, 0)),
            "multi_qubit": int(sum(c for k, c in gate_sizes.items() if k >= 3)),
        }

    two_qubit_names = {"cx", "cnot", "cry", "crx", "crz", "cz", "cu", "cp", "swap", "iswap", "ecr"}
    multi_qubit_names = {"ccx", "toffoli", "mcx", "ccz", "cswap", "fredkin"}
    single, two, multi = 0, 0, 0
    for name, count in gate_counts.items():
        n = name.lower()
        if n in multi_qubit_names or n.startswith("mcx") or n.startswith("c3") or n.startswith("c4"):
            multi += count
        elif n in two_qubit_names or n.startswith("c") and not n.startswith("ccx"):
            two += count
        else:
            single += count
    return {"single_qubit": single, "two_qubit": two, "multi_qubit": multi}


def _summary_row(payload: Dict[str, Any]) -> List[tuple]:
    config = payload.get("config") or {}
    network = payload.get("network_stats") or {}
    circuit = payload.get("circuit_stats") or {}
    transpiled = payload.get("transpiled_stats") or {}
    accuracy = payload.get("accuracy") or {}
    execution = payload.get("execution") or {}
    system = payload.get("system") or {}
    meta = payload.get("circuit_metadata") or {}

    gate_counts = circuit.get("operation_counts") or {}
    gate_buckets = _classify_gates(gate_counts, circuit.get("gate_sizes"))
    num_ops = circuit.get("num_operations") or 0
    two_q_ratio = (gate_buckets["two_qubit"] / num_ops) if num_ops else None
    gates_per_qubit = (num_ops / circuit["num_wires"]) if circuit.get("num_wires") else None

    return [
        ("experiment", payload.get("experiment")),
        ("timestamp", payload.get("timestamp")),
        ("git_commit", system.get("git_commit")),
        ("compiler", config.get("compiler")),
        ("encoding", config.get("encoding")),
        ("backend_type", config.get("backend_type")),
        ("device", config.get("device")),
        ("shots", config.get("shots")),
        ("network_seed", config.get("network_seed")),
        ("evidence", config.get("evidence")),
        ("query", config.get("query")),
        ("baseline_method", config.get("baseline_method")),
        ("num_nodes", network.get("num_nodes")),
        ("num_edges", network.get("num_edges")),
        ("max_in_degree", network.get("max_in_degree")),
        ("avg_in_degree", network.get("avg_in_degree")),
        ("max_cardinality", network.get("max_cardinality")),
        ("avg_cardinality", network.get("avg_cardinality")),
        ("total_cpt_entries", network.get("total_cpt_entries")),
        ("compile_time_s", payload.get("compile_time_s")),
        ("mode", meta.get("mode")),
        ("num_grover_iterations", meta.get("num_iterations")),
        ("approximation", meta.get("approximation")),
        ("num_qubits", circuit.get("num_wires")),
        ("num_clbits", circuit.get("num_clbits")),
        ("circuit_depth", circuit.get("depth")),
        ("num_operations", num_ops),
        ("num_parameters", circuit.get("num_parameters")),
        ("num_measurements", circuit.get("num_measurements")),
        ("single_qubit_gates", gate_buckets["single_qubit"]),
        ("two_qubit_gates", gate_buckets["two_qubit"]),
        ("multi_qubit_gates", gate_buckets["multi_qubit"]),
        ("two_qubit_gate_ratio", two_q_ratio),
        ("gates_per_qubit", gates_per_qubit),
        ("transpiled_depth", transpiled.get("depth")),
        ("transpiled_num_qubits", transpiled.get("num_wires")),
        ("transpiled_num_operations", transpiled.get("num_operations")),
        ("run_time_s", payload.get("run_time_s")),
        ("shots_requested", execution.get("shots_requested")),
        ("effective_samples", execution.get("effective_samples")),
        ("evidence_match_rate", execution.get("evidence_match_rate")),
        ("shots_per_effective_sample", execution.get("shots_per_effective_sample")),
        ("baseline_time_s", payload.get("baseline_time_s")),
        ("speedup_quantum_vs_baseline", accuracy.get("speedup")),
        ("kl_divergence", accuracy.get("kl_divergence")),
        ("tv_distance", accuracy.get("tv_distance")),
        ("hellinger_distance", accuracy.get("hellinger_distance")),
        ("js_divergence", accuracy.get("js_divergence")),
        ("l1_error", accuracy.get("l1_error")),
        ("l2_error", accuracy.get("l2_error")),
        ("top1_rank_correct", accuracy.get("top1_rank_correct")),
        ("max_prob_error", accuracy.get("max_prob_error")),
        ("python", system.get("python")),
        ("pennylane", system.get("pennylane")),
        ("qiskit", system.get("qiskit")),
    ]


def _summary_table(payload: Dict[str, Any]) -> str:
    row = _summary_row(payload)
    headers = [k for k, _ in row]
    values = [_fmt(v) for _, v in row]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    value_line = "| " + " | ".join(values) + " |"
    return "\n".join([header_line, sep_line, value_line])


def build_report(payload: Dict[str, Any]) -> str:
    config = payload.get("config") or {}
    network = payload.get("network_stats") or {}
    circuit = payload.get("circuit_stats") or {}
    transpiled = payload.get("transpiled_stats") or {}
    accuracy = payload.get("accuracy") or {}
    execution = payload.get("execution") or {}
    artifacts = payload.get("artifacts") or {}
    system = payload.get("system") or {}

    gate_counts = circuit.get("operation_counts") or {}
    gate_buckets = _classify_gates(gate_counts, circuit.get("gate_sizes"))
    num_ops = circuit.get("num_operations") or 0
    two_qubit_ratio = (gate_buckets["two_qubit"] / num_ops) if num_ops else None
    gates_per_qubit = (num_ops / circuit["num_wires"]) if circuit.get("num_wires") else None

    sections: List[str] = []
    sections.append(f"# {payload.get('experiment', 'Run')} — circuit metrics\n")

    sections.append("## Run\n")
    sections.append(_table([
        ("experiment", payload.get("experiment")),
        ("timestamp", payload.get("timestamp")),
        ("output_dir", payload.get("output_dir")),
        ("git_commit", system.get("git_commit")),
        ("python", system.get("python")),
        ("platform", system.get("platform")),
        ("pennylane", system.get("pennylane")),
        ("qiskit", system.get("qiskit")),
        ("qiskit-aer", system.get("qiskit_aer")),
        ("pgmpy", system.get("pgmpy")),
        ("numpy", system.get("numpy")),
    ]))
    sections.append("")

    if circuit.get("status") == "job_failed" or circuit.get("error"):
        sections.append("## ⚠️ Run status\n")
        sections.append(_table([
            ("status", circuit.get("status")),
            ("error", circuit.get("error")),
        ]))
        sections.append("")

    sections.append("## Configuration\n")
    sections.append(_table([
        ("compiler", config.get("compiler")),
        ("encoding", config.get("encoding")),
        ("backend_type", config.get("backend_type")),
        ("device", config.get("device")),
        ("shots", config.get("shots")),
        ("circuit_seed", config.get("circuit_seed")),
        ("network_seed", config.get("network_seed")),
        ("evidence", config.get("evidence")),
        ("query", config.get("query")),
        ("baseline_enabled", config.get("baseline_enabled")),
        ("baseline_method", config.get("baseline_method")),
    ]))
    sections.append("")

    sections.append("## Bayesian network\n")
    sections.append(_table([
        ("num_nodes", network.get("num_nodes")),
        ("num_edges", network.get("num_edges")),
        ("max_in_degree", network.get("max_in_degree")),
        ("avg_in_degree", network.get("avg_in_degree")),
        ("max_cardinality", network.get("max_cardinality")),
        ("avg_cardinality", network.get("avg_cardinality")),
        ("total_cpt_entries", network.get("total_cpt_entries")),
    ]))
    sections.append("")

    sections.append("## Compilation\n")
    sections.append(_table([
        ("compile_time_s", payload.get("compile_time_s")),
        ("mode", (payload.get("circuit_metadata") or {}).get("mode")),
        ("num_grover_iterations", (payload.get("circuit_metadata") or {}).get("num_iterations")),
        ("approximation", (payload.get("circuit_metadata") or {}).get("approximation")),
    ]))
    sections.append("")

    sections.append("## Circuit structure\n")
    sections.append(_table([
        ("num_qubits", circuit.get("num_wires")),
        ("num_clbits", circuit.get("num_clbits")),
        ("circuit_depth", circuit.get("depth")),
        ("num_operations", num_ops),
        ("num_parameters", circuit.get("num_parameters")),
        ("num_measurements", circuit.get("num_measurements")),
        ("single_qubit_gates", gate_buckets["single_qubit"]),
        ("two_qubit_gates", gate_buckets["two_qubit"]),
        ("multi_qubit_gates", gate_buckets["multi_qubit"]),
        ("two_qubit_gate_ratio", two_qubit_ratio),
        ("gates_per_qubit", gates_per_qubit),
    ]))
    sections.append("\n### Gate counts\n")
    sections.append(_gate_table(gate_counts))
    sections.append("")

    if transpiled:
        sections.append("## Transpiled circuit (hardware)\n")
        sections.append(_table([
            ("transpiled_depth", transpiled.get("depth")),
            ("transpiled_num_qubits", transpiled.get("num_wires")),
            ("transpiled_num_operations", transpiled.get("num_operations")),
            ("basis_gates", transpiled.get("basis_gates")),
        ]))
        sections.append("\n### Transpiled gate counts\n")
        sections.append(_gate_table(transpiled.get("operation_counts")))
        sections.append("")

    sections.append("## Execution\n")
    sections.append(_table([
        ("run_time_s", payload.get("run_time_s")),
        ("shots_requested", execution.get("shots_requested")),
        ("effective_samples", execution.get("effective_samples")),
        ("evidence_match_rate", execution.get("evidence_match_rate")),
        ("shots_per_effective_sample", execution.get("shots_per_effective_sample")),
    ]))
    sections.append("")

    sections.append("## Accuracy vs baseline\n")
    sections.append(_table([
        ("baseline_method", config.get("baseline_method")),
        ("baseline_time_s", payload.get("baseline_time_s")),
        ("speedup_quantum_vs_baseline", accuracy.get("speedup")),
        ("kl_divergence", accuracy.get("kl_divergence")),
        ("tv_distance", accuracy.get("tv_distance")),
        ("hellinger_distance", accuracy.get("hellinger_distance")),
        ("js_divergence", accuracy.get("js_divergence")),
        ("l1_error", accuracy.get("l1_error")),
        ("l2_error", accuracy.get("l2_error")),
        ("top1_rank_correct", accuracy.get("top1_rank_correct")),
        ("max_prob_error", accuracy.get("max_prob_error")),
    ]))
    sections.append("")

    if artifacts:
        sections.append("## Artifacts\n")
        sections.append(_table([(k, v) for k, v in artifacts.items()]))
        sections.append("")

    sections.append("## Summary row\n")
    sections.append(_summary_table(payload))
    sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def write_markdown_report(output_dir: str, payload: Dict[str, Any]) -> str:
    """Render `payload` as a markdown report and write it to `<output_dir>/report.md`."""
    ensure_dir(output_dir)
    path = os.path.join(output_dir, _REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    return path


def collect_system_info() -> Dict[str, Optional[str]]:
    """Return Python, platform, git commit, and quantum-stack package versions."""
    return _system_info()


def collect_network_stats(network) -> Dict[str, Any]:
    """Return node/edge/cardinality/CPT summary for a pgmpy DiscreteBayesianNetwork."""
    return _network_stats(network)


def compute_distribution_metrics(
    baseline: Optional[Dict[str, float]], quantum: Optional[Dict[str, float]]
) -> Dict[str, Optional[float]]:
    """Compute KL, TV, Hellinger, JS, L1, L2, top1 accuracy, and max prob error."""
    return _distribution_metrics(baseline, quantum)
