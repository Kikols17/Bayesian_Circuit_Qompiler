from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np


def _wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def init_run(config) -> None:
    w = _wandb()
    if w is None:
        return
    try:
        w.init(
            project=os.environ.get("WANDB_PROJECT", "bayesian-circuit-compiler"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=config.name,
            config={
                "experiment": config.name,
                "network/builder": config.network.builder,
                "network/type": config.network.type,
                "network/seed": config.network.seed,
                "inference/compiler": config.inference.compiler,
                "inference/encoding": config.inference.encoding,
                "inference/shots": config.inference.circuit.shots,
                "inference/backend": config.inference.backend.type,
                "inference/device": config.inference.backend.device,
                "inference/evidence": str(config.inference.evidence),
                "inference/query": config.inference.query,
                "baseline/enabled": config.inference.baseline.enabled,
                "baseline/method": config.inference.baseline.method,
            },
        )
    except Exception:
        pass


def log_network_metrics(network) -> None:
    w = _wandb()
    if w is None or w.run is None:
        return
    try:
        cpds = network.get_cpds()
        cardinalities = [int(cpd.variable_card) for cpd in cpds]
        total_cpt = sum(cpd.values.size for cpd in cpds)
        in_degrees = [network.in_degree(node) for node in network.nodes()]
        w.log({
            "network/num_nodes": len(list(network.nodes())),
            "network/num_edges": len(list(network.edges())),
            "network/max_in_degree": max(in_degrees) if in_degrees else 0,
            "network/max_cardinality": max(cardinalities) if cardinalities else 0,
            "network/avg_cardinality": float(np.mean(cardinalities)) if cardinalities else 0.0,
            "network/total_cpt_entries": total_cpt,
        })
    except Exception:
        pass


def log_compile_metrics(circuit_spec, compile_time: float) -> None:
    w = _wandb()
    if w is None or w.run is None:
        return
    try:
        w.log({
            "compile/num_qubits": circuit_spec.num_wires,
            "compile/time_s": compile_time,
        })
    except Exception:
        pass


def log_run_metrics(
    run_result: Dict[str, Any],
    run_time: float,
    shots: int,
    matched_shots: Optional[int],
) -> None:
    w = _wandb()
    if w is None or w.run is None:
        return
    try:
        effective = matched_shots if matched_shots is not None else shots
        match_rate = effective / shots if shots > 0 else 0.0

        metrics: Dict[str, Any] = {
            "run/time_s": run_time,
            "run/shots": shots,
            "run/evidence_match_rate": match_rate,
            "run/effective_samples": effective,
        }

        if effective > 0:
            metrics["efficiency/shots_per_effective_sample"] = shots / effective

        circuit_stats = run_result.get("circuit_stats") or {}
        num_ops = circuit_stats.get("num_operations")
        num_wires = circuit_stats.get("num_wires")
        depth = circuit_stats.get("depth")
        op_counts: Dict[str, int] = circuit_stats.get("operation_counts") or {}

        if num_ops is not None:
            metrics["compile/num_gates"] = num_ops
        if depth is not None:
            metrics["compile/circuit_depth"] = depth
        if num_wires and num_ops:
            metrics["efficiency/gates_per_qubit"] = num_ops / num_wires

        for gate_name, count in op_counts.items():
            metrics[f"compile/gate_counts/{gate_name}"] = count

        if num_ops:
            two_qubit_names = {"cx", "cnot", "cry", "crx", "crz", "cz", "ccx", "cu", "cp"}
            two_q = sum(v for k, v in op_counts.items() if k.lower() in two_qubit_names)
            metrics["compile/two_qubit_gate_ratio"] = two_q / num_ops

        w.log(metrics)
    except Exception:
        pass


def log_accuracy_metrics(
    baseline_dist: Optional[Dict[str, float]],
    quantum_dist: Optional[Dict[str, float]],
    baseline_kl: Optional[float],
    baseline_time: Optional[float],
    run_time: float,
) -> None:
    w = _wandb()
    if w is None or w.run is None:
        return
    try:
        metrics: Dict[str, Any] = {}

        if baseline_kl is not None:
            metrics["accuracy/kl_divergence"] = baseline_kl

        if baseline_dist and quantum_dist:
            keys = set(baseline_dist.keys()) | set(quantum_dist.keys())
            tv = 0.5 * sum(
                abs(baseline_dist.get(k, 0.0) - quantum_dist.get(k, 0.0)) for k in keys
            )
            metrics["accuracy/tv_distance"] = tv

            baseline_top = max(baseline_dist, key=lambda k: baseline_dist[k])
            quantum_top = max(quantum_dist, key=lambda k: quantum_dist[k])
            metrics["accuracy/top1_rank_correct"] = int(baseline_top == quantum_top)
            metrics["accuracy/max_prob_error"] = abs(
                baseline_dist[baseline_top] - quantum_dist.get(baseline_top, 0.0)
            )

        if baseline_time is not None and run_time > 0:
            metrics["speedup/quantum_vs_baseline"] = baseline_time / run_time

        if metrics:
            w.log(metrics)
    except Exception:
        pass


def log_convergence_series(
    prob_history: List[List[float]],
    baseline_dist: Optional[Dict[str, float]],
    step_size: int = 10,
) -> None:
    w = _wandb()
    if w is None or w.run is None or not prob_history:
        return
    try:
        num_outcomes = len(prob_history[0])

        # convert baseline dict to a dense array indexed by integer outcome
        baseline_arr: Optional[np.ndarray] = None
        if baseline_dist:
            arr = np.zeros(num_outcomes)
            for k, v in baseline_dist.items():
                # handles "3", "X=3", "X=3,Y=1" (takes last = segment for 1-var)
                idx_str = k.split("=")[-1].split(",")[0] if "=" in k else k
                try:
                    idx = int(idx_str)
                    if 0 <= idx < num_outcomes:
                        arr[idx] = float(v)
                except ValueError:
                    pass
            if arr.sum() > 0:
                baseline_arr = arr

        eps = 1e-12
        checkpoints = list(range(step_size - 1, len(prob_history), step_size))
        if not checkpoints or checkpoints[-1] != len(prob_history) - 1:
            checkpoints.append(len(prob_history) - 1)

        for i in checkpoints:
            q = np.array(prob_history[i], dtype=float)
            step_metrics: Dict[str, Any] = {"convergence/matched_shots": i + 1}

            if baseline_arr is not None:
                p = baseline_arr
                kl = float(np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0.0)))
                tv = float(0.5 * np.sum(np.abs(p - q)))
                step_metrics["convergence/kl_vs_shots"] = kl
                step_metrics["convergence/tv_vs_shots"] = tv

            step_metrics["convergence/top1_prob_vs_shots"] = float(np.max(q))

            w.log(step_metrics)
    except Exception:
        pass


def finish() -> None:
    w = _wandb()
    if w is None or w.run is None:
        return
    try:
        w.finish()
    except Exception:
        pass
