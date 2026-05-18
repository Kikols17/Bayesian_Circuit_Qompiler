from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.config.loader import load_config
from src.execution.pennylane_backend import run_pennylane
from src.execution.qiskit_backend import run_qiskit
from src.metrics.plots import (
    save_probabilities_plot,
    compute_prob_history_from_samples,
    save_probability_evolution_plot,
)
from src.metrics.markdown_report import (
    collect_network_stats,
    collect_system_info,
    compute_distribution_metrics,
    write_markdown_report,
)
from src.metrics.tracker import save_metrics
from src.metrics import wandb_logger
from src.network_builder import get_network_builder
from src.networks.export import save_cpd_markdown
from src.networks.visualize import save_bayesian_network_plot
from src.baseline import get_baseline
from src.qompiler import get_qompiler
from src.utils.io import ensure_dir, write_json, write_yaml


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def run_pipeline(config_path: str) -> Dict[str, Any]:
    _load_env()
    config = load_config(config_path)
    wandb_logger.init_run(config)
    output_base = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(output_base.parent / f"{timestamp}_{output_base.name}")
    ensure_dir(output_dir)

    write_yaml(f"{output_dir}/config_snapshot.yaml", {
        "experiment": {
            "name": config.name,
            "requested_output_dir": config.output_dir,
            "output_dir": output_dir,
        },
        "network": {
            "builder": config.network.builder,
            "save_image": config.network.save_image,
            "seed": config.network.seed,
            "type": config.network.type,
            "params": config.network.params,
        },
        "inference": {
            "compiler": config.inference.compiler,
            "encoding": config.inference.encoding,
            "baseline": {
                "enabled": config.inference.baseline.enabled,
                "method": config.inference.baseline.method,
                "params": config.inference.baseline.params,
            },
            "circuit": {
                "shots": config.inference.circuit.shots,
                "seed": config.inference.circuit.seed,
            },
            "backend": {
                "type": config.inference.backend.type,
                "device": config.inference.backend.device,
                "wires": config.inference.backend.wires,
            },
            "evidence": config.inference.evidence,
            "query": config.inference.query,
        },
    })

    network_builder = get_network_builder(config.network.builder)
    network = network_builder.build(config.network)
    wandb_logger.log_network_metrics(network)
    if config.network.save_image:
        network_image_path = save_bayesian_network_plot(output_dir, network)
    else:
        network_image_path = None

    if config.output.save_cpd_md:
        cpd_md_path = save_cpd_markdown(output_dir, network)
    else:
        cpd_md_path = None

    qompiler = get_qompiler(config.inference.compiler)

    baseline_result = None
    baseline_time = None
    if config.inference.baseline.enabled:
        baseline_runner = get_baseline(config.inference.baseline.method)
        baseline_start = time.perf_counter()
        baseline_result = baseline_runner.run(
            network,
            config.inference.evidence,
            config.inference.query,
            config.inference.baseline.params,
        )
        baseline_time = time.perf_counter() - baseline_start
        write_json(f"{output_dir}/baseline_result.json", baseline_result)

    compile_start = time.perf_counter()
    circuit_spec = qompiler.compile(
        network,
        config.inference.evidence,
        config.inference.query,
        encoding=config.inference.encoding,
    )
    compile_time = time.perf_counter() - compile_start
    wandb_logger.log_compile_metrics(circuit_spec, compile_time)

    run_start = time.perf_counter()
    if config.inference.backend.type.startswith("pennylane"):
        run_result = run_pennylane(
            output_dir=output_dir,
            spec=circuit_spec,
            circuit_config=config.inference.circuit,
            backend_config=config.inference.backend,
            evidence=config.inference.evidence,
            query=config.inference.query,
            save_circuit_image=config.output.save_circuit_image,
        )
    elif config.inference.backend.type.startswith("qiskit"):
        run_result = run_qiskit(
            output_dir=output_dir,
            spec=circuit_spec,
            circuit_config=config.inference.circuit,
            backend_config=config.inference.backend,
            evidence=config.inference.evidence,
            query=config.inference.query,
            save_circuit_image=config.output.save_circuit_image,
        )
    else:
        raise ValueError(f"Unsupported backend: {config.inference.backend.type}")
    run_time = time.perf_counter() - run_start

    baseline_distribution = None
    if baseline_result is not None:
        baseline_distribution = baseline_result.get("distribution")

    plot_path = None
    if isinstance(run_result.get("probs"), list):
        plot_path = save_probabilities_plot(
            output_dir,
            run_result["probs"],
            title=f"{config.name} probabilities",
            query=config.inference.query,
            wires_map=circuit_spec.wires_map,
            baseline_distribution=baseline_distribution,
        )

    # compute and save probability evolution (top outcomes by final prob)
    evolution_path = None
    hist = None
    matched_shots = None
    raw_samples = run_result.get("raw")
    if raw_samples is not None:
        try:
            # Only post-select on evidence nodes that have wires in the circuit.
            # QAA pre-conditions evidence out of the circuit, so those nodes are absent from wires_map.
            measurable_evidence = {k: v for k, v in config.inference.evidence.items() if k in circuit_spec.wires_map}
            hist, matched_shots = compute_prob_history_from_samples(
                raw_samples, circuit_spec.wires_map, config.inference.query, measurable_evidence
            )
        except Exception as exc:
            print(f"[evolution] compute_prob_history_from_samples failed: {exc}")
            hist, matched_shots = None, None

    if hist:
        try:
            evolution_path = save_probability_evolution_plot(
                output_dir,
                hist,
                title=f"{config.name} probability evolution",
                filename="prob_evolution_top10.png",
                query=config.inference.query,
                wires_map=circuit_spec.wires_map,
                top_n=10,
            )
        except Exception:
            evolution_path = None
        try:
            samples_for_json = raw_samples.tolist() if isinstance(raw_samples, np.ndarray) else list(raw_samples)
            write_json(f"{output_dir}/raw_samples.json", {"samples": samples_for_json})
        except Exception:
            pass

    def _format_quantum_distribution(
        probs: List[float], query: List[str], wires_map: Dict[str, List[int]]
    ) -> Dict[str, float]:
        if len(query) <= 1:
            return {str(idx): float(val) for idx, val in enumerate(probs)}

        total_bits = int(math.log2(len(probs))) if len(probs) > 0 else 0
        if 2 ** total_bits != len(probs):
            return {str(idx): float(val) for idx, val in enumerate(probs)}

        key_distribution: Dict[str, float] = {}
        for idx, prob in enumerate(probs):
            if prob <= 0.0:
                continue

            bitstr = format(idx, f"0{total_bits}b")
            position = 0
            coords: list[str] = []
            for node in query:
                num_bits = len(wires_map[node])
                node_bits = bitstr[position : position + num_bits]
                position += num_bits
                coords.append(str(int(node_bits, 2)))

            key = ",".join(f"{node}={value}" for node, value in zip(query, coords))
            key_distribution[key] = float(prob)

        return key_distribution

    def _distribution_from_result(result: Dict[str, Any]) -> Dict[str, float] | None:
        probs = result.get("probs")
        if isinstance(probs, list):
            return _format_quantum_distribution(probs, config.inference.query, circuit_spec.wires_map)
        if isinstance(probs, dict):
            return {str(key): float(val) for key, val in probs.items()}
        return None

    def _kl_divergence(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        eps = 1e-12
        keys = set(p_dist.keys()) | set(q_dist.keys())
        kl = 0.0
        for key in keys:
            p = p_dist.get(key, 0.0)
            q = q_dist.get(key, 0.0)
            if p <= 0.0:
                continue
            q = max(q, eps)
            kl += p * math.log(p / q)
        return kl

    baseline_kl = None
    quantum_distribution = _distribution_from_result(run_result)
    if baseline_result is not None:
        baseline_distribution = baseline_result.get("distribution")
        if isinstance(baseline_distribution, dict) and quantum_distribution is not None:
            baseline_kl = _kl_divergence(baseline_distribution, quantum_distribution)

    wandb_logger.log_run_metrics(run_result, run_time, config.inference.circuit.shots, matched_shots)
    wandb_logger.log_accuracy_metrics(baseline_distribution, quantum_distribution, baseline_kl, baseline_time, run_time)
    wandb_logger.log_convergence_series(hist or [], baseline_distribution, step_size=10)

    result_payload = {
        "quantum_result": run_result,
        "baseline_result": baseline_result,
        "baseline_kl": baseline_kl,
        "baseline_distribution": baseline_distribution,
        "quantum_distribution": quantum_distribution,
    }

    shots_requested = config.inference.circuit.shots
    effective_samples = matched_shots if matched_shots is not None else shots_requested
    evidence_match_rate = (effective_samples / shots_requested) if shots_requested else None
    shots_per_effective = (shots_requested / effective_samples) if effective_samples else None

    accuracy_metrics = compute_distribution_metrics(baseline_distribution, quantum_distribution)
    if baseline_time is not None and run_time > 0:
        accuracy_metrics["speedup"] = baseline_time / run_time
    else:
        accuracy_metrics["speedup"] = None

    metrics_payload = {
        "experiment": config.name,
        "timestamp": timestamp,
        "compiler": circuit_spec.name,
        "backend": config.inference.backend.type,
        "baseline_enabled": config.inference.baseline.enabled,
        "baseline_method": config.inference.baseline.method,
        "baseline_time_s": baseline_time,
        "baseline_result": baseline_result,
        "baseline_kl": baseline_kl,
        "compile_time_s": compile_time,
        "run_time_s": run_time,
        "output_dir": output_dir,
        "plot": plot_path,
        "network_image": network_image_path,
        "cpd_markdown": cpd_md_path,
        "result": run_result,
        "circuit_metadata": circuit_spec.metadata,
        "circuit_stats": run_result.get("circuit_stats"),
        "transpiled_stats": run_result.get("transpiled_stats"),
        "circuit_image": run_result.get("circuit_image"),
        "config": {
            "compiler": config.inference.compiler,
            "encoding": config.inference.encoding,
            "backend_type": config.inference.backend.type,
            "device": config.inference.backend.device,
            "shots": shots_requested,
            "circuit_seed": config.inference.circuit.seed,
            "network_seed": config.network.seed,
            "evidence": config.inference.evidence,
            "query": config.inference.query,
            "baseline_enabled": config.inference.baseline.enabled,
            "baseline_method": config.inference.baseline.method,
        },
        "network_stats": collect_network_stats(network),
        "execution": {
            "shots_requested": shots_requested,
            "effective_samples": effective_samples,
            "evidence_match_rate": evidence_match_rate,
            "shots_per_effective_sample": shots_per_effective,
        },
        "accuracy": accuracy_metrics,
        "system": collect_system_info(),
        "artifacts": {
            "config_snapshot": f"{output_dir}/config_snapshot.yaml",
            "metrics_json": f"{output_dir}/metrics.json",
            "result_json": f"{output_dir}/result.json",
            "probs_plot": plot_path,
            "prob_evolution_plot": evolution_path,
            "network_image": network_image_path,
            "cpd_markdown": cpd_md_path,
            "circuit_image": run_result.get("circuit_image"),
            "raw_samples": f"{output_dir}/raw_samples.json" if hist else None,
        },
    }

    save_metrics(output_dir, metrics_payload)
    write_json(f"{output_dir}/result.json", result_payload)
    report_path = write_markdown_report(output_dir, metrics_payload)
    metrics_payload["report"] = report_path

    wandb_logger.finish()
    return metrics_payload
