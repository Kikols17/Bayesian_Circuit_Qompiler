from __future__ import annotations

from typing import Any, Dict

import yaml

from .types import (
    BackendConfig,
    BaselineConfig,
    CircuitConfig,
    ExperimentConfig,
    InferenceConfig,
    NetworkConfig,
    OutputConfig,
)


def _get_required(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    exp_raw = _get_required(raw, "experiment")
    network_raw = _get_required(raw, "network")
    inference_raw = _get_required(raw, "inference")

    network = NetworkConfig(
        builder=network_raw.get("builder", "generator"),
        type=_get_required(network_raw, "type"),
        save_image=bool(network_raw.get("save_image", False)),
        seed=network_raw.get("seed"),
        params=network_raw.get("params", {}),
    )

    circuit_raw = inference_raw.get("circuit", {})
    circuit = CircuitConfig(
        shots=int(circuit_raw.get("shots", 1024)),
        seed=circuit_raw.get("seed"),
    )

    backend_raw = _get_required(inference_raw, "backend")
    backend = BackendConfig(
        type=_get_required(backend_raw, "type"),
        device=backend_raw.get("device"),
        wires=backend_raw.get("wires"),
    )

    output_raw = exp_raw.get("output", {})
    output = OutputConfig(
        save_cpd_md=bool(output_raw.get("save_cpd_md", False)),
        save_circuit_image=bool(output_raw.get("save_circuit_image", False)),
    )

    baseline_raw = inference_raw.get("baseline", {})
    baseline = BaselineConfig(
        enabled=bool(baseline_raw.get("enabled", False)),
        method=baseline_raw.get("method", "variable_elimination"),
        params=baseline_raw.get("params", {}),
    )

    inference = InferenceConfig(
        compiler=_get_required(inference_raw, "compiler"),
        circuit=circuit,
        backend=backend,
        baseline=baseline,
        encoding=inference_raw.get("encoding", "binary"),
        evidence=inference_raw.get("evidence", {}),
        query=inference_raw.get("query", []),
    )

    return ExperimentConfig(
        name=_get_required(exp_raw, "name"),
        output_dir=_get_required(exp_raw, "output_dir"),
        output=output,
        network=network,
        inference=inference,
    )
