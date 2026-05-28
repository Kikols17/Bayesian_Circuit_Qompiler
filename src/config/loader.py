from __future__ import annotations

from typing import Any, Dict, List

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

_SUPPORTED_ENCODINGS = ("binary", "one_hot", "sparse_topk")
_TOPK_REQUIRED_KEYS = ("k", "marginalize")


def _get_required(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def _validate_encoding_params(
    encoding: str, params: Dict[str, Any], query: List[str]
) -> None:
    if encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(
            f"inference.encoding={encoding!r} is not supported. "
            f"Supported values: {list(_SUPPORTED_ENCODINGS)}"
        )

    if encoding == "binary":
        if params:
            raise ValueError(
                f"inference.encoding={encoding!r} accepts no parameters; "
                f"got encoding_params={params!r}. Use 'encoding_params: {{}}' explicitly."
            )
        return

    if encoding == "one_hot" and not params:
        return

    missing = [k for k in _TOPK_REQUIRED_KEYS if k not in params]
    if missing:
        raise ValueError(
            f"inference.encoding={encoding!r} with encoding_params requires the keys: "
            f"{list(_TOPK_REQUIRED_KEYS)}, but missing: {missing}. "
            "Required schema: {k: <positive int>, marginalize: <list of query names>}. "
            "Pass `encoding_params: {}` for full-posterior one_hot (no truncation)."
        )
    extra = [k for k in params if k not in _TOPK_REQUIRED_KEYS]
    if extra:
        raise ValueError(
            f"inference.encoding={encoding!r} got unknown encoding_params keys: {extra}. "
            f"Allowed keys: {list(_TOPK_REQUIRED_KEYS)}."
        )

    k = params["k"]
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(
            f"inference.encoding_params.k must be a positive integer, got {k!r}"
        )

    marginalize = params["marginalize"]
    if not isinstance(marginalize, list) or not all(isinstance(m, str) for m in marginalize):
        raise ValueError(
            "inference.encoding_params.marginalize must be a list of strings "
            f"(query variable names to marginalize), got {marginalize!r}"
        )
    unknown = [m for m in marginalize if m not in query]
    if unknown:
        raise ValueError(
            f"inference.encoding_params.marginalize references variables not in query: "
            f"{unknown}. query={query}"
        )
    if marginalize and len(marginalize) >= len(query):
        raise ValueError(
            "inference.encoding_params.marginalize cannot remove every query variable; "
            f"got marginalize={marginalize} for query={query}"
        )


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
        params=backend_raw.get("params") or {},
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

    encoding = _get_required(inference_raw, "encoding")
    if not isinstance(encoding, str) or not encoding:
        raise ValueError(
            "inference.encoding must be a non-empty string "
            "(one of: 'binary', 'one_hot', 'sparse_topk')"
        )

    if "encoding_params" not in inference_raw:
        raise ValueError(
            "inference.encoding_params is required (use '{}' if the chosen "
            "encoding takes no parameters; sparse_topk requires k and marginalize)"
        )
    encoding_params = inference_raw["encoding_params"]
    if encoding_params is None:
        encoding_params = {}
    if not isinstance(encoding_params, dict):
        raise ValueError(
            f"inference.encoding_params must be a mapping, got {type(encoding_params).__name__}"
        )

    _validate_encoding_params(encoding, encoding_params, inference_raw.get("query", []))

    inference = InferenceConfig(
        compiler=_get_required(inference_raw, "compiler"),
        circuit=circuit,
        backend=backend,
        baseline=baseline,
        encoding=encoding,
        encoding_params=encoding_params,
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
