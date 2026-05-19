from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NetworkConfig:
    type: str
    builder: str = "generator"
    save_image: bool = False
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitConfig:
    shots: int = 1024
    seed: Optional[int] = None


@dataclass
class OutputConfig:
    save_cpd_md: bool = False
    save_circuit_image: bool = False


@dataclass
class BackendConfig:
    type: str
    device: Optional[str] = None
    wires: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineConfig:
    enabled: bool = False
    method: str = "variable_elimination"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    compiler: str
    circuit: CircuitConfig
    backend: BackendConfig
    encoding: str
    encoding_params: Dict[str, Any]
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    evidence: Dict[str, int] = field(default_factory=dict)
    query: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    output: OutputConfig
    network: NetworkConfig
    inference: InferenceConfig
