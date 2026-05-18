from __future__ import annotations

import json
import os
import pickle
import shlex
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec
from src.utils.env_validation import require_env


NAVIGATOR_REQUIRED_ENV = [
    "NAVIGATOR_USER",
    "NAVIGATOR_SSH_KEY",
    "NAVIGATOR_ACCOUNT",
    "NAVIGATOR_REMOTE_DIR",
    "NAVIGATOR_VENV",
]

_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "OUT_OF_MEMORY",
}


class NavigatorJobDetached(Exception):
    """Signals to the runner that a job was submitted in detach mode and
    metrics should be deferred until the user resumes the run."""

    def __init__(self, tag: str, jobid: str, remote_dir: str, checkpoint_path: str):
        super().__init__(f"Navigator job {jobid} submitted with tag '{tag}'")
        self.tag = tag
        self.jobid = jobid
        self.remote_dir = remote_dir
        self.checkpoint_path = checkpoint_path


def run_navigator(
    output_dir: str,
    spec: CircuitSpec,
    circuit_config: CircuitConfig,
    backend_config: BackendConfig,
    evidence: Dict[str, int],
    query: List[str],
    save_circuit_image: bool = False,
    network: Optional[DiscreteBayesianNetwork] = None,
    compiler: Optional[str] = None,
    encoding: str = "binary",
) -> Dict[str, Any]:
    env = require_env(NAVIGATOR_REQUIRED_ENV)
    host = os.environ.get("NAVIGATOR_HOST", "navigator.lca.uc.pt")
    user = env["NAVIGATOR_USER"]
    ssh_key = env["NAVIGATOR_SSH_KEY"]
    account = env["NAVIGATOR_ACCOUNT"]
    remote_base = env["NAVIGATOR_REMOTE_DIR"].rstrip("/")
    venv = env["NAVIGATOR_VENV"]

    params = backend_config.params or {}
    engine = (params.get("engine") or "pennylane").lower()
    if engine not in {"pennylane", "qiskit_aer"}:
        raise ValueError(
            f"navigator backend engine must be 'pennylane' or 'qiskit_aer', got {engine!r}"
        )

    checkpoint_path = Path(output_dir) / "navigator_checkpoint.json"

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        return _wait_and_collect(output_dir, host, user, ssh_key, checkpoint, params)

    if network is None or compiler is None:
        raise ValueError(
            "navigator backend (fresh submit) requires the network and compiler name"
        )

    partition = params.get("partition")
    if not partition:
        raise ValueError(
            "inference.backend.params.partition is required for the navigator backend"
        )

    tag = params.get("tag") or _autotag(spec.name)
    remote_dir = f"{remote_base}/{tag}"
    staging = Path(output_dir) / "_navigator" / tag
    staging.mkdir(parents=True, exist_ok=True)

    payload = {
        "compiler": compiler,
        "encoding": encoding,
        "network": network,
        "evidence": dict(evidence),
        "query": list(query),
        "circuit_config": {"shots": circuit_config.shots, "seed": circuit_config.seed},
        "engine": engine,
        "device": backend_config.device,
        "wires": backend_config.wires,
        "save_circuit_image": save_circuit_image,
    }
    (staging / "payload.pkl").write_bytes(pickle.dumps(payload, protocol=4))
    (staging / "remote_run.py").write_text(_REMOTE_RUN_PY)
    (staging / "submit.sbatch").write_text(_build_sbatch(tag, account, venv, params))

    project_src = Path(__file__).resolve().parents[2] / "src"
    _ssh(host, user, ssh_key, f"mkdir -p {shlex.quote(remote_dir)}")
    _rsync_to(host, user, ssh_key, f"{staging}/", f"{remote_dir}/")
    _rsync_to(host, user, ssh_key, f"{project_src}/", f"{remote_dir}/src/")

    submit_out = _ssh(
        host,
        user,
        ssh_key,
        f"cd {shlex.quote(remote_dir)} && sbatch --parsable submit.sbatch",
    ).strip()
    jobid = submit_out.split(";")[0].strip()
    if not jobid.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {submit_out!r}")

    checkpoint = {
        "tag": tag,
        "jobid": jobid,
        "remote_dir": remote_dir,
        "host": host,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "engine": engine,
        "device": backend_config.device,
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    if params.get("detach"):
        raise NavigatorJobDetached(tag, jobid, remote_dir, str(checkpoint_path))

    return _wait_and_collect(output_dir, host, user, ssh_key, checkpoint, params)


def _autotag(base_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{stamp}_{uuid.uuid4().hex[:6]}"


def _wait_and_collect(
    output_dir: str,
    host: str,
    user: str,
    ssh_key: str,
    checkpoint: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    poll_s = int(params.get("poll_interval_s", 30))
    jobid = str(checkpoint["jobid"])
    remote_dir = checkpoint["remote_dir"]
    tag = checkpoint["tag"]

    state = _wait_for_terminal(host, user, ssh_key, jobid, poll_s)

    local_artifacts = Path(output_dir) / "_navigator" / tag
    local_artifacts.mkdir(parents=True, exist_ok=True)
    _rsync_from(host, user, ssh_key, f"{remote_dir}/", f"{local_artifacts}/")

    if state != "COMPLETED":
        log_tail = ""
        for log_path in sorted(local_artifacts.glob("slurm-*.out")):
            tail = log_path.read_text().splitlines()[-80:]
            log_tail = "\n--- " + log_path.name + " (tail) ---\n" + "\n".join(tail)
        raise RuntimeError(
            f"Navigator job {jobid} ended in state {state}{log_tail}"
        )

    result_path = local_artifacts / "result.json"
    if not result_path.exists():
        raise RuntimeError(
            f"Navigator job {jobid} completed but result.json was not produced"
        )

    result = json.loads(result_path.read_text())
    raw_path = local_artifacts / "raw_samples.npy"
    raw_samples = np.load(raw_path) if raw_path.exists() else None

    return {
        "probs": result.get("probs"),
        "raw": raw_samples,
        "device": result.get("device"),
        "circuit_stats": result.get("circuit_stats"),
        "circuit_image": result.get("circuit_image"),
    }


def _build_sbatch(tag: str, account: str, venv: str, params: Dict[str, Any]) -> str:
    partition = params["partition"]
    time_limit = params.get("time_limit", "00:30:00")
    cpus = int(params.get("cpus", 8))
    mem = params.get("mem", "8G")
    nodes = int(params.get("nodes", 1))
    python_module = os.environ.get("NAVIGATOR_PYTHON_MODULE")
    module_line = f"module load {python_module}" if python_module else "true"

    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name=bcc_{tag}\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --nodes={nodes}\n"
        "#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        "#SBATCH --output=slurm-%j.out\n"
        "#SBATCH --error=slurm-%j.out\n"
        "\n"
        "set -euo pipefail\n"
        f"{module_line}\n"
        f"source {shlex.quote(venv)}/bin/activate\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "export PYTHONPATH=\"$PWD\"\n"
        "python -u remote_run.py\n"
    )


def _ssh_opts(ssh_key: str) -> List[str]:
    return [
        "-i",
        ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


def _ssh(host: str, user: str, ssh_key: str, command: str) -> str:
    proc = subprocess.run(
        ["ssh", *_ssh_opts(ssh_key), f"{user}@{host}", "bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _rsync_to(host: str, user: str, ssh_key: str, src: str, dst: str) -> None:
    rsh = "ssh " + " ".join(shlex.quote(o) for o in _ssh_opts(ssh_key))
    proc = subprocess.run(
        ["rsync", "-az", "-e", rsh, src, f"{user}@{host}:{dst}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rsync upload failed: {proc.stderr.strip()}")


def _rsync_from(host: str, user: str, ssh_key: str, src: str, dst: str) -> None:
    rsh = "ssh " + " ".join(shlex.quote(o) for o in _ssh_opts(ssh_key))
    proc = subprocess.run(
        ["rsync", "-az", "-e", rsh, f"{user}@{host}:{src}", dst],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rsync download failed: {proc.stderr.strip()}")


def _wait_for_terminal(host: str, user: str, ssh_key: str, jobid: str, poll_s: int) -> str:
    while True:
        out = _ssh(
            host,
            user,
            ssh_key,
            f"sacct -j {shlex.quote(jobid)} -X -n -P -o State",
        ).strip()
        first = out.splitlines()[0] if out else ""
        state = first.split()[0].upper() if first else ""
        if state.startswith("CANCELLED"):
            state = "CANCELLED"
        if state in _TERMINAL_STATES:
            return state
        time.sleep(poll_s)


_REMOTE_RUN_PY = '''\
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.qompiler import get_qompiler
from src.execution.pennylane_backend import run_pennylane
from src.execution.qiskit_backend import run_qiskit
from src.config.types import BackendConfig, CircuitConfig


def main() -> None:
    payload = pickle.loads((HERE / "payload.pkl").read_bytes())

    qomp = get_qompiler(payload["compiler"])
    spec = qomp.compile(
        payload["network"],
        payload["evidence"],
        payload["query"],
        encoding=payload["encoding"],
    )

    circuit_cfg = CircuitConfig(
        shots=int(payload["circuit_config"]["shots"]),
        seed=payload["circuit_config"]["seed"],
    )
    engine = payload["engine"]
    if engine == "pennylane":
        backend_cfg = BackendConfig(
            type="pennylane",
            device=payload["device"] or "lightning.qubit",
            wires=payload["wires"],
        )
        run_fn = run_pennylane
    elif engine == "qiskit_aer":
        backend_cfg = BackendConfig(
            type="qiskit",
            device=payload["device"] or "aer_simulator",
            wires=payload["wires"],
        )
        run_fn = run_qiskit
    else:
        raise ValueError(f"unknown navigator engine: {engine}")

    result = run_fn(
        output_dir=str(HERE),
        spec=spec,
        circuit_config=circuit_cfg,
        backend_config=backend_cfg,
        evidence=payload["evidence"],
        query=payload["query"],
        save_circuit_image=bool(payload["save_circuit_image"]),
    )

    raw = result.get("raw")
    if isinstance(raw, np.ndarray):
        np.save(HERE / "raw_samples.npy", raw)

    persisted = {
        "probs": result.get("probs"),
        "device": result.get("device"),
        "circuit_stats": result.get("circuit_stats"),
        "circuit_image": result.get("circuit_image"),
    }
    (HERE / "result.json").write_text(json.dumps(persisted, default=str))


if __name__ == "__main__":
    main()
'''
