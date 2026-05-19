from __future__ import annotations

import json
import os
import pickle
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

# Per SLURM docs: states a job will not leave on its own.
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
    "REVOKED",
}

_DEFAULT_HOST = "navigator.lca.uc.pt"
_SSH_ATTEMPTS = 5
_SSH_BACKOFF_BASE = 2.0


class NavigatorJobDetached(Exception):
    """Raised when a job has been submitted in detach mode so the runner can
    short-circuit the metrics path and write a pending sidecar instead."""

    def __init__(self, tag: str, jobid: str, remote_dir: str, checkpoint_path: str):
        super().__init__(f"Navigator job {jobid} submitted with tag '{tag}'")
        self.tag = tag
        self.jobid = jobid
        self.remote_dir = remote_dir
        self.checkpoint_path = checkpoint_path


@dataclass
class _Conn:
    host: str
    user: str
    ssh_key: str
    # ControlMaster path makes repeated polls reuse one TCP/auth session.
    control_path: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


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
    user = env["NAVIGATOR_USER"]
    ssh_key = os.path.abspath(os.path.expanduser(env["NAVIGATOR_SSH_KEY"]))
    if not os.path.isfile(ssh_key):
        raise FileNotFoundError(
            f"NAVIGATOR_SSH_KEY points to a missing file: {ssh_key}"
        )
    account = env["NAVIGATOR_ACCOUNT"]
    remote_base = env["NAVIGATOR_REMOTE_DIR"].rstrip("/")
    venv = env["NAVIGATOR_VENV"]
    host = os.environ.get("NAVIGATOR_HOST") or _DEFAULT_HOST

    params = backend_config.params or {}
    engine = (params.get("engine") or "pennylane").lower()
    if engine not in {"pennylane", "qiskit_aer"}:
        raise ValueError(
            f"navigator backend engine must be 'pennylane' or 'qiskit_aer', got {engine!r}"
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir_path / "navigator_checkpoint.json"
    sentinel_path = output_dir_path / "navigator_submitting.json"

    if checkpoint_path.exists():
        checkpoint = _read_state_file(checkpoint_path)
        conn = _make_conn(
            checkpoint.get("host", host), user, ssh_key, output_dir_path
        )
        try:
            return _wait_and_collect(output_dir_path, conn, checkpoint, params)
        finally:
            _stop_control_master(conn)

    if sentinel_path.exists():
        sentinel = _read_state_file(sentinel_path)
        conn = _make_conn(sentinel.get("host", host), user, ssh_key, output_dir_path)
        try:
            jobid = _find_job_by_name(conn, f"bcc_{sentinel['tag']}")
            if not jobid:
                raise RuntimeError(
                    f"navigator_submitting.json exists for tag {sentinel['tag']!r} but no "
                    f"matching SLURM job was found via squeue/sacct. Delete the sentinel "
                    f"and rerun if the submission did not succeed."
                )
            checkpoint = {
                "tag": sentinel["tag"],
                "jobid": jobid,
                "remote_dir": sentinel["remote_dir"],
                "host": sentinel.get("host", host),
                "submitted_at": sentinel.get("submitted_at"),
                "recovered_at": _now(),
                "engine": sentinel.get("engine", engine),
                "device": sentinel.get("device"),
            }
            _atomic_write_json(checkpoint_path, checkpoint)
            sentinel_path.unlink(missing_ok=True)
            print(
                f"[navigator] recovered orphaned submission tag={sentinel['tag']!r} jobid={jobid}",
                flush=True,
            )
            return _wait_and_collect(output_dir_path, conn, checkpoint, params)
        finally:
            _stop_control_master(conn)

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
    if not _is_safe_tag(tag):
        raise ValueError(
            f"navigator tag {tag!r} must match [A-Za-z0-9._-]+ (used in SLURM job name and remote path)"
        )
    remote_dir = f"{remote_base}/{tag}"
    staging = output_dir_path / "_navigator" / tag
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
    try:
        payload_bytes = pickle.dumps(payload, protocol=4)
    except Exception as exc:
        raise RuntimeError(
            f"failed to pickle navigator payload (likely the Bayesian network is not "
            f"picklable): {exc}"
        ) from exc
    (staging / "payload.pkl").write_bytes(payload_bytes)
    (staging / "remote_run.py").write_text(_REMOTE_RUN_PY)
    (staging / "submit.sbatch").write_text(_build_sbatch(tag, account, venv, params))

    project_src = Path(__file__).resolve().parents[2] / "src"
    conn = _make_conn(host, user, ssh_key, output_dir_path)
    try:
        _ssh(conn, f"mkdir -p {shlex.quote(remote_dir)}")
        _rsync(conn, f"{staging}/", f"{conn.target}:{remote_dir}/", direction="upload")
        _rsync(
            conn,
            f"{project_src}/",
            f"{conn.target}:{remote_dir}/src/",
            direction="upload",
        )

        sentinel = {
            "tag": tag,
            "remote_dir": remote_dir,
            "host": host,
            "submitted_at": _now(),
            "engine": engine,
            "device": backend_config.device,
        }
        _atomic_write_json(sentinel_path, sentinel)

        submit_out = _ssh(
            conn,
            f"cd {shlex.quote(remote_dir)} && sbatch --parsable submit.sbatch",
        ).strip()
        jobid = submit_out.split(";")[0].strip()
        if not jobid.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {submit_out!r}")

        checkpoint = {
            **sentinel,
            "jobid": jobid,
            "submitted_at_full": _now(),
        }
        _atomic_write_json(checkpoint_path, checkpoint)
        sentinel_path.unlink(missing_ok=True)

        print(
            f"[navigator] submitted jobid={jobid} tag={tag!r} remote_dir={remote_dir}",
            flush=True,
        )

        if params.get("detach"):
            raise NavigatorJobDetached(tag, jobid, remote_dir, str(checkpoint_path))

        return _wait_and_collect(output_dir_path, conn, checkpoint, params)
    finally:
        _stop_control_master(conn)


_SAFE_TAG_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _is_safe_tag(tag: str) -> bool:
    return bool(tag) and all(c in _SAFE_TAG_CHARS for c in tag)


def _sanitize_for_tag(name: str) -> str:
    return "".join(c if c in _SAFE_TAG_CHARS else "_" for c in name) or "job"


def _autotag(base_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_for_tag(base_name)}_{stamp}_{uuid.uuid4().hex[:6]}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_conn(host: str, user: str, ssh_key: str, output_dir_path: Path) -> _Conn:
    control_dir = output_dir_path / "_navigator"
    control_dir.mkdir(parents=True, exist_ok=True)
    cp = control_dir / f"ssh-{uuid.uuid4().hex[:8]}.sock"
    return _Conn(host=host, user=user, ssh_key=ssh_key, control_path=str(cp))


def _ssh_opts(conn: _Conn) -> List[str]:
    return [
        "-i",
        conn.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={conn.control_path}",
        "-o",
        "ControlPersist=600",
    ]


def _ssh(conn: _Conn, command: str) -> str:
    def _do() -> str:
        proc = subprocess.run(
            ["ssh", *_ssh_opts(conn), conn.target, "bash", "-lc", command],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise _TransientError(
                f"ssh exit {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    return _retry(_do, label="ssh")


_RSYNC_EXCLUDES = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    ".mypy_cache/",
    ".DS_Store",
)


def _rsync(conn: _Conn, src: str, dst: str, *, direction: str) -> None:
    rsh = "ssh " + " ".join(shlex.quote(o) for o in _ssh_opts(conn))
    cmd = ["rsync", "-az", "--partial"]
    for pat in _RSYNC_EXCLUDES:
        cmd += ["--exclude", pat]
    cmd += ["-e", rsh, src, dst]

    def _do() -> str:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise _TransientError(
                f"rsync {direction} exit {proc.returncode}: {proc.stderr.strip()}"
            )
        return proc.stdout

    _retry(_do, label=f"rsync-{direction}")


class _TransientError(RuntimeError):
    pass


def _retry(fn: Callable[[], str], *, label: str) -> str:
    last_err: Optional[BaseException] = None
    for attempt in range(1, _SSH_ATTEMPTS + 1):
        try:
            return fn()
        except _TransientError as exc:
            last_err = exc
            if attempt == _SSH_ATTEMPTS:
                break
            delay = _SSH_BACKOFF_BASE * (2 ** (attempt - 1))
            print(
                f"[navigator] {label} failed (attempt {attempt}/{_SSH_ATTEMPTS}): {exc} — retry in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {_SSH_ATTEMPTS} attempts: {last_err}")


def _stop_control_master(conn: _Conn) -> None:
    if not os.path.exists(conn.control_path):
        return
    try:
        subprocess.run(
            ["ssh", "-O", "exit", "-o", f"ControlPath={conn.control_path}", conn.target],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        pass


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _read_state_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"navigator state file {path} is corrupted ({exc}). Delete it to force a "
            f"fresh submit, or restore it from a backup."
        ) from exc


def _parse_sacct_state(raw: str) -> str:
    if not raw:
        return ""
    state = raw.splitlines()[0].split()[0].upper()
    # SLURM appends "+" if state was truncated; strip it.
    state = state.rstrip("+")
    # Forms like "CANCELLED BY 1234" → first token already handled by split() above.
    return state


def _find_job_by_name(conn: _Conn, jobname: str) -> Optional[str]:
    # squeue covers PENDING/RUNNING and reflects controller truth.
    out = _ssh(conn, f"squeue --name={shlex.quote(jobname)} -h -o '%i'").strip()
    for line in out.splitlines():
        token = line.strip().split("_")[0]
        if token.isdigit():
            return token
    # sacct covers finished jobs (lag tolerated).
    out = _ssh(
        conn,
        f"sacct --name={shlex.quote(jobname)} -X -n -P -o JobID,Submit "
        f"| sort -t'|' -k2 | tail -1",
    ).strip()
    if out:
        jid = out.split("|")[0].strip().split("_")[0]
        if jid.isdigit():
            return jid
    return None


def _wait_and_collect(
    output_dir_path: Path,
    conn: _Conn,
    checkpoint: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    poll_s = int(params.get("poll_interval_s", 30))
    max_wait_s = params.get("max_wait_s")
    jobid = str(checkpoint["jobid"])
    remote_dir = checkpoint["remote_dir"]
    tag = checkpoint["tag"]

    state = _wait_for_terminal(conn, jobid, poll_s, max_wait_s)

    local_artifacts = output_dir_path / "_navigator" / tag
    local_artifacts.mkdir(parents=True, exist_ok=True)
    _rsync(
        conn,
        f"{conn.target}:{remote_dir}/",
        f"{local_artifacts}/",
        direction="download",
    )

    if state != "COMPLETED":
        tail = ""
        for log_path in sorted(local_artifacts.glob("slurm-*.out")):
            lines = log_path.read_text(errors="replace").splitlines()[-80:]
            tail += "\n--- " + log_path.name + " (tail) ---\n" + "\n".join(lines)
        raise RuntimeError(
            f"Navigator job {jobid} ended in state {state}{tail}"
        )

    result_path = local_artifacts / "result.json"
    if not result_path.exists():
        raise RuntimeError(
            f"Navigator job {jobid} completed but result.json was not produced"
        )

    result = json.loads(result_path.read_text())

    raw_path = local_artifacts / "raw_samples.npy"
    raw_samples = np.load(raw_path) if raw_path.exists() else None

    circuit_image = result.get("circuit_image")
    if circuit_image:
        local_img = local_artifacts / Path(circuit_image).name
        circuit_image = str(local_img) if local_img.exists() else None

    return {
        "probs": result.get("probs"),
        "raw": raw_samples,
        "device": result.get("device"),
        "circuit_stats": result.get("circuit_stats"),
        "circuit_image": circuit_image,
    }


def _wait_for_terminal(
    conn: _Conn,
    jobid: str,
    poll_s: int,
    max_wait_s: Optional[int],
) -> str:
    started = time.monotonic()
    last_state = ""
    while True:
        out = _ssh(
            conn, f"sacct -j {shlex.quote(jobid)} -X -n -P -o State"
        ).strip()
        state = _parse_sacct_state(out)
        if state in _TERMINAL_STATES:
            print(
                f"[navigator] job {jobid} reached terminal state {state}",
                flush=True,
            )
            return state
        if state and state != last_state:
            print(
                f"[navigator] job {jobid} state={state}",
                flush=True,
            )
            last_state = state
        elapsed = time.monotonic() - started
        if max_wait_s is not None and elapsed > float(max_wait_s):
            raise RuntimeError(
                f"Navigator job {jobid} exceeded max_wait_s={max_wait_s} "
                f"(current state={state or 'unknown'}). Resume later with --resume."
            )
        time.sleep(poll_s)


def _build_sbatch(tag: str, account: str, venv: str, params: Dict[str, Any]) -> str:
    partition = params["partition"]
    time_limit = params.get("time_limit", "00:30:00")
    cpus = int(params.get("cpus", 8))
    mem = params.get("mem", "8G")
    nodes = int(params.get("nodes", 1))
    gres = params.get("gres")
    python_module = os.environ.get("NAVIGATOR_PYTHON_MODULE")
    module_line = (
        f"module load {python_module}" if python_module else "true  # no module load"
    )
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""

    # bash -l guarantees /etc/profile.d/modules.sh is sourced so `module` exists.
    return (
        "#!/bin/bash -l\n"
        f"#SBATCH --job-name=bcc_{tag}\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --nodes={nodes}\n"
        "#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        f"{gres_line}"
        "#SBATCH --output=slurm-%j.out\n"
        "#SBATCH --error=slurm-%j.out\n"
        "\n"
        "set -eo pipefail\n"
        'cd "${SLURM_SUBMIT_DIR:-$PWD}"\n'
        f"{module_line}\n"
        f"source {shlex.quote(venv)}/bin/activate\n"
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n"
        "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n"
        "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n"
        'export PYTHONPATH="$PWD"\n'
        'echo "[remote] host=$(hostname) jobid=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} cwd=$PWD"\n'
        "python -V\n"
        "python -u remote_run.py\n"
    )


_REMOTE_RUN_PY = '''\
from __future__ import annotations

import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.qompiler import get_qompiler
from src.config.types import BackendConfig, CircuitConfig


def _engine_runner(engine: str):
    if engine == "pennylane":
        from src.execution.pennylane_backend import run_pennylane

        return run_pennylane
    if engine == "qiskit_aer":
        from src.execution.qiskit_backend import run_qiskit

        return run_qiskit
    raise ValueError(f"unknown navigator engine: {engine}")


def main() -> int:
    try:
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
        else:
            backend_cfg = BackendConfig(
                type="qiskit",
                device=payload["device"] or "aer_simulator",
                wires=payload["wires"],
            )

        run_fn = _engine_runner(engine)
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

        circuit_image = result.get("circuit_image")
        if circuit_image:
            circuit_image = Path(circuit_image).name

        persisted = {
            "probs": result.get("probs"),
            "device": result.get("device"),
            "circuit_stats": result.get("circuit_stats"),
            "circuit_image": circuit_image,
        }
        (HERE / "result.json").write_text(json.dumps(persisted, default=str))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''
