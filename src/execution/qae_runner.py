"""QAE driver: faithful joint loading, indicator marking, Grover Q, MLAE.

Supports three execution paths, selected by `backend.type` + `backend.device`:

* `qiskit_aer` (no device, or device `aer_simulator`) — noiseless statevector
  with binomial shot simulation (fast, exact correctness check).
* `qiskit_aer` + device `noisy:<ibm-backend>` — `AerSimulator` with
  `NoiseModel.from_backend(<FakeXxx>)`, batched per-circuit sampling.
* `qiskit_ibm` + device `<ibm backend name>` — `QiskitRuntimeService` +
  `SamplerV2` (DD + measurement twirling), batched per-circuit sampling.

The MLE-fit step (`_mle_fit_theta`) is shared across paths; backends differ
only in how `(hits, totals)` are gathered. The joint loader is built ONCE per
`run_qae` call and reused across all amplitude estimates — at large grids this
is the dominant cost.

The prototype at scripts/qae_ship_prototype.py imports the primitives here so
the prototype and the pipeline stay in sync; the prototype uses the
statevector path exclusively.

Posterior recovery uses amplitude *ratios*: P(x|e) = a_{x,e}/a_e. The classical
posterior is never computed up front — this is the structural property that
distinguishes QAE from preconditioned QAA (see problems/qae_prototype.md).
"""

from __future__ import annotations

import logging
import math
import os
import time
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec

logger = logging.getLogger(__name__)

_THETA_GRID = np.linspace(0.0, math.pi / 2, 200_001)


def _bits_msb(value: int, n: int) -> List[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def _cpd_values(model: DiscreteBayesianNetwork, node: str) -> np.ndarray:
    return np.asarray(model.get_cpds(node).get_values(), dtype=float)


def joint_qubit_layout(
    model: DiscreteBayesianNetwork,
) -> Tuple[Dict[str, List[int]], int]:
    """Allocate ceil(log2(card)) qubits per node, in model traversal order.

    Returns (layout, num_state_qubits). The flag qubit (if any) sits at
    num_state_qubits and is allocated by the caller.
    """
    bits_per: Dict[str, int] = {}
    for n in model.nodes():
        card = int(model.get_cpds(n).variable_card)
        bits_per[n] = max(1, int(math.ceil(math.log2(card))))
    layout: Dict[str, List[int]] = {}
    w = 0
    for n in model.nodes():
        layout[n] = list(range(w, w + bits_per[n]))
        w += bits_per[n]
    return layout, w


def load_joint(
    qc: QuantumCircuit, model: DiscreteBayesianNetwork, layout: Dict[str, List[int]]
) -> None:
    """Prepare sum_x sqrt(P(x)) |x> on the qubits referenced by `layout`.

    Walks the network in `model.nodes()` order. For each node: unconditional
    StatePreparation of its prior column on the node's qubits if it has no
    parents; otherwise one controlled-StatePreparation per parent assignment,
    with the parent register X-conjugated to make the all-ones-controls form
    fire on the desired pattern. Target wires are appended reversed to
    reconcile our MSB-first convention with qiskit's LSB-first StatePreparation.
    """
    for node in model.nodes():
        cpd = model.get_cpds(node)
        parents = cpd.variables[1:]
        vals = _cpd_values(model, node)
        target = layout[node]
        if not parents:
            amps = np.sqrt(vals[:, 0])
            qc.append(StatePreparation(amps), [qc.qubits[i] for i in target[::-1]])
            continue
        parent_qubits = [layout[p] for p in parents]
        parent_cards = [int(model.get_cpds(p).variable_card) for p in parents]
        for col in range(vals.shape[1]):
            rem = col
            assignment: List[int] = []
            for card in reversed(parent_cards):
                assignment.insert(0, rem % card)
                rem //= card
            ctrl_qubits: List[int] = []
            ctrl_bits: List[Tuple[int, int]] = []
            for pq, pval in zip(parent_qubits, assignment):
                for q, b in zip(pq, _bits_msb(pval, len(pq))):
                    ctrl_qubits.append(q)
                    ctrl_bits.append((q, b))
            amps = np.sqrt(vals[:, col])
            if amps.sum() <= 0:
                continue
            zero_ctrls = [q for q, b in ctrl_bits if b == 0]
            for q in zero_ctrls:
                qc.x(q)
            sp = StatePreparation(amps).control(len(ctrl_qubits))
            qc.append(sp, [qc.qubits[i] for i in ctrl_qubits + target[::-1]])
            for q in zero_ctrls:
                qc.x(q)


def add_indicator(
    qc: QuantumCircuit,
    layout: Dict[str, List[int]],
    flag_qubit: int,
    conditions: Dict[str, int],
) -> None:
    """Flip `flag_qubit` to |1> iff every (node == value) in `conditions` holds."""
    ctrl_qubits: List[int] = []
    zero_ctrls: List[int] = []
    for node, value in conditions.items():
        qubits = layout[node]
        for q, b in zip(qubits, _bits_msb(int(value), len(qubits))):
            ctrl_qubits.append(q)
            if b == 0:
                zero_ctrls.append(q)
    for q in zero_ctrls:
        qc.x(q)
    qc.mcx(ctrl_qubits, flag_qubit)
    for q in zero_ctrls:
        qc.x(q)


def build_A(
    model: DiscreteBayesianNetwork,
    layout: Dict[str, List[int]],
    num_qubits: int,
    flag_qubit: int,
    conditions: Dict[str, int],
):
    """A = (load joint) ∘ (indicator on `conditions` → flag). Returns a Gate."""
    qc = QuantumCircuit(num_qubits)
    load_joint(qc, model, layout)
    add_indicator(qc, layout, flag_qubit, conditions)
    return qc.to_gate(label="A")


def build_grover_Q(A_gate, num_qubits: int, flag_qubit: int):
    """Q = A · (2|0><0| - I) · A^dagger · Z_flag. Sign convention drops a global
    -1 (irrelevant for measurement probabilities). Validated in the prototype:
    P(flag=1) after Q^m A|0> equals sin^2((2m+1)theta) to 1e-6.
    """
    qc = QuantumCircuit(num_qubits)
    qc.z(flag_qubit)
    qc.append(A_gate.inverse(), range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.append(A_gate, range(num_qubits))
    return qc.to_gate(label="Q")


def measure_good_prob_statevector(
    A_gate, Q_gate, num_qubits: int, flag_qubit: int, m: int
) -> float:
    qc = QuantumCircuit(num_qubits)
    qc.append(A_gate, range(num_qubits))
    for _ in range(m):
        qc.append(Q_gate, range(num_qubits))
    sv = Statevector.from_instruction(qc)
    return float(sv.probabilities([flag_qubit])[1])


def _mle_fit_theta(
    schedule: Sequence[int], hits: Sequence[int], totals: Sequence[int]
) -> float:
    """MLE for theta given (m_k, h_k, n_k) tuples; shared by every backend.

    L(theta) = prod_k sin^2((2m_k+1)theta)^{h_k} cos^2((2m_k+1)theta)^{n_k-h_k}.
    """
    angles = (2 * np.array(schedule) + 1)[:, None] * _THETA_GRID[None, :]
    s2 = np.sin(angles) ** 2
    eps = 1e-12
    ll = np.zeros_like(_THETA_GRID)
    for h, n, row in zip(hits, totals, s2):
        ll += h * np.log(row + eps) + (n - h) * np.log(1.0 - row + eps)
    return float(_THETA_GRID[int(np.argmax(ll))])


def mlae(
    A_gate,
    Q_gate,
    num_qubits: int,
    flag_qubit: int,
    schedule: Sequence[int],
    shots: int,
    rng: np.random.Generator,
) -> Tuple[float, int]:
    """Noiseless shots simulated as Binomial draws from the exact statevector
    probability — statistically identical to running shots on a noiseless
    sampler, without the Aer transpile/dispatch overhead.

    Returns (a_hat, total_grover_queries).
    """
    hits: List[int] = []
    totals: List[int] = []
    queries = 0
    for m in schedule:
        p = measure_good_prob_statevector(A_gate, Q_gate, num_qubits, flag_qubit, m)
        hits.append(int(rng.binomial(shots, min(1.0, max(0.0, p)))))
        totals.append(shots)
        queries += shots * (2 * m + 1)
    theta_hat = _mle_fit_theta(schedule, hits, totals)
    return float(math.sin(theta_hat) ** 2), queries


def build_loader_gate(
    model: DiscreteBayesianNetwork, layout: Dict[str, List[int]], num_state_qubits: int
):
    """Joint-loader as a reusable Gate. Built once per `run_qae` invocation and
    appended to every amplitude's A circuit, sharing the gate definition. At
    8x8 / 16x16 ship this is the biggest single speedup over the per-cell
    rebuild that `build_A` does."""
    qc = QuantumCircuit(num_state_qubits)
    load_joint(qc, model, layout)
    return qc.to_gate(label="load_joint")


def build_A_from_loader(
    loader_gate,
    num_qubits: int,
    num_state_qubits: int,
    layout: Dict[str, List[int]],
    flag_qubit: int,
    conditions: Dict[str, int],
):
    """A = (cached loader_gate) ∘ (indicator on `conditions` → flag)."""
    qc = QuantumCircuit(num_qubits)
    qc.append(loader_gate, range(num_state_qubits))
    add_indicator(qc, layout, flag_qubit, conditions)
    return qc.to_gate(label="A")


def _amp_circuit(A_gate, Q_gate, num_qubits: int, flag_qubit: int, m: int) -> QuantumCircuit:
    """Fully-measured circuit for A · Q^m, one classical bit on the flag.

    Uses a named classical register 'meas' so SamplerV2 PubResults can be
    addressed as `pub.data.meas` reliably regardless of how the transpiler
    reorders registers.
    """
    qc = QuantumCircuit(num_qubits)
    qc.append(A_gate, range(num_qubits))
    for _ in range(m):
        qc.append(Q_gate, range(num_qubits))
    creg = ClassicalRegister(1, name="meas")
    qc.add_register(creg)
    qc.measure(flag_qubit, creg[0])
    return qc


def _padded_index(cell_values: Sequence[int], bits_per_var: Sequence[int]) -> int:
    idx = 0
    for v, b in zip(cell_values, bits_per_var):
        idx = (idx << b) | int(v)
    return idx


def _gate_sizes(qc: QuantumCircuit) -> Dict[int, int]:
    sizes: Dict[int, int] = {}
    for instr in qc.data:
        arity = len(instr.qubits)
        if arity:
            sizes[arity] = sizes.get(arity, 0) + 1
    return sizes


def _resolve_aer_noisy_backend(device: str):
    """`device='noisy:<ibm-backend>'` → (AerSimulator with NoiseModel, FakeBackend for transpile)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider

    bk_name = device.split(":", 1)[1] if ":" in device else device
    fake_attr = "Fake" + bk_name.removeprefix("ibm_").capitalize()
    fake_cls = getattr(fake_provider, fake_attr, None)
    if fake_cls is None:
        raise ValueError(
            f"No fake_provider entry for {bk_name!r} (looked for "
            f"qiskit_ibm_runtime.fake_provider.{fake_attr!r}). Use a supported "
            "device (e.g. 'noisy:ibm_fez', 'noisy:ibm_brisbane') or run noiseless."
        )
    fake = fake_cls()
    nm = NoiseModel.from_backend(fake)
    sim = AerSimulator(noise_model=nm)
    return sim, fake


def _resolve_ibm_backend(device: str):
    """`device='<ibm backend>'` → (service, backend) via QiskitRuntimeService.

    Uses QISKIT_IBM_TOKEN / IBMQ_TOKEN / QISKIT_IBM_CHANNEL / QISKIT_IBM_INSTANCE
    from the environment (loaded from .env if present), matching the discovery
    used by src/execution/qiskit_backend.py.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBMQ_TOKEN") or os.getenv("IBMQ_API_TOKEN")
    channel = os.getenv("QISKIT_IBM_CHANNEL")
    instance = os.getenv("QISKIT_IBM_INSTANCE")
    try:
        if token:
            service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
        else:
            service = QiskitRuntimeService()
    except TypeError:
        service = QiskitRuntimeService()

    svc_get = getattr(service, "backend", None) or getattr(service, "get_backend", None)
    backend = svc_get(device)
    return service, backend


def _extract_good_counts_aer(result, num_circuits: int) -> List[int]:
    counts: List[int] = []
    for i in range(num_circuits):
        d = result.get_counts(i)
        counts.append(int(d.get("1", 0)))
    return counts


def _extract_good_counts_sampler(result) -> List[int]:
    counts: List[int] = []
    for pub in result:
        data = pub.data
        bit_array = getattr(data, "meas", None)
        if bit_array is None or not hasattr(bit_array, "get_bitstrings"):
            for fname in (getattr(data, "keys", lambda: [])()):
                candidate = getattr(data, fname, None)
                if candidate is not None and hasattr(candidate, "get_bitstrings"):
                    bit_array = candidate
                    break
        if bit_array is None:
            raise RuntimeError("SamplerV2 PubResult has no bitstring-bearing data field.")
        bitstrings = bit_array.get_bitstrings()
        counts.append(sum(1 for bs in bitstrings if bs.endswith("1")))
    return counts


def _print_hardware_summary(
    backend_label: str,
    num_amplitudes: int,
    num_circuits: int,
    shots: int,
    schedule: Sequence[int],
) -> None:
    print(f"\n--- QAE IBM Hardware Job Summary (pre-transpile) ---")
    print(f"  Backend           : {backend_label}")
    print(f"  Amplitudes to est : {num_amplitudes}  (P(e) + {num_amplitudes - 1} cells)")
    print(f"  Circuits per job  : {num_circuits}  (schedule={list(schedule)})")
    print(f"  Shots per circuit : {shots}")
    print(f"  Total shots       : {num_circuits * shots}")
    print(f"  Max Grover power  : {max(schedule) if schedule else 0}  "
          f"(deepest = A · Q^{max(schedule) if schedule else 0})")
    print(f"  Note: per qae_prototype.md, even A·Q^1 exceeds ibm_fez's T2 budget on")
    print(f"        a 9-qubit toy; hardware results will be noise-dominated.")
    print(f"---------------------------------------------------")


def _run_qae_sampling(
    network: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: List[str],
    query_cards: Sequence[int],
    layout: Dict[str, List[int]],
    num_state_qubits: int,
    flag_qubit: int,
    num_qubits: int,
    schedule: Sequence[int],
    shots: int,
    backend_kind: str,           # "aer_noisy" | "ibm_hw"
    runtime_target,              # AerSimulator (aer_noisy) or IBM backend (ibm_hw)
    transpile_backend,           # FakeXxx (aer_noisy) or IBM backend (ibm_hw)
    backend_label: str,
    skip_confirmation: bool,
) -> Tuple[float, Dict[Tuple[int, ...], float], int, QuantumCircuit, Any, Any, List[QuantumCircuit]]:
    """Build all measured circuits up front, transpile once, submit as ONE job,
    distribute results back to per-amplitude (hits, totals) for MLE fit.

    Returns (a_e_hat, a_joint, total_queries, A_e_circ, A_e_gate, Q_e_gate,
    transpiled_circuits).
    """
    loader_gate = build_loader_gate(network, layout, num_state_qubits)

    A_e = build_A_from_loader(
        loader_gate, num_qubits, num_state_qubits, layout, flag_qubit, dict(evidence)
    )
    Q_e = build_grover_Q(A_e, num_qubits, flag_qubit)

    circuits: List[QuantumCircuit] = []
    for m in schedule:
        circuits.append(_amp_circuit(A_e, Q_e, num_qubits, flag_qubit, m))

    cells = list(product(*[range(c) for c in query_cards]))
    for cell in cells:
        cond = {**evidence, **{node: int(v) for node, v in zip(query, cell)}}
        A_c = build_A_from_loader(
            loader_gate, num_qubits, num_state_qubits, layout, flag_qubit, cond
        )
        Q_c = build_grover_Q(A_c, num_qubits, flag_qubit)
        for m in schedule:
            circuits.append(_amp_circuit(A_c, Q_c, num_qubits, flag_qubit, m))

    num_circuits = len(circuits)
    num_amplitudes = 1 + len(cells)

    if backend_kind == "ibm_hw":
        _print_hardware_summary(backend_label, num_amplitudes, num_circuits, shots, schedule)
        if skip_confirmation:
            print("Submit to hardware? [y/N] y  (auto-confirmed via -y)")
        else:
            confirm = input("Submit to hardware? [y/N] ").strip().lower()
            if confirm != "y":
                raise RuntimeError("QAE hardware job cancelled by user.")

    logger.info("QAE[%s]: transpiling %d circuits...", backend_kind, num_circuits)
    t_tr = time.perf_counter()
    pm = generate_preset_pass_manager(backend=transpile_backend, optimization_level=1)
    transpiled = pm.run(circuits)
    if not isinstance(transpiled, list):
        transpiled = [transpiled]
    logger.info("QAE[%s]: transpiled in %.1fs", backend_kind, time.perf_counter() - t_tr)

    logger.info("QAE[%s]: submitting batched job (%d circuits × %d shots)...",
                backend_kind, num_circuits, shots)
    t_run = time.perf_counter()
    if backend_kind == "aer_noisy":
        job = runtime_target.run(transpiled, shots=shots)
        result = job.result()
        good_counts = _extract_good_counts_aer(result, num_circuits)
    elif backend_kind == "ibm_hw":
        from qiskit_ibm_runtime import SamplerV2
        try:
            from qiskit_ibm_runtime.options import SamplerOptions
            opts = SamplerOptions()
            opts.dynamical_decoupling.enable = True
            opts.twirling.enable_gates = True
            opts.twirling.enable_measure = True
            sampler = SamplerV2(mode=transpile_backend, options=opts)
        except Exception:
            sampler = SamplerV2(mode=transpile_backend)
        job = sampler.run(transpiled, shots=shots)
        result = job.result()
        good_counts = _extract_good_counts_sampler(result)
    else:
        raise ValueError(f"unknown backend_kind: {backend_kind!r}")
    logger.info("QAE[%s]: job completed in %.1fs", backend_kind, time.perf_counter() - t_run)

    s_len = len(schedule)
    queries_per_amp = sum(shots * (2 * m + 1) for m in schedule)

    pe_hits = good_counts[:s_len]
    pe_totals = [shots] * s_len
    a_e_hat = float(math.sin(_mle_fit_theta(schedule, pe_hits, pe_totals)) ** 2)

    a_joint: Dict[Tuple[int, ...], float] = {}
    offset = s_len
    for cell in cells:
        cell_hits = good_counts[offset:offset + s_len]
        cell_totals = [shots] * s_len
        a_joint[cell] = float(math.sin(_mle_fit_theta(schedule, cell_hits, cell_totals)) ** 2)
        offset += s_len

    total_queries = num_amplitudes * queries_per_amp

    A_e_circ = QuantumCircuit(num_qubits)
    A_e_circ.append(A_e, range(num_qubits))

    return a_e_hat, a_joint, total_queries, A_e_circ, A_e, Q_e, transpiled


def _run_qae_statevector(
    network: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: List[str],
    query_cards: Sequence[int],
    layout: Dict[str, List[int]],
    num_state_qubits: int,
    flag_qubit: int,
    num_qubits: int,
    schedule: Sequence[int],
    shots: int,
    rng: np.random.Generator,
) -> Tuple[float, Dict[Tuple[int, ...], float], int, QuantumCircuit, Any, Any]:
    """Noiseless statevector + binomial shots (the fast, exact-correctness path)."""
    loader_gate = build_loader_gate(network, layout, num_state_qubits)

    A_e = build_A_from_loader(
        loader_gate, num_qubits, num_state_qubits, layout, flag_qubit, dict(evidence)
    )
    Q_e = build_grover_Q(A_e, num_qubits, flag_qubit)
    t_e0 = time.perf_counter()
    a_e, q_e = mlae(A_e, Q_e, num_qubits, flag_qubit, schedule, shots, rng)
    logger.info("QAE P(e) estimate=%.6e queries=%d in %.1fs",
                a_e, q_e, time.perf_counter() - t_e0)

    a_joint: Dict[Tuple[int, ...], float] = {}
    total_queries = q_e
    total_cells = 1
    for c in query_cards:
        total_cells *= c
    t_cells0 = time.perf_counter()
    for cell_idx, cell in enumerate(product(*[range(c) for c in query_cards])):
        cond = {**evidence, **{node: int(v) for node, v in zip(query, cell)}}
        A_c = build_A_from_loader(
            loader_gate, num_qubits, num_state_qubits, layout, flag_qubit, cond
        )
        Q_c = build_grover_Q(A_c, num_qubits, flag_qubit)
        a_hat, q = mlae(A_c, Q_c, num_qubits, flag_qubit, schedule, shots, rng)
        a_joint[cell] = a_hat
        total_queries += q
        logger.info("  cell %s: a_hat=%.4e (%d/%d, %.1fs)", cell, a_hat,
                    cell_idx + 1, total_cells, time.perf_counter() - t_cells0)

    A_e_circ = QuantumCircuit(num_qubits)
    A_e_circ.append(A_e, range(num_qubits))
    return a_e, a_joint, total_queries, A_e_circ, A_e, Q_e


def run_qae(
    output_dir: str,
    spec: CircuitSpec,
    circuit_config: CircuitConfig,
    backend_config: BackendConfig,
    network: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: List[str],
    save_circuit_image: bool = False,
    skip_confirmation: bool = False,
) -> Dict[str, Any]:
    """Drive the full QAE inference and return a run_result dict compatible
    with the existing pipeline (`probs`, `raw`, `counts`, `circuit_stats`,
    `transpiled_stats`, `backend`, `circuit_image`).

    Dispatch on (backend.type, backend.device):
      * `qiskit_aer` (no device or `aer_simulator`) → statevector + binomial.
      * `qiskit_aer` + `device='noisy:<name>'` → AerSimulator with the
        corresponding FakeXxx NoiseModel, batched sampling.
      * `qiskit_ibm` + `device='<ibm backend>'` → QiskitRuntimeService +
        SamplerV2 (DD + measurement twirling), batched sampling.
    """
    params = dict(backend_config.params or {})
    driver = params.get("driver", "mlae")
    if driver != "mlae":
        raise ValueError(
            f"QAE v1 only implements driver='mlae'; got {driver!r}. IQAE is a follow-up."
        )
    schedule: List[int] = list(params.get("mlae_schedule", [0, 1, 2, 4]))
    shots: int = int(params.get("shots_per_round", circuit_config.shots))
    seed: int = int(circuit_config.seed if circuit_config.seed is not None else 0)
    rng = np.random.default_rng(seed)

    layout, num_state_qubits = joint_qubit_layout(network)
    flag_qubit = num_state_qubits
    num_qubits = num_state_qubits + 1

    query_cards = [int(network.get_cpds(node).variable_card) for node in query]
    bits_per_query_var = [max(1, int(math.ceil(math.log2(c)))) for c in query_cards]
    padded_size = 1
    for b in bits_per_query_var:
        padded_size *= 2 ** b

    backend_type = (backend_config.type or "").strip()
    device = (backend_config.device or "").strip()

    t_total = time.perf_counter()
    transpiled_circuits: Optional[List[QuantumCircuit]] = None

    if backend_type == "qiskit_ibm":
        if not device:
            raise ValueError("backend.type='qiskit_ibm' requires backend.device (the IBM backend name).")
        service, ibm_backend = _resolve_ibm_backend(device)
        backend_label = f"qiskit_ibm:{device}"
        a_e, a_joint, total_queries, A_circ, A_e_gate, Q_e_gate, transpiled_circuits = \
            _run_qae_sampling(
                network, evidence, query, query_cards, layout, num_state_qubits,
                flag_qubit, num_qubits, schedule, shots,
                backend_kind="ibm_hw", runtime_target=None,
                transpile_backend=ibm_backend, backend_label=backend_label,
                skip_confirmation=skip_confirmation,
            )
    elif backend_type.startswith("qiskit_aer") and device.startswith("noisy:"):
        sim, fake = _resolve_aer_noisy_backend(device)
        backend_label = f"qiskit_aer:{device}"
        a_e, a_joint, total_queries, A_circ, A_e_gate, Q_e_gate, transpiled_circuits = \
            _run_qae_sampling(
                network, evidence, query, query_cards, layout, num_state_qubits,
                flag_qubit, num_qubits, schedule, shots,
                backend_kind="aer_noisy", runtime_target=sim,
                transpile_backend=fake, backend_label=backend_label,
                skip_confirmation=skip_confirmation,
            )
    elif backend_type.startswith("qiskit_aer"):
        backend_label = "qiskit_aer (statevector)"
        a_e, a_joint, total_queries, A_circ, A_e_gate, Q_e_gate = \
            _run_qae_statevector(
                network, evidence, query, query_cards, layout, num_state_qubits,
                flag_qubit, num_qubits, schedule, shots, rng,
            )
    else:
        raise ValueError(
            f"QAE supports backend.type in {{qiskit_aer, qiskit_ibm}}; got "
            f"{backend_type!r}. For noisy simulation use "
            "device='noisy:<ibm-backend>' (e.g. 'noisy:ibm_fez')."
        )

    total_a = sum(a_joint.values())
    probs = [0.0] * padded_size
    if total_a > 0:
        for cell, a in a_joint.items():
            probs[_padded_index(cell, bits_per_query_var)] = float(a / total_a)
    else:
        for cell in a_joint:
            probs[_padded_index(cell, bits_per_query_var)] = 1.0 / len(a_joint)

    circuit_stats = {
        "num_wires": num_qubits,
        "num_clbits": 1 if transpiled_circuits is not None else 0,
        "num_operations": sum(A_circ.count_ops().values()),
        "operation_counts": dict(A_circ.count_ops()),
        "depth": int(A_circ.depth()),
        "gate_sizes": _gate_sizes(A_circ),
        "num_parameters": int(A_circ.num_parameters),
        "num_measurements": 0,
        "qae_total_grover_queries": total_queries,
        "qae_amplitudes_estimated": 1 + len(a_joint),
        "qae_schedule": list(schedule),
        "qae_shots_per_round": shots,
        "qae_a_e": float(a_e),
    }

    transpiled_stats: Optional[Dict[str, Any]] = None
    try:
        if transpiled_circuits is not None:
            tA = transpiled_circuits[0]
            tQ = transpiled_circuits[len(schedule) - 1]
            transpiled_stats = {
                "num_wires": tA.num_qubits,
                "A_depth": int(tA.depth()),
                "A_operation_counts": dict(tA.count_ops()),
                "A_Qmax_depth": int(tQ.depth()),
                "A_Qmax_operation_counts": dict(tQ.count_ops()),
                "max_grover_power": max(schedule) if schedule else 0,
                "num_transpiled_circuits": len(transpiled_circuits),
            }
        else:
            basis = ["rz", "sx", "x", "cx"]
            tA = transpile(A_circ, basis_gates=basis, optimization_level=1)
            Q_max_circ = QuantumCircuit(num_qubits)
            Q_max_circ.append(A_e_gate, range(num_qubits))
            for _ in range(max(schedule) if schedule else 0):
                Q_max_circ.append(Q_e_gate, range(num_qubits))
            tQ = transpile(Q_max_circ, basis_gates=basis, optimization_level=1)
            transpiled_stats = {
                "num_wires": tA.num_qubits,
                "A_depth": int(tA.depth()),
                "A_operation_counts": dict(tA.count_ops()),
                "A_Qmax_depth": int(tQ.depth()),
                "A_Qmax_operation_counts": dict(tQ.count_ops()),
                "basis_gates": basis,
                "max_grover_power": max(schedule) if schedule else 0,
            }
    except Exception as exc:
        logger.warning("QAE transpile depth report failed: %s", exc)

    circuit_image = None
    if save_circuit_image:
        try:
            from src.circuits.visualize import save_qiskit_circuit_image
            circuit_image = save_qiskit_circuit_image(output_dir, A_circ)
        except Exception as exc:
            logger.warning("QAE circuit image save failed: %s", exc)

    counts = {format(i, f"0{int(math.log2(padded_size))}b"): int(round(p * shots))
              for i, p in enumerate(probs) if p > 0 and padded_size > 1}

    logger.info("QAE total wall time %.1fs, total Grover queries=%d, backend=%s",
                time.perf_counter() - t_total, total_queries, backend_label)

    return {
        "counts": counts,
        "probs": probs,
        "raw": None,
        "backend": backend_label,
        "circuit_stats": circuit_stats,
        "transpiled_stats": transpiled_stats,
        "circuit_image": circuit_image,
    }
