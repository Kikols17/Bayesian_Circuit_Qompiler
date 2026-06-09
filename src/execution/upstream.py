"""Modular upstream-inference engine: swappable Loader x Marker x Driver.

This implements the *upstream* encoding of problems/advantage_bearing_encoding.md:
load the factored joint prior, then let amplitude estimation/amplification do the
#P-hard marginalisation. The point of this module is **composability** — the three
stages are independent registries so any combination can be tried from config:

    QuerySpec
        |
        v
    [Loader]  prepares  sum_x sqrt(P(x)) |x>     LOADERS  = {"faithful": ...}
        |
    [Marker]  flips flag to |1> iff "good"       MARKERS  = {"equality","cells","threshold"}
        |
       A = Loader then Marker ;  Q = Grover(A)
        |
    [Driver]  estimates a = P(good)              DRIVERS  = {"mlae","iqae"}

Selection happens via backend.params:

    backend:
      type: qiskit_aer
      params:
        loader: faithful
        marker: threshold          # equality | cells | threshold
        driver: iqae               # mlae | iqae
        query_mode: single         # single | enumerate
        marker_params: {var: Y, op: lt, value: 16}
        driver_params: {epsilon: 0.005, alpha: 0.05, shots: 100}

To add a new component, register a function (Loader/Driver) or a `Marker` in the
relevant dict — nothing else in the pipeline needs to change. The default
combination (faithful/equality/mlae/enumerate) reproduces the legacy QAE path.

Only the noiseless statevector backend is wired here; it is the fast, exact
engine the investigation needs to compare combinations. The legacy noisy/hardware
path in qae_runner.py is preserved for the default combination.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from qiskit import QuantumCircuit

from src.execution.qae_runner import (
    _bits_msb,
    _mle_fit_theta,
    add_indicator,
    build_grover_Q,
    joint_qubit_layout,
    load_joint,
    measure_good_prob_statevector,
)
from src.qompiler.qaa import _build_conditional_probs

logger = logging.getLogger(__name__)


# ============================================================== Loaders
# A loader appends gates preparing sum_x sqrt(P(x)) |x> on the state qubits.

def _topological_nodes(model: DiscreteBayesianNetwork) -> List[str]:
    import networkx as nx

    return list(nx.topological_sort(model))


def _load_faithful(
    qc: QuantumCircuit, model: DiscreteBayesianNetwork, layout: Dict[str, List[int]]
) -> None:
    load_joint(qc, model, layout, node_order=_topological_nodes(model))


LOADERS: Dict[
    str, Callable[[QuantumCircuit, DiscreteBayesianNetwork, Dict[str, List[int]]], None]
] = {
    "faithful": _load_faithful,
}


# ============================================================== Markers
# A marker appends gates flipping `flag` to |1> iff the "good" subspace holds:
# (evidence matches) AND (query target holds). Markers that use ancilla qubits
# MUST uncompute them back to |0> so Grover's reflection-about-|0> stays valid.

def _cells_from_predicate(
    query: Sequence[str],
    query_cards: Sequence[int],
    predicate: Callable[[Dict[str, int]], bool],
) -> List[Dict[str, int]]:
    """Classically enumerate the query assignments satisfying `predicate`.

    Distinct assignments are mutually exclusive on computational basis states, so
    OR-ing their equality indicators into a fresh flag is a correct OR.
    """
    cells: List[Dict[str, int]] = []
    for combo in product(*[range(c) for c in query_cards]):
        assignment = {v: int(x) for v, x in zip(query, combo)}
        if predicate(assignment):
            cells.append(assignment)
    return cells


def _threshold_predicate(var: str, op: str, value: int) -> Callable[[Dict[str, int]], bool]:
    value = int(value)
    if op == "lt":
        return lambda a: a[var] < value
    if op == "le":
        return lambda a: a[var] <= value
    if op == "gt":
        return lambda a: a[var] > value
    if op == "ge":
        return lambda a: a[var] >= value
    raise ValueError(f"threshold marker: op must be one of lt|le|gt|ge, got {op!r}")


def _comparator_geq_value(op: str, value: int) -> Tuple[bool, int]:
    """Map an op/value to qiskit IntegerComparator's (geq, value) convention."""
    value = int(value)
    if op == "ge":
        return True, value
    if op == "gt":
        return True, value + 1
    if op == "lt":
        return False, value
    if op == "le":
        return False, value + 1
    raise ValueError(f"threshold marker: op must be one of lt|le|gt|ge, got {op!r}")


@dataclass
class Marker:
    apply: Callable[..., None]
    ancilla_count: Callable[[Dict[str, Any], Sequence[str], Sequence[int], Dict[str, List[int]]], int]
    good_cells: Callable[
        [Dict[str, Any], Sequence[str], Sequence[int]], List[Dict[str, int]]
    ]


def _mark_equality(
    qc, layout, flag, ancillas, *, evidence, query, query_cards, params
) -> None:
    cell = params["cell"]
    add_indicator(qc, layout, flag, {**evidence, **{k: int(v) for k, v in cell.items()}})


def _mark_cells(
    qc, layout, flag, ancillas, *, evidence, query, query_cards, params
) -> None:
    for cell in params["cells"]:
        add_indicator(qc, layout, flag, {**evidence, **{k: int(v) for k, v in cell.items()}})


def _mark_threshold(
    qc, layout, flag, ancillas, *, evidence, query, query_cards, params
) -> None:
    from qiskit.circuit.library import IntegerComparator

    var = params["var"]
    var_qubits = layout[var]
    n = len(var_qubits)
    geq, value = _comparator_geq_value(params["op"], params["value"])

    cmp = IntegerComparator(num_state_qubits=n, value=value, geq=geq).to_gate()
    result_anc = ancillas[0]
    scratch = ancillas[1:n]
    # IntegerComparator reads its data qubits LSB-first; our layout is MSB-first.
    data_lsb_first = list(var_qubits)[::-1]
    cmp_wires = [qc.qubits[i] for i in (data_lsb_first + [result_anc] + list(scratch))]

    qc.append(cmp, cmp_wires)

    ctrl_qubits: List[int] = []
    zero_ctrls: List[int] = []
    for node, value_e in evidence.items():
        qubits = layout[node]
        for q, b in zip(qubits, _bits_msb(int(value_e), len(qubits))):
            ctrl_qubits.append(q)
            if b == 0:
                zero_ctrls.append(q)
    ctrl_qubits.append(result_anc)
    for q in zero_ctrls:
        qc.x(q)
    qc.mcx(ctrl_qubits, flag)
    for q in zero_ctrls:
        qc.x(q)

    qc.append(cmp.inverse(), cmp_wires)


MARKERS: Dict[str, Marker] = {
    "equality": Marker(
        apply=_mark_equality,
        ancilla_count=lambda params, q, qc, layout: 0,
        good_cells=lambda params, q, qc: [
            {k: int(v) for k, v in params["cell"].items()}
        ],
    ),
    "cells": Marker(
        apply=_mark_cells,
        ancilla_count=lambda params, q, qc, layout: 0,
        good_cells=lambda params, q, qc: [
            {k: int(v) for k, v in cell.items()} for cell in params["cells"]
        ],
    ),
    "threshold": Marker(
        apply=_mark_threshold,
        ancilla_count=lambda params, q, qc, layout: len(layout[params["var"]]),
        good_cells=lambda params, q, qc: _cells_from_predicate(
            q, qc, _threshold_predicate(params["var"], params["op"], params["value"])
        ),
    ),
}


# ============================================================== A operator
def build_A_operator(
    model: DiscreteBayesianNetwork,
    layout: Dict[str, List[int]],
    num_state_qubits: int,
    loader_name: str,
    marker_name: str,
    marker_params: Dict[str, Any],
    evidence: Dict[str, int],
    query: Sequence[str],
    query_cards: Sequence[int],
) -> Tuple[Any, int, int]:
    """Compose loader + marker into the A = (load joint) then (mark good) gate.

    Returns (A_gate, num_qubits, flag_qubit). flag sits right after the state
    qubits; ancillas (if the marker needs them) follow and are uncomputed by the
    marker so they are |0> at the gate boundary.
    """
    if loader_name not in LOADERS:
        raise ValueError(f"unknown loader {loader_name!r}; available: {list(LOADERS)}")
    if marker_name not in MARKERS:
        raise ValueError(f"unknown marker {marker_name!r}; available: {list(MARKERS)}")

    marker = MARKERS[marker_name]
    num_ancilla = marker.ancilla_count(marker_params, query, query_cards, layout)
    flag = num_state_qubits
    ancillas = list(range(num_state_qubits + 1, num_state_qubits + 1 + num_ancilla))
    num_qubits = num_state_qubits + 1 + num_ancilla

    qc = QuantumCircuit(num_qubits)
    LOADERS[loader_name](qc, model, layout)
    marker.apply(
        qc,
        layout,
        flag,
        ancillas,
        evidence=dict(evidence),
        query=list(query),
        query_cards=list(query_cards),
        params=marker_params,
    )
    return qc.to_gate(label="A"), num_qubits, flag


# ============================================================== Drivers
@dataclass
class AmplitudeResult:
    a_hat: float
    queries: int
    ci: Tuple[float, float]
    info: Dict[str, Any] = field(default_factory=dict)


def driver_mlae(
    measure_hits: Callable[[int, int], int],
    *,
    driver_params: Dict[str, Any],
    rng: np.random.Generator,
) -> AmplitudeResult:
    """Maximum-likelihood amplitude estimation on a fixed Grover-power schedule.

    A geometric schedule [0,1,2,4,...] gives ~O(1/eps) query complexity
    (Suzuki et al. 2020) — quadratically better than classical Monte Carlo's
    O(1/eps^2).
    """
    schedule = list(driver_params.get("schedule", [0, 1, 2, 4, 8]))
    shots = int(driver_params.get("shots", 100))
    hits: List[int] = []
    totals: List[int] = []
    queries = 0
    for m in schedule:
        h = measure_hits(m, shots)
        hits.append(int(h))
        totals.append(shots)
        queries += shots * (2 * m + 1)
    theta = _mle_fit_theta(schedule, hits, totals)
    a = float(math.sin(theta) ** 2)
    return AmplitudeResult(a, queries, (a, a), {"schedule": schedule, "hits": hits})


def _find_next_k(k: int, theta_lo: float, theta_hi: float, k_max: int) -> Tuple[int, bool]:
    """Largest k' in (k, k_max] s.t. (2k'+1)*[theta_lo,theta_hi] stays inside one
    monotonic half-branch of sin^2 (Grinko et al. 2021). Returns (k', up); falls
    back to k (stay, accumulate shots) when no larger power fits."""
    half = math.pi / 2
    for kp in range(k_max, k, -1):
        scale = 2 * kp + 1
        j_lo = math.floor(scale * theta_lo / half)
        j_hi = math.floor(scale * theta_hi / half)
        if j_lo == j_hi:
            return kp, (j_lo % 2 == 0)
    return k, math.floor((2 * k + 1) * theta_lo / half) % 2 == 0


def _invert_theta(p: float, scale: int, j: int) -> float:
    s = math.sqrt(min(1.0, max(0.0, p)))
    half = math.pi / 2
    if j % 2 == 0:
        x = j * half + math.asin(s)
    else:
        x = (j + 1) * half - math.asin(s)
    return x / scale


def driver_iqae(
    measure_hits: Callable[[int, int], int],
    *,
    driver_params: Dict[str, Any],
    rng: np.random.Generator,
) -> AmplitudeResult:
    """Iterative Quantum Amplitude Estimation (Grinko, Gacon, Zoufal, Woerner 2021).

    Estimates a = sin^2(theta) to additive accuracy `epsilon` with confidence
    `1 - alpha`, adaptively increasing the Grover power. Query complexity is
    ~O(1/epsilon) — quadratically better than Monte Carlo.
    """
    epsilon = float(driver_params.get("epsilon", 0.005))
    alpha = float(driver_params.get("alpha", 0.05))
    shots = int(driver_params.get("shots", 100))
    k_max = int(driver_params.get("k_max", 1 << 14))
    max_rounds = int(driver_params.get("max_rounds", 200))
    alpha_round = alpha / max_rounds

    theta_lo, theta_hi = 0.0, math.pi / 2
    k = 0
    cum_hits = 0
    cum_shots = 0
    queries = 0
    rounds = 0
    half = math.pi / 2

    while rounds < max_rounds:
        rounds += 1
        next_k, _up = _find_next_k(k, theta_lo, theta_hi, k_max)
        if next_k != k:
            k = next_k
            cum_hits = 0
            cum_shots = 0
        scale = 2 * k + 1

        cum_hits += int(measure_hits(k, shots))
        cum_shots += shots
        queries += shots * scale

        f = cum_hits / cum_shots
        delta = math.sqrt(math.log(2.0 / alpha_round) / (2.0 * cum_shots))
        p_lo = max(0.0, f - delta)
        p_hi = min(1.0, f + delta)

        j = math.floor(scale * theta_lo / half)
        if j != math.floor(scale * theta_hi / half):
            j = math.floor(scale * ((theta_lo + theta_hi) / 2) / half)
        if j % 2 == 0:
            t_a, t_b = _invert_theta(p_lo, scale, j), _invert_theta(p_hi, scale, j)
        else:
            t_a, t_b = _invert_theta(p_hi, scale, j), _invert_theta(p_lo, scale, j)
        theta_lo = max(theta_lo, min(t_a, t_b))
        theta_hi = min(theta_hi, max(t_a, t_b))
        if theta_hi < theta_lo:
            theta_lo, theta_hi = theta_hi, theta_lo

        a_lo = math.sin(theta_lo) ** 2
        a_hi = math.sin(theta_hi) ** 2
        if (a_hi - a_lo) <= 2 * epsilon:
            break

    a_lo = math.sin(theta_lo) ** 2
    a_hi = math.sin(theta_hi) ** 2
    a_hat = 0.5 * (a_lo + a_hi)
    return AmplitudeResult(
        a_hat, queries, (a_lo, a_hi),
        {"rounds": rounds, "final_k": k, "epsilon": epsilon, "alpha": alpha},
    )


DRIVERS: Dict[str, Callable[..., AmplitudeResult]] = {
    "mlae": driver_mlae,
    "iqae": driver_iqae,
}


# ============================================================== Engine
def _statevector_measure_hits(A_gate, Q_gate, num_qubits, flag, rng):
    def measure_hits(m: int, shots: int) -> int:
        p = measure_good_prob_statevector(A_gate, Q_gate, num_qubits, flag, m)
        return int(rng.binomial(shots, min(1.0, max(0.0, p))))
    return measure_hits


def estimate_amplitude(
    model: DiscreteBayesianNetwork,
    layout: Dict[str, List[int]],
    num_state_qubits: int,
    *,
    loader_name: str,
    marker_name: str,
    marker_params: Dict[str, Any],
    driver_name: str,
    driver_params: Dict[str, Any],
    evidence: Dict[str, int],
    query: Sequence[str],
    query_cards: Sequence[int],
    rng: np.random.Generator,
) -> AmplitudeResult:
    """Compose Loader x Marker x Driver to estimate one amplitude a = P(good)."""
    if driver_name not in DRIVERS:
        raise ValueError(f"unknown driver {driver_name!r}; available: {list(DRIVERS)}")
    A_gate, num_qubits, flag = build_A_operator(
        model, layout, num_state_qubits, loader_name, marker_name, marker_params,
        evidence, query, query_cards,
    )
    Q_gate = build_grover_Q(A_gate, num_qubits, flag)
    measure_hits = _statevector_measure_hits(A_gate, Q_gate, num_qubits, flag, rng)
    return DRIVERS[driver_name](measure_hits, driver_params=driver_params, rng=rng)


def classical_reference(
    model: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: Sequence[str],
    good_cells: Sequence[Dict[str, int]],
) -> float:
    """Exact P(query in good_cells | evidence) via the model CPDs (the truth to
    validate the quantum estimate against)."""
    query = list(query)
    query_cards = [int(model.get_cpds(n).variable_card) for n in query]
    flat = _build_conditional_probs(model, dict(evidence), query)
    good = 0.0
    for cell in good_cells:
        idx = 0
        for v, c in zip(query, query_cards):
            idx = idx * c + int(cell[v])
        good += flat[idx]
    return float(good)


def run_single_query(
    model: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: Sequence[str],
    *,
    loader_name: str = "faithful",
    marker_name: str = "threshold",
    marker_params: Optional[Dict[str, Any]] = None,
    driver_name: str = "iqae",
    driver_params: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Estimate one posterior number P(good | e) = P(good, e) / P(e).

    This is the end-to-end advantage path: a single amplitude rather than a full
    posterior, so the O(1/eps) (estimation) / O(1/sqrt(P(e))) (amplification)
    quantum advantage is not cancelled by per-cell enumeration.
    """
    marker_params = dict(marker_params or {})
    driver_params = dict(driver_params or {})
    rng = np.random.default_rng(seed)

    layout, num_state_qubits = joint_qubit_layout(model)
    query_cards = [int(model.get_cpds(n).variable_card) for n in query]

    good_cells = MARKERS[marker_name].good_cells(marker_params, query, query_cards)

    num = estimate_amplitude(
        model, layout, num_state_qubits,
        loader_name=loader_name, marker_name=marker_name, marker_params=marker_params,
        driver_name=driver_name, driver_params=driver_params,
        evidence=evidence, query=query, query_cards=query_cards, rng=rng,
    )
    if evidence:
        den = estimate_amplitude(
            model, layout, num_state_qubits,
            loader_name=loader_name, marker_name="cells", marker_params={"cells": [{}]},
            driver_name=driver_name, driver_params=driver_params,
            evidence=evidence, query=query, query_cards=query_cards, rng=rng,
        )
    else:
        den = AmplitudeResult(1.0, 0, (1.0, 1.0), {"note": "no evidence: P(e)=1"})

    posterior = (num.a_hat / den.a_hat) if den.a_hat > 0 else 0.0
    posterior = min(1.0, max(0.0, posterior))
    exact = classical_reference(model, evidence, query, good_cells)
    exact_pe = classical_reference(
        model, evidence, query,
        [dict(zip(query, combo)) for combo in product(*[range(c) for c in query_cards])],
    ) if evidence else 1.0

    probs = [1.0 - posterior, posterior]
    return {
        "counts": {"0": 0, "1": 0},
        "probs": probs,
        "raw": None,
        "backend": "qiskit_aer (statevector, upstream-single)",
        "circuit_stats": {
            "mode": "upstream_single",
            "loader": loader_name,
            "marker": marker_name,
            "marker_params": marker_params,
            "driver": driver_name,
            "good_cells": len(good_cells),
            "a_numerator": num.a_hat,
            "a_numerator_ci": list(num.ci),
            "a_denominator": den.a_hat,
            "posterior_estimate": posterior,
            "posterior_exact": exact / exact_pe if exact_pe > 0 else 0.0,
            "abs_error": abs(posterior - (exact / exact_pe if exact_pe > 0 else 0.0)),
            "total_grover_queries": num.queries + den.queries,
            "numerator_info": num.info,
        },
        "transpiled_stats": None,
        "circuit_image": None,
    }


def run_enumerate(
    model: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: Sequence[str],
    *,
    loader_name: str = "faithful",
    driver_name: str = "mlae",
    driver_params: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full-posterior reconstruction by per-cell amplitude estimation, but routed
    through the swappable Loader/Driver registries. Mirrors the legacy QAE output
    so the pipeline plotting/metrics work unchanged."""
    driver_params = dict(driver_params or {})
    rng = np.random.default_rng(seed)

    layout, num_state_qubits = joint_qubit_layout(model)
    query = list(query)
    query_cards = [int(model.get_cpds(n).variable_card) for n in query]
    bits_per = [max(1, int(math.ceil(math.log2(c)))) for c in query_cards]
    padded = 1
    for b in bits_per:
        padded *= 2 ** b

    a_e = estimate_amplitude(
        model, layout, num_state_qubits, loader_name=loader_name,
        marker_name="cells", marker_params={"cells": [{}]},
        driver_name=driver_name, driver_params=driver_params,
        evidence=evidence, query=query, query_cards=query_cards, rng=rng,
    ).a_hat

    a_joint: Dict[Tuple[int, ...], float] = {}
    total_queries = 0
    for combo in product(*[range(c) for c in query_cards]):
        cell = {v: int(x) for v, x in zip(query, combo)}
        res = estimate_amplitude(
            model, layout, num_state_qubits, loader_name=loader_name,
            marker_name="equality", marker_params={"cell": cell},
            driver_name=driver_name, driver_params=driver_params,
            evidence=evidence, query=query, query_cards=query_cards, rng=rng,
        )
        a_joint[combo] = res.a_hat
        total_queries += res.queries

    total = sum(a_joint.values())
    probs = [0.0] * padded
    for combo, a in a_joint.items():
        idx = 0
        for v, b in zip(combo, bits_per):
            idx = (idx << b) | int(v)
        probs[idx] = (a / total) if total > 0 else 1.0 / len(a_joint)

    return {
        "counts": {},
        "probs": probs,
        "raw": None,
        "backend": "qiskit_aer (statevector, upstream-enumerate)",
        "circuit_stats": {
            "mode": "upstream_enumerate",
            "loader": loader_name,
            "driver": driver_name,
            "a_e": a_e,
            "amplitudes_estimated": 1 + len(a_joint),
            "total_grover_queries": total_queries,
        },
        "transpiled_stats": None,
        "circuit_image": None,
    }


def run_upstream(
    model: DiscreteBayesianNetwork,
    evidence: Dict[str, int],
    query: Sequence[str],
    backend_params: Dict[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """Engine entry point: dispatch on query_mode using the requested components."""
    params = dict(backend_params or {})
    query_mode = params.get("query_mode", "enumerate")
    loader_name = params.get("loader", "faithful")
    driver_name = params.get("driver", "mlae")
    driver_params = dict(params.get("driver_params", {}))

    if query_mode == "single":
        return run_single_query(
            model, evidence, query,
            loader_name=loader_name,
            marker_name=params.get("marker", "threshold"),
            marker_params=dict(params.get("marker_params", {})),
            driver_name=driver_name, driver_params=driver_params, seed=seed,
        )
    if query_mode == "enumerate":
        return run_enumerate(
            model, evidence, query, loader_name=loader_name,
            driver_name=driver_name, driver_params=driver_params, seed=seed,
        )
    raise ValueError(f"query_mode must be 'single' or 'enumerate', got {query_mode!r}")
