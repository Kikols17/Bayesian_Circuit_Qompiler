from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging

try:
    from qiskit_aer import Aer
except Exception:
    Aer = None

try:
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, ReadoutError
except Exception:
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    except Exception:
        NoiseModel = None
        depolarizing_error = None
        ReadoutError = None

from src.circuit_builder import get_circuit_builder
from src.circuits.visualize import save_qiskit_circuit_image
from src.config.types import BackendConfig, CircuitConfig
from src.qompiler.base import CircuitSpec


def _get_token_from_env() -> Tuple[Optional[str], Optional[str]]:
    candidate_vars = ["QISKIT_IBM_TOKEN", "IBMQ_TOKEN", "IBMQ_API_TOKEN", "IBM_TOKEN"]
    for var in candidate_vars:
        val = os.getenv(var)
        if val:
            return val, var
    try:
        from dotenv import load_dotenv

        load_dotenv()
        for var in candidate_vars:
            val = os.getenv(var)
            if val:
                return val, f".env:{var}"
    except Exception:
        pass
    return None, None


def _name_of_backend(b: object) -> Optional[str]:
    try:
        name = getattr(b, "name", None)
        if callable(name):
            return name()
        if name:
            return name
        return str(b)
    except Exception:
        try:
            return str(b)
        except Exception:
            return None


def _is_hardware_backend_obj(b: object, name: Optional[str]) -> bool:
    if not name:
        return False
    nl = name.lower()
    # obvious simulator name hints
    if "simulator" in nl or "qasm_simulator" in nl:
        return False
    try:
        # Many provider runtime/backend objects expose a `configuration` with `simulator` flag
        cfg = None
        if hasattr(b, "configuration"):
            try:
                cfg = b.configuration()
            except Exception:
                cfg = getattr(b, "configuration", None)
        sim_attr = getattr(cfg, "simulator", None) if cfg is not None else None
        if sim_attr is not None:
            return not bool(sim_attr)
        # If status is available, try to ensure it's operational
        if hasattr(b, "status"):
            try:
                st = b.status() if callable(getattr(b, "status", None)) else b.status
                op = getattr(st, "operational", None)
                if op is False:
                    return False
            except Exception:
                pass
    except Exception:
        return False
    # If nothing indicates simulator, assume hardware
    return True


def run_qiskit(
    output_dir: str,
    spec: CircuitSpec,
    circuit_config: CircuitConfig,
    backend_config: BackendConfig,
    evidence: Dict[str, int],
    query: List[str],
    save_circuit_image: bool = False,
) -> Dict[str, Any]:
    """Compile and run the circuit using Qiskit/Aer or IBM backends.

    This will try to use IBM hardware when `backend_config.type` contains
    "ibm" or when `backend_config.device` starts with "ibmq". It searches
    for a real device (non-simulator) using the runtime -> provider -> IBMQ
    lookup order and raises if no hardware is available.
    """
    logger = logging.getLogger(__name__)

    builder = get_circuit_builder("qiskit")
    circuit = builder.build(
        spec=spec,
        evidence=evidence,
        query=query,
        circuit_config=circuit_config,
        backend_config=backend_config,
    )

    backend_name = backend_config.device or "aer_simulator"
    backend = None

    use_ibm = False

    # Support a noisy simulator target using the prefix `noisy:TARGET_NAME`.
    # Example: `noisy:ibm_fez` will build a NoiseModel from the IBM backend
    # `ibm_fez` (via provider/runtime/IBMQ) and run locally on Aer with that
    # noise model.
    noisy_sim = False
    noise_source_backend_name = None
    noise_model = None
    if isinstance(backend_name, str) and backend_name.startswith("noisy:"):
        noisy_sim = True
        parts = backend_name.split(":", 1)
        noise_source_backend_name = parts[1] if len(parts) > 1 else None
        backend_name = "aer_simulator"
        if noise_source_backend_name and noise_source_backend_name.startswith("ibm"):
            use_ibm = True
    if backend_config and getattr(backend_config, "type", "").lower().find("ibm") != -1:
        use_ibm = True
    if isinstance(backend_name, str) and backend_name.startswith("ibmq"):
        use_ibm = True

    token, token_origin = _get_token_from_env()
    if token:
        os.environ["QISKIT_IBM_TOKEN"] = token
        logger.info("Using IBM token from %s", token_origin or "environment")

    service = None
    provider = None

    if use_ibm:
        # Try IBM Runtime (preferred). prefer QiskitRuntimeService name, fall back
        # to any available runtime class exported by qiskit_ibm_runtime.
        RuntimeService = None
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService as QRS

            RuntimeService = QRS
        except Exception:
            try:
                from qiskit_ibm_runtime import IBMRuntimeService as QRS

                RuntimeService = QRS
            except Exception:
                RuntimeService = None

        if RuntimeService is not None:
            try:
                try:
                    service = (
                        RuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL"), token=token, instance=os.getenv("QISKIT_IBM_INSTANCE"))
                        if token
                        else RuntimeService()
                    )
                except TypeError:
                    service = RuntimeService()

                # try exact name
                _svc_get = getattr(service, "backend", None) or getattr(service, "get_backend", None)
                try:
                    backend = _svc_get(backend_name)
                except Exception:
                    # enumerate and pick a hardware device
                    backends_list = None
                    try:
                        backends_list = service.backends()
                    except Exception:
                        try:
                            backends_list = service.available_backends()
                        except Exception:
                            backends_list = None

                    if backends_list:
                        # first try exact match
                        for b in backends_list:
                            name = _name_of_backend(b)
                            if name and name.lower() == backend_name.lower():
                                try:
                                    backend = _svc_get(name)
                                    backend_name = name
                                    break
                                except Exception:
                                    continue
                        # otherwise pick first hardware device
                        if backend is None:
                            for b in backends_list:
                                name = _name_of_backend(b)
                                if not name:
                                    continue
                                try:
                                    if _is_hardware_backend_obj(b, name):
                                        backend = _svc_get(name)
                                        backend_name = name
                                        break
                                except Exception:
                                    continue
            except Exception:
                service = None

        # Try IBM Provider
        if backend is None:
            try:
                from qiskit_ibm_provider import IBMProvider

                try:
                    provider = (
                        IBMProvider(channel=os.getenv("QISKIT_IBM_CHANNEL"), token=token, instance=os.getenv("QISKIT_IBM_INSTANCE"))
                        if token
                        else IBMProvider()
                    )
                except TypeError:
                    provider = IBMProvider()

                try:
                    backend = provider.get_backend(backend_name)
                except Exception:
                    prov_backends = None
                    try:
                        prov_backends = provider.backends()
                    except Exception:
                        try:
                            prov_backends = provider.available_backends()
                        except Exception:
                            prov_backends = None

                    if prov_backends:
                        for b in prov_backends:
                            name = _name_of_backend(b)
                            if name and name.lower() == backend_name.lower():
                                try:
                                    backend = provider.get_backend(name)
                                    backend_name = name
                                    break
                                except Exception:
                                    continue
                        if backend is None:
                            for b in prov_backends:
                                name = _name_of_backend(b)
                                if not name:
                                    continue
                                try:
                                    if _is_hardware_backend_obj(b, name):
                                        backend = provider.get_backend(name)
                                        backend_name = name
                                        break
                                except Exception:
                                    continue
            except Exception:
                provider = None

        # Try legacy IBMQ
        if backend is None:
            try:
                try:
                    from qiskit.providers.ibmq import IBMQ as IBMQ_class  # type: ignore
                except Exception:
                    try:
                        from qiskit import IBMQ as IBMQ_class  # type: ignore
                    except Exception:
                        IBMQ_class = None

                if IBMQ_class is not None:
                    try:
                        if token:
                            try:
                                IBMQ_class.enable_account(token)
                            except Exception:
                                try:
                                    IBMQ_class.save_account(token, overwrite=True)
                                    IBMQ_class.load_account()
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    try:
                        prov = IBMQ_class.get_provider()
                        backend = prov.get_backend(backend_name)
                    except Exception:
                        try:
                            ibmq_backends = IBMQ_class.backends()
                            for b in ibmq_backends:
                                name = _name_of_backend(b)
                                if not name:
                                    continue
                                try:
                                    if _is_hardware_backend_obj(b, name):
                                        try:
                                            backend = prov.get_backend(name)
                                            backend_name = name
                                            break
                                        except Exception:
                                            continue
                                except Exception:
                                    continue
                        except Exception:
                            pass
            except Exception:
                pass

        if backend is None:
            # If user requested a noisy simulator, attempt to build a noise
            # model from the requested IBM target and run locally on Aer.
            if noisy_sim and noise_source_backend_name:
                target_backend = None
                # try provider
                if provider is not None:
                    try:
                        target_backend = provider.get_backend(noise_source_backend_name)
                    except Exception:
                        try:
                            prov_backends = provider.backends()
                        except Exception:
                            try:
                                prov_backends = provider.available_backends()
                            except Exception:
                                prov_backends = None
                        if prov_backends:
                            for b in prov_backends:
                                name = _name_of_backend(b)
                                if name and name.lower() == noise_source_backend_name.lower():
                                    try:
                                        target_backend = provider.get_backend(name)
                                        break
                                    except Exception:
                                        continue

                # try runtime service
                if target_backend is None and service is not None:
                    _svc_get2 = getattr(service, "backend", None) or getattr(service, "get_backend", None)
                    try:
                        target_backend = _svc_get2(noise_source_backend_name)
                    except Exception:
                        try:
                            svc_backends = service.backends()
                        except Exception:
                            try:
                                svc_backends = service.available_backends()
                            except Exception:
                                svc_backends = None
                        if svc_backends:
                            for b in svc_backends:
                                name = _name_of_backend(b)
                                if name and name.lower() == noise_source_backend_name.lower():
                                    try:
                                        target_backend = _svc_get2(name)
                                        break
                                    except Exception:
                                        continue

                # try legacy IBMQ
                if target_backend is None:
                    try:
                        try:
                            from qiskit.providers.ibmq import IBMQ as IBMQ_class  # type: ignore
                        except Exception:
                            try:
                                from qiskit import IBMQ as IBMQ_class  # type: ignore
                            except Exception:
                                IBMQ_class = None

                        if IBMQ_class is not None:
                            try:
                                prov = IBMQ_class.get_provider()
                                try:
                                    target_backend = prov.get_backend(noise_source_backend_name)
                                except Exception:
                                    try:
                                        ibmq_backends = IBMQ_class.backends()
                                        for b in ibmq_backends:
                                            name = _name_of_backend(b)
                                            if name and name.lower() == noise_source_backend_name.lower():
                                                try:
                                                    target_backend = prov.get_backend(name)
                                                    break
                                                except Exception:
                                                    continue
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Build noise model from discovered target backend
                if target_backend is not None and NoiseModel is not None:
                    try:
                        noise_model = NoiseModel.from_backend(target_backend)
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.exception("NoiseModel.from_backend failed for %s: %s", noise_source_backend_name, e)
                        # Try retrieving backend properties via service/provider/legacy and build from that
                        props = None
                        try:
                            if service is not None and hasattr(service, "backend_properties"):
                                try:
                                    props = service.backend_properties(noise_source_backend_name)
                                    logger.debug("Got backend properties from service: %s", type(props))
                                except Exception:
                                    logger.exception("service.backend_properties failed for %s", noise_source_backend_name)
                        except Exception:
                            logger.exception("service.backend_properties attempt error")

                        if props is None and provider is not None:
                            try:
                                if hasattr(provider, "backend_properties"):
                                    try:
                                        props = provider.backend_properties(noise_source_backend_name)
                                        logger.debug("Got backend properties from provider: %s", type(props))
                                    except Exception:
                                        logger.exception("provider.backend_properties failed for %s", noise_source_backend_name)
                            except Exception:
                                logger.exception("provider.backend_properties attempt error")

                        if props is None:
                            try:
                                if hasattr(target_backend, "properties"):
                                    try:
                                        props = target_backend.properties()
                                        logger.debug("Got backend.properties(): %s", type(props))
                                    except Exception:
                                        logger.exception("target_backend.properties() call failed for %s", noise_source_backend_name)
                            except Exception:
                                logger.exception("target_backend.properties attempt error")

                        if props is not None:
                            try:
                                noise_model = NoiseModel.from_backend(props)
                            except Exception:
                                logger.exception("NoiseModel.from_backend(props) failed for %s", noise_source_backend_name)

                # If we still don't have a noise model, try constructing an
                # approximate local noise model (depolarizing + readout).
                if noise_model is None and NoiseModel is not None:
                    try:
                        # reasonable defaults for single- and two-qubit errors
                        p1 = 1e-3
                        p2 = 2e-2
                        readout_p = 0.02
                        try:
                            dep_err = depolarizing_error
                            RO = ReadoutError
                        except Exception:
                            dep_err = None
                            RO = None

                        if dep_err is not None:
                            nm = NoiseModel()
                            e1 = dep_err(p1, 1)
                            e2 = dep_err(p2, 2)
                            # add common single-qubit and two-qubit gate errors
                            nm.add_all_qubit_quantum_error(e1, ["u1", "u2", "u3", "x", "y", "z", "h", "rx", "ry", "rz", "sx", "sxdg", "id"])
                            nm.add_all_qubit_quantum_error(e2, ["cx", "cz"])
                            try:
                                if RO is not None:
                                    ro = RO([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]])
                                    nm.add_all_qubit_readout_error(ro, range(circuit.num_qubits))
                            except Exception:
                                logger.exception("Failed to add readout error to approximate noise model")
                            noise_model = nm
                            logger.info("Using approximate local noise model (p1=%s, p2=%s, readout=%s)", p1, p2, readout_p)
                    except Exception:
                        logging.getLogger(__name__).exception("Failed to construct approximate noise model")

                if noise_model is not None:
                    if Aer is None:
                        raise RuntimeError("Aer is required for noisy simulation.")
                    backend = Aer.get_backend(backend_name)
                    backend_name = f"noisy:{noise_source_backend_name}"
                else:
                    raise RuntimeError("Requested noisy simulator target not available or noise model could not be created.")
            else:
                raise RuntimeError("No IBM hardware backend available; ensure your account has access to a real device and the token is valid.")

    # If not using IBM or no IBM backend, fall back to Aer
    if backend is None:
        if Aer is None:
            raise RuntimeError("Aer is not available and no IBM backend was selected.")
        backend = Aer.get_backend(backend_name)

    circuit = circuit.copy()
    circuit = circuit.decompose(reps=1)

    circuit_stats = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "operation_counts": dict(circuit.count_ops()),
        "total_operations": sum(circuit.count_ops().values()),
    }
    circuit_image = None
    if save_circuit_image:
        circuit_image = save_qiskit_circuit_image(output_dir, circuit)

    counts: Dict[str, int] = {}
    memory = None

    if service is not None:
        # qiskit-ibm-runtime 0.40+ removed backend.run(); use SamplerV2 primitives.
        from qiskit_ibm_runtime import SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(circuit)

        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa_circuit], shots=circuit_config.shots)
        result = job.result()

        pub_result = result[0]
        for creg in isa_circuit.cregs:
            try:
                bit_array = getattr(pub_result.data, creg.name)
                for bitstring, count in bit_array.get_counts().items():
                    counts[bitstring] = counts.get(bitstring, 0) + count
            except Exception:
                pass
        try:
            first_creg = isa_circuit.cregs[0]
            memory = getattr(pub_result.data, first_creg.name).get_bitstrings()
        except Exception:
            memory = None
    else:
        # Aer local simulator — backend.run() still works here
        run_kwargs: Dict[str, Any] = {"shots": circuit_config.shots, "memory": True}
        if noise_model is not None:
            run_kwargs["noise_model"] = noise_model
        job = backend.run(circuit, **run_kwargs)
        result = job.result()
        counts = result.get_counts()
        try:
            memory = result.get_memory()
        except Exception:
            memory = None

    total_shots = sum(counts.values())
    probs = {key: val / total_shots for key, val in counts.items()}

    return {
        "counts": counts,
        "probs": probs,
        "raw": memory,
        "backend": backend_name,
        "circuit_stats": circuit_stats,
        "circuit_image": circuit_image,
    }
