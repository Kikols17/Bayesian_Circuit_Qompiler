Running with IBM Quantum (IBMQ)

- **Requirements**: install one of the IBM provider packages and Aer plus dotenv:

```
pip install qiskit-aer
pip install qiskit-ibm-provider   # recommended OR
pip install qiskit-ibm-runtime    # alternative runtime package
pip install python-dotenv
```

- **Environment**: ensure `.env` contains `QISKIT_IBM_TOKEN` (and optionally `QISKIT_IBM_CHANNEL` and `QISKIT_IBM_INSTANCE`). This project looks for those vars at runtime.

- **Config**: I created `config/ship_ibmq.yaml` which targets an IBM backend via `inference.backend.type: "qiskit_ibm"`. Adjust `inference.backend.device` to a specific backend name (for example `ibmq_qasm_simulator` or a real device like `ibmq_manila`).

- **Run**:

```
python3 run_pipeline.py --config config/ship_ibmq.yaml
```

- **Notes**:
  - The code attempts to connect in this order: `qiskit-ibm-runtime` -> `qiskit-ibm-provider` -> legacy `qiskit.IBMQ`.
  - If you target a real device, expect queue times and stricter shot limits.
  - If you get provider import errors, install one of the packages above and try again.
