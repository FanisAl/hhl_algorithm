# HHL Algorithm — Quantum Linear Systems Solver

A from-scratch, gate-level implementation of the **Harrow–Hassidim–Lloyd (HHL)**
quantum algorithm in [Qiskit](https://www.ibm.com/quantum/qiskit). HHL solves the
linear system **A x = b** for the (normalized) solution vector, and is the
canonical example of a quantum algorithm with an *exponential* asymptotic speedup
over classical solvers for suitable sparse, well-conditioned matrices — it
produces the quantum state `|x⟩` in time `O(log N · κ² · 1/ε)` versus `O(N · κ)`
for classical conjugate gradient.

This repository is an educational implementation: every stage is built from
primitive gates (no prebuilt `HHL` / `PhaseEstimation` convenience classes), so
the quantum phase estimation, eigenvalue-inversion rotation, and uncomputation
are all visible and inspectable. It also deliberately demonstrates the dominant
finite-resource **failure mode** of HHL — quantum phase estimation (QPE) register
overflow — which is usually hidden by higher-level libraries.

## What this implements

The circuit acts on a 2×2 Hermitian system using a register of
`nc` *clock* qubits, one *system* qubit, and one *ancilla* qubit. The four stages
mirror the textbook HHL construction:

1. **State preparation** — encode `b = (1/√2)(1, 1)ᵀ` as `H|0⟩` on the system qubit.
2. **Quantum Phase Estimation (QPE)** — Hadamard the clock register, apply the
   controlled time-evolution ladder `controlled-e^{iAt·2^k}` (computed exactly via
   `scipy.linalg.expm`, no Trotterization at this scale), then an inverse QFT to
   write each eigenvalue `λ` as an integer into the clock register.
3. **Ancilla rotation** — a multi-controlled `R_Y(2·arcsin(C/λ̃))` loads the
   classically intractable `1/λ` factor into the amplitude. Amplitude
   amplification is intentionally omitted so the raw ancilla success probability
   `P = C²/λ̃²` stays observable.
4. **Inverse QPE** — uncompute the clock register so the answer is isolated in
   the `{ancilla = |1⟩, clock = |0…0⟩}` subspace.

The solution is read directly from the statevector simulator and compared against
`numpy.linalg.solve`. Three experiments are included:

| Experiment | System | κ | Outcome |
|---|---|---|---|
| 1 — well-conditioned | `[[1.5, 0.5], [0.5, 1.5]]` | 2 | Fidelity ≈ 1.000, solution recovered |
| 2 — ill-conditioned | κ = 200, eigenvalues {0.1, 20} | 200 | Large eigenvalue overflows the 3-qubit clock register |
| 3 — direction collapse | `diag(2, 3)` | 1.5 | One extra clock qubit changes fidelity 1.000 → 0.308 |

Each experiment also reports **circuit-depth and CNOT-count metrics** at
transpiler optimization levels 0 and 3 (e.g. the well-conditioned circuit is
depth ≈ 96 with ≈ 58 CNOTs after optimization), illustrating why even this small
demonstration is well beyond reliable execution on current NISQ hardware.

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/<your-username>/hhl-algorithm.git
cd hhl-algorithm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies (pinned in `requirements.txt`): `qiskit==2.3.1`,
`qiskit-aer==0.17.2`, `numpy`, `scipy`, `tabulate`.

## Usage

Run the full demonstration (all three experiments + scientific summary):

```bash
python src/hhl_algorithm.py
```

Or use the building blocks directly to solve your own well-conditioned system:

```python
import numpy as np
from src.hhl_algorithm import build_hhl_circuit, simulate_and_extract, fidelity

# A well-conditioned 2x2 system whose eigenvalues {1, 2} map exactly onto a
# 3-qubit clock register at t = pi/2.
A = np.array([[1.5, 0.5],
              [0.5, 1.5]])
b = (1 / np.sqrt(2)) * np.array([1.0, 1.0])   # encoded as H|0>

circuit = build_hhl_circuit(A, b, t=np.pi / 2, nc=3, C=1.9, tilde_lambdas=[2, 4])
x_quantum, p_success = simulate_and_extract(circuit, nc=3)

x_classical = np.linalg.solve(A, b)
print("quantum  :", x_quantum)
print("classical:", x_classical / np.linalg.norm(x_classical))
print("fidelity :", fidelity(x_quantum, x_classical))   # ~1.0
```

## Tests

A `pytest` suite validates the solver against the classical solution (fidelity,
normalization, register size, and input-validation guards):

```bash
pip install -r requirements.txt
python -m pytest
```

## License

[MIT](LICENSE)
