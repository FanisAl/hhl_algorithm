"""
HHL Algorithm -- Educational From-Scratch Implementation
========================================================

A single-file, fully executable demonstration of the Harrow-Hassidim-Lloyd
(HHL) quantum linear-solver algorithm, built from gate-level primitives on
modern Qiskit (>= 1.0). Three contrasting experiments are run on the same
small architecture to expose both the success regime and the dominant
finite-resource failure mode (QPE register overflow):

  * Experiment 1 -- well-conditioned 2x2 system (kappa = 2).
  * Experiment 2 -- ill-conditioned 2x2 system (kappa = 200) whose large
    eigenvalue overflows the 3-qubit clock register.
  * Experiment 3 -- diagonal system A = diag(2, 3) where overflow rotates
    the solution onto the wrong eigenvector, collapsing the fidelity.

No prebuilt HHL or QPE convenience classes are used. Amplitude amplification
is intentionally omitted so that the raw ancilla success probability remains
visible.

The reusable building blocks -- `build_hhl_circuit`, `simulate_and_extract`,
and `fidelity` -- are importable and unit-tested (see ``tests/``). Running the
module as a script (``python src/hhl_algorithm.py``) executes the three
narrative experiments and prints the scientific summary.
"""

import sys

import numpy as np
from scipy.linalg import expm
from tabulate import tabulate

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit_aer import AerSimulator

# Statevector simulator (no shot noise, gives access to the full amplitude
# distribution so we can read off the HHL signal directly).
simulator = AerSimulator(method="statevector")

np.set_printoptions(precision=4, suppress=True)


# ===================================================
# SECTION 1: HELPER FUNCTIONS
# ===================================================

def make_controlled_unitary(A: np.ndarray, t: float, k: int) -> Gate:
    """Build the controlled-U^(2^k) gate where U = e^{i A t}.

    For small (2x2) Hermitian ``A`` we use exact matrix exponentiation via
    :func:`scipy.linalg.expm` -- no Trotterization is needed at this scale, and
    the educational goal is transparency rather than scalability.

    Parameters
    ----------
    A : np.ndarray
        Hermitian system matrix of the linear system ``A x = b``.
    t : float
        Hamiltonian-simulation time. Negate ``t`` to obtain U^{-1} (used by
        the inverse QPE stage), since ``(e^{i A t})^dagger = e^{-i A t}``.
    k : int
        Clock-qubit index; the returned gate implements ``U`` raised to the
        ``2^k`` power, i.e. ``e^{i A t 2^k}``.

    Returns
    -------
    qiskit.circuit.Gate
        A single-control controlled-unitary gate ready to ``append`` with
        ``[control_qubit, target_qubit]``.
    """
    U = expm(1j * A * t * (2 ** k))
    gate = UnitaryGate(U, label=f"e^iAt*2^{k}")
    return gate.control(1)


def apply_ancilla_rotation(circuit: QuantumCircuit,
                           ancilla,
                           clock_qubits,
                           tilde_lambda: int,
                           C: float,
                           nc: int) -> None:
    """Apply the eigenvalue-inversion R_Y rotation onto the ancilla qubit.

    Implements ``R_Y(2 * arcsin(C / tilde_lambda))`` on ``ancilla``, conditioned
    on the clock register encoding the integer ``tilde_lambda`` (little-endian:
    ``clock_qubits[0]`` is the LSB). This is the step that loads the classically
    intractable ``1/lambda`` factor into the amplitudes.

    Strategy: bit-flip sandwich. Flip every clock qubit whose target bit is 0,
    apply an all-|1> multi-controlled R_Y, then undo the flips.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to mutate in place.
    ancilla
        Index of the ancilla qubit that is rotated.
    clock_qubits
        Sequence of clock-qubit indices, LSB first.
    tilde_lambda : int
        Nonzero integer eigenvalue estimate produced by QPE. Must be > 0.
    C : float
        Rotation scale constant; must satisfy ``C <= min(tilde_lambdas)`` so the
        ``arcsin`` argument stays in [-1, 1].
    nc : int
        Number of clock qubits.

    Raises
    ------
    ValueError
        If ``tilde_lambda == 0`` (an eigenvalue that aliased to zero through QPE
        register overflow and must be excluded before calling this function).
    """
    if tilde_lambda == 0:
        raise ValueError(
            "apply_ancilla_rotation called with tilde_lambda=0. "
            "This eigenvalue aliases to zero (QPE register overflow) and "
            "must be excluded from tilde_lambdas before calling this function."
        )
    theta = 2.0 * np.arcsin(C / tilde_lambda)

    # Flip qubits where the target bit is 0.
    for bit_pos in range(nc):
        if not (tilde_lambda >> bit_pos) & 1:
            circuit.x(clock_qubits[bit_pos])

    # Multi-controlled R_Y on ancilla, controlled by all clock qubits = |1>.
    circuit.mcry(theta, clock_qubits, ancilla)

    # Undo the bit-flips.
    for bit_pos in range(nc):
        if not (tilde_lambda >> bit_pos) & 1:
            circuit.x(clock_qubits[bit_pos])


def extract_quantum_solution(statevector: np.ndarray, nc: int):
    """Read the HHL solution and success probability out of the statevector.

    Register layout (Qiskit little-endian):
        bit 0 (LSB)   = ancilla
        bit 1         = system qubit
        bits 2..nc+1  = clock qubits

    The valid HHL signal lives in the subspace ``{ancilla = |1>}`` AND
    ``{clock = |0...0>}``. In the perfect case the clock register is fully
    uncomputed by the inverse QPE and we lose nothing; in an aliased case any
    residual non-zero-clock amplitude is incoherent noise and must NOT be
    summed in.

    Parameters
    ----------
    statevector : np.ndarray
        Full ``2^(nc+2)`` amplitude vector returned by the simulator.
    nc : int
        Number of clock qubits.

    Returns
    -------
    (np.ndarray, float)
        ``x_quantum`` -- the length-2 normalized solution amplitudes, and
        ``p_success`` -- the probability mass in the valid subspace.

    Raises
    ------
    ValueError
        If the success probability is numerically zero (signals a construction
        error, a bad ``C``, or a fully overflowed problem).
    """
    n_total = nc + 2  # clock + system + ancilla
    assert len(statevector) == 2 ** n_total

    anc_mask = 1                          # bit 0: ancilla = |1>
    sys_mask = 2                          # bit 1: system qubit
    clock_mask = ((1 << nc) - 1) << 2     # bits 2..nc+1: clock qubits

    # Indices with ancilla = |1> AND clock = |0...0>.
    indices_valid = [
        i for i in range(2 ** n_total)
        if (i & anc_mask) and not (i & clock_mask)
    ]

    amp_sys0 = sum(statevector[i] for i in indices_valid if not (i & sys_mask))
    amp_sys1 = sum(statevector[i] for i in indices_valid if (i & sys_mask))

    raw = np.array([amp_sys0, amp_sys1], dtype=complex)
    p_success = float(np.real(np.dot(raw.conj(), raw)))

    if p_success < 1e-12:
        raise ValueError(
            "Ancilla success probability in clock=0 subspace is numerically "
            "zero. Check C, tilde_lambdas, circuit construction, and IQPE."
        )

    x_quantum = raw / np.sqrt(p_success)
    return x_quantum, p_success


# ===================================================
# SECTION 2: HHL CIRCUIT BUILDER
# ===================================================

def build_hhl_circuit(A: np.ndarray,
                      b: np.ndarray,
                      t: float,
                      nc: int,
                      C: float,
                      tilde_lambdas) -> QuantumCircuit:
    """Construct the full HHL circuit for a 2x2 system.

    Register layout (Qiskit little-endian):
        q[0]        : ancilla qubit
        q[1]        : system qubit
        q[2..nc+1]  : clock qubits (q[2] = LSB, controls U^1)

    Stages (each delimited by a labelled barrier):
        A -- State Preparation (encodes ``b``)
        B -- Quantum Phase Estimation (eigenvalues into the clock register)
        C -- Ancilla Rotation (loads the 1/lambda factor)
        D -- Inverse QPE (uncomputes the clock register)

    Parameters
    ----------
    A : np.ndarray
        Hermitian 2x2 system matrix.
    b : np.ndarray
        Right-hand side. Only ``b = (1/sqrt(2))(1, 1)^T = H|0>`` is supported,
        since Stage A is a single primitive Hadamard.
    t : float
        Hamiltonian-simulation time chosen so the eigenvalues map onto the
        integers in ``tilde_lambdas``.
    nc : int
        Number of clock qubits.
    C : float
        Ancilla-rotation scale constant (``C <= min(tilde_lambdas)``).
    tilde_lambdas : Sequence[int]
        Nonzero integer eigenvalue estimates to invert. Eigenvalues that alias
        to 0 through register overflow are excluded by the caller.

    Returns
    -------
    QuantumCircuit
        The assembled HHL circuit (no measurements; meant for statevector
        simulation).

    Raises
    ------
    ValueError
        If ``b`` is not the supported ``(1/sqrt(2))(1, 1)^T`` state.
    """
    # Stage A hardcodes H|0> = (1/sqrt(2))(1,1)^T. Validate b here so callers
    # get a clear error rather than silently solving the wrong system.
    expected_b = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0])
    if not np.allclose(b, expected_b):
        raise ValueError(
            "build_hhl_circuit only supports b = (1/sqrt(2))(1,1)^T. "
            "A different b requires replacing Stage A with the appropriate "
            "state-preparation gate."
        )

    n_total = nc + 2
    qc = QuantumCircuit(n_total, name=f"HHL_nc{nc}")

    ancilla = 0
    system_qubit = 1
    clock_qubits = list(range(2, 2 + nc))   # clock_qubits[0] = q[2] = LSB

    # -------- Stage A: State Preparation --------
    # b = (1/sqrt(2))(1,1)^T = H|0>. A primitive Hadamard is used rather than
    # QuantumCircuit.initialize() / StatePreparation, both of which are
    # high-level instructions and are avoided here for transparency.
    qc.h(system_qubit)
    qc.barrier(label="A: State Prep")

    # -------- Stage B: QPE --------
    for q in clock_qubits:
        qc.h(q)
    # Controlled e^{iAt * 2^k}, with clock_qubits[k] (= q[2+k]) as the control.
    # Standard little-endian phase-kickback ladder: q[2] controls U^1, q[3]
    # controls U^2, q[4] controls U^4, ...
    for k in range(nc):
        cU = make_controlled_unitary(A, t, k)
        qc.append(cU, [clock_qubits[k], system_qubit])
    # Inverse QFT on the clock register. do_swaps=True undoes the bit-order
    # reversal of the standard QFT so the readout integer matches the
    # little-endian phase encoding produced by the controlled-U ladder above.
    iqft = QFT(num_qubits=nc, inverse=True, do_swaps=True, name="IQFT_clock")
    qc.append(iqft, clock_qubits)
    qc.barrier(label="B: QPE")

    # -------- Stage C: Ancilla Rotation --------
    # Amplitude Amplification is intentionally omitted so the raw ancilla
    # success probability P = C^2/tilde_lambda^2 stays visible.
    for tl in tilde_lambdas:
        apply_ancilla_rotation(qc, ancilla, clock_qubits,
                               tilde_lambda=tl, C=C, nc=nc)
    qc.barrier(label="C: Ancilla R_Y")

    # -------- Stage D: IQPE --------
    # Mirror of stage B with the controlled-U gates inverted: forward QFT, then
    # controlled e^{-iAt * 2^k} in reverse order, then H on the clock register.
    qft = QFT(num_qubits=nc, inverse=False, do_swaps=True, name="QFT_clock")
    qc.append(qft, clock_qubits)
    for k in reversed(range(nc)):
        cUdg = make_controlled_unitary(A, -t, k)
        qc.append(cUdg, [clock_qubits[k], system_qubit])
    for q in clock_qubits:
        qc.h(q)
    qc.barrier(label="D: IQPE")

    return qc


# ===================================================
# SECTION 3: RUN / METRICS UTILITIES
# ===================================================

def simulate_and_extract(circuit: QuantumCircuit, nc: int):
    """Simulate the circuit and return ``(x_quantum, p_success)``.

    Appends ``save_statevector``, transpiles to the simulator's basis, runs the
    Aer statevector backend, and forwards the result to
    :func:`extract_quantum_solution`.

    The transpilation step is necessary because Aer does not natively consume
    controlled ``UnitaryGate`` ('c-unitary') instructions -- it is handed a
    circuit compiled to a basis it understands instead.
    """
    sim_circuit = circuit.copy()
    sim_circuit.save_statevector()
    sim_circuit = transpile(sim_circuit, simulator, optimization_level=0)
    result = simulator.run(sim_circuit).result()
    statevector = np.asarray(result.get_statevector())
    return extract_quantum_solution(statevector, nc)


def transpile_metrics(circuit: QuantumCircuit):
    """Transpile to ``{u, cx}`` at optimization levels 0 and 3.

    Returns a dict keyed by level, each holding ``depth``, ``gate_count`` and
    ``cx`` (CNOT) counts -- a rough NISQ-cost proxy for the circuit.
    """
    out = {}
    for level in (0, 3):
        t_circ = transpile(circuit,
                           basis_gates=["u", "cx"],
                           optimization_level=level)
        out[level] = {
            "depth": t_circ.depth(),
            "gate_count": sum(t_circ.count_ops().values()),
            "cx": t_circ.count_ops().get("cx", 0),
        }
    return out


def fidelity(x_quantum: np.ndarray, x_classical: np.ndarray) -> float:
    """Return ``F = |<x_quantum|x_hat_classical>|^2`` with normalized classical x.

    Fidelity is a cosine-angle (direction) measure: it is invariant to global
    phase and to the overall scale of ``x_classical``.
    """
    x_hat = x_classical / np.linalg.norm(x_classical)
    overlap = np.vdot(x_quantum, x_hat)
    return float(np.abs(overlap) ** 2)


def print_metrics_table(label: str, metrics: dict) -> None:
    """Pretty-print the level-0 vs level-3 transpilation metrics table."""
    rows = [
        ["Circuit depth", metrics[0]["depth"], metrics[3]["depth"]],
        ["Total gate count", metrics[0]["gate_count"], metrics[3]["gate_count"]],
        ["CNOT (cx) count", metrics[0]["cx"], metrics[3]["cx"]],
    ]
    print(f"\n--- Transpilation metrics: {label} ---")
    print(tabulate(rows,
                   headers=["Metric", "Level 0", "Level 3"],
                   tablefmt="github"))


def prob_bar(p: float, width: int = 10) -> str:
    """Return an ASCII bar (e.g. ``[#####.....]``) for a probability in [0, 1]."""
    filled = int(round(p * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ===================================================
# SECTION 4: EXPERIMENT 1 -- WELL-CONDITIONED CASE
# ===================================================

def run_experiment_1() -> dict:
    """Well-conditioned 2x2 system (kappa = 2); both eigenvalues fit the register."""
    print("=" * 71)
    print("EXPERIMENT 1 -- WELL-CONDITIONED CASE  (kappa = 2)")
    print("=" * 71)

    nc = 3

    A1 = np.array([[1.5, 0.5],
                   [0.5, 1.5]])
    b1 = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0])

    eigs1 = np.linalg.eigvalsh(A1)
    kappa1 = eigs1.max() / eigs1.min()
    print(f"  Eigenvalues of A1     = {eigs1}")
    print(f"  Condition number     kappa = {kappa1:.3f}")

    # Target QPE encoding: lambda=1 -> tilde=2, lambda=2 -> tilde=4 with nc=3.
    # Solve 2^nc * lambda * t / (2*pi) = integer  =>  t = pi/2.
    t1 = np.pi / 2.0
    assert abs(8 * 1.0 * t1 / (2 * np.pi) - 2.0) < 1e-10, "Phase mapping failed for lambda_1"
    assert abs(8 * 2.0 * t1 / (2 * np.pi) - 4.0) < 1e-10, "Phase mapping failed for lambda_2"

    tilde_lambdas_1 = [2, 4]
    C1 = 1.9  # C must satisfy C <= min(tilde_lambdas) = 2; 1.9 leaves headroom.
    assert C1 <= min(tilde_lambdas_1), (
        f"C constraint violated: {C1} > {min(tilde_lambdas_1)}"
    )

    print(f"  Time parameter        t = pi/2  ({t1:.6f})")
    print(f"  tilde_lambdas         = {tilde_lambdas_1}  (both fit in nc=3 register [1..7])")
    print(f"  C                     = {C1}   (must satisfy C <= tilde_lambda_min = "
          f"{min(tilde_lambdas_1)})")

    qc1 = build_hhl_circuit(A1, b1, t1, nc, C1, tilde_lambdas_1)
    x_q1, p1 = simulate_and_extract(qc1, nc)

    # Make the global phase deterministic: align the first nonzero entry to be
    # real and positive. Fidelity is invariant, but the printed vector reads
    # more cleanly.
    phase = np.exp(-1j * np.angle(x_q1[0])) if abs(x_q1[0]) > 1e-12 else 1.0
    x_q1_disp = x_q1 * phase

    x_c1 = np.linalg.solve(A1, b1)
    x_c1_hat = x_c1 / np.linalg.norm(x_c1)
    F1 = fidelity(x_q1, x_c1)

    print(f"\n  x_classical (raw)     = {x_c1}")
    print(f"  x_classical (norm.)   = {x_c1_hat}")
    print(f"  x_quantum   (norm.)   = {x_q1_disp}")
    print(f"  Fidelity              F = {F1:.6f}")
    print(f"  P(|1>_anc, clock=0)   = {p1:.6f}")

    metrics1 = transpile_metrics(qc1)
    print_metrics_table("Experiment 1", metrics1)

    return {"p": p1, "F": F1, "metrics": metrics1, "tilde_lambdas": tilde_lambdas_1}


# ===================================================
# SECTION 5: EXPERIMENT 2 -- ILL-CONDITIONED CASE
# ===================================================

def run_experiment_2() -> dict:
    """Ill-conditioned 2x2 system (kappa = 200); the large eigenvalue overflows."""
    print("\n" + "=" * 71)
    print("EXPERIMENT 2 -- ILL-CONDITIONED CASE  (kappa = 200, QPE overflow)")
    print("=" * 71)

    nc = 3

    # Off-diagonal sign note: with A2 = [[10.05, 9.95], [9.95, 10.05]] the
    # (1,1)/sqrt(2) direction is the eigenvector of the LARGE eigenvalue (20),
    # so b would be orthogonal to the small-eigenvalue eigenspace and the
    # surviving rotation acts on zero amplitude (P_success = 0). Flipping the
    # off-diagonal sign keeps the same eigenvalues {0.1, 20} and kappa = 200,
    # but now (1,1)/sqrt(2) is the SMALL-eigenvalue eigenvector, preserving the
    # intended pedagogy: the small-eigenvalue branch is rotated successfully
    # while the large-eigenvalue branch overflows.
    A2 = np.array([[10.05, -9.95],
                   [-9.95, 10.05]])
    b2 = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0])

    eigs2 = np.linalg.eigvalsh(A2)
    kappa2 = eigs2.max() / eigs2.min()
    print(f"  Eigenvalues of A2     = {eigs2}")
    print(f"  Condition number     kappa = {kappa2:.3f}")

    # Choose t2 so the small eigenvalue maps to tilde_lambda_1 = 1:
    #   tilde_lambda_1 = 2^nc * lambda_1 * t2 / (2*pi) = 1
    #   => t2 = (2*pi) / (8 * 0.1) = 5*pi/2
    t2 = 5.0 * np.pi / 2.0

    tilde_1_raw = 8 * 0.1 * t2 / (2 * np.pi)
    tilde_2_raw = 8 * 20.0 * t2 / (2 * np.pi)
    tilde_1_aliased = int(round(tilde_1_raw)) % (2 ** nc)   # = 1
    tilde_2_aliased = int(round(tilde_2_raw)) % (2 ** nc)   # = 0  <- overflow!

    assert abs(tilde_1_raw - 1.0) < 1e-10, "Small-eigenvalue QPE mapping failed"
    assert tilde_2_aliased == 0, "Large eigenvalue should alias to 0 (overflow)"

    print(f"\n  Time parameter        t = 5*pi/2  ({t2:.6f})")
    print(f"  tilde_lambda_1 (raw)  = {tilde_1_raw:.4f}  -> aliased = {tilde_1_aliased}")
    print(f"  tilde_lambda_2 (raw)  = {tilde_2_raw:.1f}  -> aliased = {tilde_2_aliased}")
    print(f"  *** lambda_2 aliases to zero: its component is NOT rotated. ***")

    tilde_lambdas_2 = [1]  # only the small-eigenvalue branch survives
    C2 = 0.9
    assert C2 <= min(tilde_lambdas_2), (
        f"C constraint violated: {C2} > {min(tilde_lambdas_2)}"
    )
    print(f"  C = {C2}  (must satisfy C <= tilde_lambda_min = {min(tilde_lambdas_2)})")

    # Theoretical raw success probability (without amplitude amplification).
    P_theory = C2 ** 2 / min(tilde_lambdas_2) ** 2
    print(f"  P_theory = C^2 / tilde_lambda_min^2 = {C2}^2 / {min(tilde_lambdas_2)}^2 "
          f"= {P_theory:.4f}")
    print("  (P_theory is the per-eigencomponent rotation factor; the actual ")
    print("   measured P also carries the |c_j|^2 weight of the surviving ")
    print("   eigencomponent in the expansion of b.)")

    qc2 = build_hhl_circuit(A2, b2, t2, nc, C2, tilde_lambdas_2)
    x_q2, p2 = simulate_and_extract(qc2, nc)

    phase = np.exp(-1j * np.angle(x_q2[0])) if abs(x_q2[0]) > 1e-12 else 1.0
    x_q2_disp = x_q2 * phase

    x_c2 = np.linalg.solve(A2, b2)
    x_c2_hat = x_c2 / np.linalg.norm(x_c2)
    F2 = fidelity(x_q2, x_c2)

    print(f"\n  x_classical (raw)     = {x_c2}")
    print(f"  x_classical (norm.)   = {x_c2_hat}")
    print(f"  x_quantum   (norm.)   = {x_q2_disp}")
    print(f"  Fidelity              F = {F2:.6f}")
    print(f"  P(|1>_anc, clock=0)   = {p2:.6f}")

    overflow_rows = [
        ["lambda_1", f"{eigs2[0]:.4f}", f"{tilde_1_raw:.4f}", tilde_1_aliased, "kept"],
        ["lambda_2", f"{eigs2[1]:.4f}", f"{tilde_2_raw:.4f}", tilde_2_aliased,
         "DROPPED (aliases to 0)"],
    ]
    print("\n--- QPE overflow analysis (nc = 3, max representable integer = 7) ---")
    print(tabulate(overflow_rows,
                   headers=["Eigenvalue", "lambda", "tilde_lambda (raw)",
                            "aliased", "status"],
                   tablefmt="github"))

    metrics2 = transpile_metrics(qc2)
    print_metrics_table("Experiment 2", metrics2)

    return {"p": p2, "F": F2, "metrics": metrics2, "C": C2}


# ===================================================
# SECTION 6: COMPARATIVE ANALYSIS & SCIENTIFIC SUMMARY
# ===================================================

def print_comparative_analysis(r1: dict, r2: dict) -> None:
    """Print the Experiment 1 vs Experiment 2 comparison and interpretation."""
    p1, F1, metrics1 = r1["p"], r1["F"], r1["metrics"]
    p2, F2, metrics2 = r2["p"], r2["F"], r2["metrics"]

    print("\n" + "=" * 71)
    print("COMPARATIVE ANALYSIS")
    print("=" * 71)

    print("\n=== ANCILLA SUCCESS PROBABILITY COMPARISON ===")
    print(f"Experiment 1  |  kappa =   2  |  P(|1>, clock=0) = {p1:.4f}  |  "
          f"{prob_bar(p1)} {p1 * 100:5.1f}%")
    print(f"Experiment 2  |  kappa = 200  |  P(|1>, clock=0) = {p2:.4f}  |  "
          f"{prob_bar(p2)} {p2 * 100:5.1f}%")
    print()
    print("Note: In Experiment 2, P(|1>) is NOT suppressed by kappa^2 -- it is")
    print("numerically high because C/tilde_lambda_min = C/1 ~ 1. The failure")
    print("mode is QPE *register overflow*, not ancilla probability suppression.")
    print("The lambda_2 eigencomponent is simply absent from the answer.")

    print("\n=== SOLUTION FIDELITY COMPARISON ===")
    print(f"Experiment 1  |  kappa =   2  |  Fidelity = {F1:.4f}  "
          f"(correct solution recovered)")
    print(f"Experiment 2  |  kappa = 200  |  Fidelity = {F2:.4f}  "
          f"(lambda_2 lost to QPE overflow; F=1 because b aligns with surviving eigenspace)")
    ratio = F1 / F2 if F2 > 1e-12 else float("inf")
    print(f"\nFidelity degradation factor: Exp1_fidelity / Exp2_fidelity = {ratio:.3f}x")
    print("Root cause: nc=3 clock qubits can represent at most 7 nonzero integers;")
    print(f"            kappa=200 requires nc >= ceil(log2(200)) = "
          f"{int(np.ceil(np.log2(200)))} clock qubits for faithful QPE.")

    print("""
INTERPRETATION NOTE on F2 = {f2:.4f}:
With b = (1,1)/sqrt(2) and A2 symmetric with eigenvectors aligned to that
direction, b lies *entirely* in the surviving (lambda_1 = 0.1) eigenspace.
The lost lambda_2 component has zero amplitude in b to begin with, so the
overflow does not corrupt the *direction* of the recovered solution -- it
corrupts the *normalization / scale*, which fidelity (cosine-angle) does
not see. The real damage of QPE overflow shows up in:
  (i)  the ancilla success probability budget (here P = C^2, not the full
       sum over rotated eigencomponents),
  (ii) the un-normalized magnitude of x_quantum vs. x_classical,
  (iii) fidelity, but only for inputs b that have nontrivial overlap with
        *both* eigenspaces, which a 2x2 system with eigenvectors (1,+/-1)
        cannot provide for b = (1,1)/sqrt(2). The same algorithm with
        nc=3 and kappa=200 applied to a larger matrix -- or to a b not
        aligned with a single eigenvector -- would exhibit the full
        fidelity collapse predicted by theory.
""".format(f2=F2))

    print("\n--- Combined transpilation comparison ---")
    combined_rows = [
        ["Depth (Level 0)", metrics1[0]["depth"], metrics2[0]["depth"]],
        ["Depth (Level 3)", metrics1[3]["depth"], metrics2[3]["depth"]],
        ["CX    (Level 0)", metrics1[0]["cx"], metrics2[0]["cx"]],
        ["CX    (Level 3)", metrics1[3]["cx"], metrics2[3]["cx"]],
        ["Total (Level 0)", metrics1[0]["gate_count"], metrics2[0]["gate_count"]],
        ["Total (Level 3)", metrics1[3]["gate_count"], metrics2[3]["gate_count"]],
    ]
    print(tabulate(combined_rows,
                   headers=["Metric", "Experiment 1", "Experiment 2"],
                   tablefmt="github"))


def print_scientific_summary(r1: dict, r2: dict) -> None:
    """Print the multiline scientific summary that ties the experiments together."""
    tl1 = r1["tilde_lambdas"]
    p1, f1, metrics1 = r1["p"], r1["F"], r1["metrics"]
    p2, f2, metrics2 = r2["p"], r2["F"], r2["metrics"]
    C2 = r2["C"]

    d1_l0, d1_l3 = metrics1[0]["depth"], metrics1[3]["depth"]
    d2_l0, d2_l3 = metrics2[0]["depth"], metrics2[3]["depth"]
    cx1_l0, cx1_l3 = metrics1[0]["cx"], metrics1[3]["cx"]
    cx2_l0, cx2_l3 = metrics2[0]["cx"], metrics2[3]["cx"]

    summary = f"""
=======================================================================
HHL ALGORITHM -- SCIENTIFIC SUMMARY
=======================================================================

1. WHY EXPERIMENT 1 IS EFFICIENT
   Condition number kappa = 2. Time parameter t = pi/2.
   QPE integers: tilde_lambda_1 = {tl1[0]}, tilde_lambda_2 = {tl1[1]} -- both fit within
   the 3-qubit register [1..7] with no aliasing. Ancilla rotations for
   both eigenvalues are valid. Success probability:
       P(|1>, clock=0) = {p1:.4f}.
   Fidelity:  F = {f1:.6f}.

2. WHY EXPERIMENT 2 FAILS TO BE EFFICIENT
   Condition number kappa = 200. With nc=3 clock qubits (max representable
   integer = 7), tilde_lambda_2 = 200 aliases to 0 mod 8. Only
   tilde_lambda_1 = 1 contributes to the ancilla rotation -- the large-
   eigenvalue component is discarded.
   C constraint: C <= 1. P_theory = C^2 / tilde_lambda_min^2 = {C2 ** 2:.4f}.
   Actual P(|1>, clock=0) = {p2:.4f}. Fidelity: F = {f2:.6f}.

   Key distinction: the failure is NOT kappa^2-suppression of the ancilla
   probability (P remains numerically high because C/tilde_lambda_min ~ 1).
   The failure is QPE register overflow: the lambda_2 eigencomponent is
   simply absent from the solution. For kappa=200, faithful QPE requires
   nc >= ceil(log2(200)) = 8 clock qubits.

3. NISQ HARDWARE IMPLICATIONS
   Experiment 1 -- Depth (L0/L3): {d1_l0}/{d1_l3}. CX count (L0/L3): {cx1_l0}/{cx1_l3}.
   Experiment 2 -- Depth (L0/L3): {d2_l0}/{d2_l3}. CX count (L0/L3): {cx2_l0}/{cx2_l3}.
   For an NxN system, the full HHL circuit depth scales as
   O(log^2 N * kappa * polylog(1/epsilon)).  Even the small demonstration
   here requires tens of CX gates, whose error rate on current hardware
   (~0.1-1%) would overwhelm the signal. HHL's exponential speedup over
   classical solvers is asymptotic: it becomes advantageous only for
   N >> 10^6 with fault-tolerant hardware.

=======================================================================
"""
    print(summary)


# ===================================================
# SECTION 7: EXPERIMENT 3 -- DRAMATIC QPE OVERFLOW
#   Two sub-experiments on the same matrix A3 = diag(2, 3):
#     3a  nc=4, t=pi/2  -> no overflow, F ~ 1.000  (control)
#     3b  nc=3, t=pi    -> lambda_1 overflows, F ~ 0.308  (failure)
#
#   Why this is more striking than Experiment 2:
#   * A3 is diagonal with eigenvectors e1=(1,0), e2=(0,1).
#   * b=(1,1)/sqrt(2) has EQUAL weight on both eigenvectors.
#   * The classical solution x_c ~ (3,2)/sqrt(13) sits diagonally.
#   * When lambda_1 overflows (alias->0), only e2=(0,1) survives, so x_q
#     points along (0,1) -- nearly PERPENDICULAR to x_c.
#   * Fidelity: 1.000 -> 0.308, a 69% drop that is immediately visible.
# ===================================================

def run_experiment_3() -> dict:
    """Diagonal system diag(2, 3): overflow steers the answer onto the wrong axis."""
    print("\n" + "=" * 71)
    print("EXPERIMENT 3 -- DRAMATIC QPE OVERFLOW: DIRECTION MISMATCH (kappa=1.5)")
    print("=" * 71)

    A3 = np.array([[2.0, 0.0],
                   [0.0, 3.0]])
    b3 = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0])   # same H|0> prep as Exp 1

    eigs3 = np.linalg.eigvalsh(A3)
    kappa3 = eigs3.max() / eigs3.min()
    x_c3 = np.linalg.solve(A3, b3)
    x_c3_hat = x_c3 / np.linalg.norm(x_c3)

    print(f"  A3 = diag(2, 3)   eigenvalues = {eigs3}   kappa = {kappa3:.3f}")
    print(f"  b  = (1,1)/sqrt(2)  [equal weight on both eigenvectors]")
    print(f"  x_classical (normalized) = {x_c3_hat}")
    print(f"  Eigenvectors: u1=(1,0) for lambda=2,  u2=(0,1) for lambda=3")

    # --- Sub-experiment 3a: nc=4, t=pi/2 -- both eigenvalues fit, no overflow ---
    print("\n--- Sub-experiment 3a: nc=4, t=pi/2 (CONTROL -- no overflow) ---")

    nc3a = 4
    t3a = np.pi / 2.0

    tl1_3a_raw = (2 ** nc3a) * eigs3[0] * t3a / (2 * np.pi)  # = 8
    tl2_3a_raw = (2 ** nc3a) * eigs3[1] * t3a / (2 * np.pi)  # = 12

    assert abs(tl1_3a_raw - 8.0) < 1e-10, "3a: tilde_lambda_1 mapping failed"
    assert abs(tl2_3a_raw - 12.0) < 1e-10, "3a: tilde_lambda_2 mapping failed"
    assert tl1_3a_raw <= (2 ** nc3a - 1), "3a: lambda_1 overflows -- check nc/t"
    assert tl2_3a_raw <= (2 ** nc3a - 1), "3a: lambda_2 overflows -- check nc/t"

    tilde_lambdas_3a = [int(round(tl1_3a_raw)), int(round(tl2_3a_raw))]  # [8, 12]
    C3a = 7.6   # C <= min(tilde_lambdas_3a) = 8

    print(f"  tilde_lambda_1 = {tl1_3a_raw:.1f}  tilde_lambda_2 = {tl2_3a_raw:.1f}  "
          f"(both fit in nc=4 register [1..15])")
    print(f"  C = {C3a}  (must satisfy C <= {min(tilde_lambdas_3a)})")

    qc3a = build_hhl_circuit(A3, b3, t3a, nc3a, C3a, tilde_lambdas_3a)
    x_q3a, p3a = simulate_and_extract(qc3a, nc3a)
    phase3a = np.exp(-1j * np.angle(x_q3a[0])) if abs(x_q3a[0]) > 1e-12 else 1.0
    x_q3a_disp = (x_q3a * phase3a).real
    F3a = fidelity(x_q3a, x_c3)

    print(f"\n  x_quantum   (normalized) = {x_q3a_disp}")
    print(f"  x_classical (normalized) = {x_c3_hat}")
    print(f"  Fidelity                 F = {F3a:.6f}")
    print(f"  P(|1>_anc, clock=0)      = {p3a:.6f}  {prob_bar(p3a)}")

    # --- Sub-experiment 3b: nc=3, t=pi -- lambda_1 overflows! ---
    print("\n--- Sub-experiment 3b: nc=3, t=pi (FAILURE -- lambda_1 overflows) ---")

    nc3b = 3
    t3b = np.pi

    tl1_3b_raw = (2 ** nc3b) * eigs3[0] * t3b / (2 * np.pi)   # = 8
    tl2_3b_raw = (2 ** nc3b) * eigs3[1] * t3b / (2 * np.pi)   # = 12
    tl1_3b_aliased = int(round(tl1_3b_raw)) % (2 ** nc3b)      # = 0  <- overflow!
    tl2_3b_aliased = int(round(tl2_3b_raw)) % (2 ** nc3b)      # = 4  <- OK

    assert abs(tl1_3b_raw - 8.0) < 1e-10, "3b: tilde_lambda_1 mapping failed"
    assert abs(tl2_3b_raw - 12.0) < 1e-10, "3b: tilde_lambda_2 mapping failed"
    assert tl1_3b_aliased == 0, "lambda_1 should alias to 0 (overflow)"
    assert tl2_3b_aliased == 4, "lambda_2 should alias to 4"

    print(f"  tilde_lambda_1 (raw) = {tl1_3b_raw:.1f}  -> aliased = {tl1_3b_aliased}"
          f"  *** OVERFLOW: lambda_1 component LOST ***")
    print(f"  tilde_lambda_2 (raw) = {tl2_3b_raw:.1f}  -> aliased = {tl2_3b_aliased}  (OK)")

    tilde_lambdas_3b = [tl2_3b_aliased]   # [4] -- only lambda_2 survives
    C3b = 3.8   # C <= min(tilde_lambdas_3b) = 4
    print(f"  C = {C3b}  (only lambda_2 branch is rotated)")

    qc3b = build_hhl_circuit(A3, b3, t3b, nc3b, C3b, tilde_lambdas_3b)
    x_q3b, p3b = simulate_and_extract(qc3b, nc3b)
    phase3b = np.exp(-1j * np.angle(x_q3b[0])) if abs(x_q3b[0]) > 1e-12 else 1.0
    x_q3b_disp = (x_q3b * phase3b).real
    F3b = fidelity(x_q3b, x_c3)

    print(f"\n  x_quantum   (normalized) = {x_q3b_disp}  <- points along u2=(0,1) only!")
    print(f"  x_classical (normalized) = {x_c3_hat}")
    print(f"  Fidelity                 F = {F3b:.6f}  <-- DRAMATIC DROP")
    print(f"  P(|1>_anc, clock=0)      = {p3b:.6f}  {prob_bar(p3b)}")

    # --- Side-by-side comparison ---
    print("\n" + "=" * 71)
    print("EXPERIMENT 3 -- SIDE-BY-SIDE COMPARISON")
    print("=" * 71)

    comparison_rows = [
        ["Matrix A", "diag(2, 3)", "diag(2, 3)"],
        ["b", "(1,1)/sqrt(2)", "(1,1)/sqrt(2)"],
        ["Clock qubits nc", f"{nc3a}", f"{nc3b}"],
        ["Time t", "pi/2", "pi"],
        ["tilde_lambda_1", f"{tl1_3a_raw:.0f} (fits)", f"{tl1_3b_raw:.0f} mod 8 = 0  OVERFLOW"],
        ["tilde_lambda_2", f"{tl2_3a_raw:.0f} (fits)", f"{tl2_3b_raw:.0f} mod 8 = {tl2_3b_aliased}  OK"],
        ["x_quantum", f"{x_q3a_disp}", f"{x_q3b_disp}"],
        ["x_classical", f"{x_c3_hat}", f"{x_c3_hat}"],
        ["Fidelity F", f"{F3a:.4f}", f"{F3b:.4f}  (-{(1 - F3b) * 100:.0f}%)"],
        ["P(|1>, clock=0)", f"{p3a:.4f}", f"{p3b:.4f}"],
    ]
    print(tabulate(comparison_rows,
                   headers=["Metric", "3a: nc=4 CORRECT", "3b: nc=3 OVERFLOW"],
                   tablefmt="github"))

    print(f"""
KEY INSIGHT:
  Both experiments use the SAME matrix A3 and the SAME input b.
  The only difference is one extra clock qubit (nc=4 vs nc=3).

  With nc=3, lambda_1=2 maps to tilde=8 which OVERFLOWS the 3-bit
  register (max=7), aliasing to 0. The lambda_1 eigenvector u1=(1,0)
  is silently dropped from the rotation.

  Result: x_quantum points along u2=(0,1), while the correct answer
  lies along (3,2)/sqrt(13) ~= (0.83, 0.55). The solution is nearly
  PERPENDICULAR to the true answer -- Fidelity collapses from 1.000
  to {F3b:.3f} ({(1 - F3b) * 100:.0f}% drop).

  Unlike Experiment 2, this failure is IMMEDIATELY VISIBLE in the
  fidelity metric -- no subtle diagnostics needed.
  Rule: nc >= ceil(log2(kappa * lambda_max / lambda_min)) for faithful QPE.
""")

    return {"F_control": F3a, "F_overflow": F3b}


def main() -> None:
    """Run the three narrative experiments and print the scientific summary."""
    # Make Unicode output (em-dashes etc.) render cleanly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    r1 = run_experiment_1()
    r2 = run_experiment_2()
    print_comparative_analysis(r1, r2)
    print_scientific_summary(r1, r2)
    run_experiment_3()


if __name__ == "__main__":
    main()
