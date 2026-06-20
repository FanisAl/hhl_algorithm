"""End-to-end and unit tests for the HHL quantum linear solver.

The headline test (``test_hhl_matches_classical_solution``) runs the full HHL
circuit on a small, well-conditioned system and checks that the quantum
solution agrees -- up to normalization and global phase -- with the classical
``numpy.linalg.solve`` answer.
"""

import numpy as np
import pytest

import hhl_algorithm as hhl


# A well-conditioned 2x2 system whose eigenvalues (1, 2) map exactly onto the
# 3-qubit clock register at t = pi/2, so HHL recovers the solution faithfully.
A_WELL = np.array([[1.5, 0.5],
                   [0.5, 1.5]])
B_VEC = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0])
NC = 3
T = np.pi / 2.0
TILDE_LAMBDAS = [2, 4]
C = 1.9


def _solve_hhl(A, b, t, nc, c, tilde_lambdas):
    """Build, simulate, and return the normalized HHL solution vector."""
    qc = hhl.build_hhl_circuit(A, b, t, nc, c, tilde_lambdas)
    x_quantum, _ = hhl.simulate_and_extract(qc, nc)
    return x_quantum


def _align_phase(vec):
    """Rotate ``vec`` so its first entry is real and non-negative."""
    if abs(vec[0]) > 1e-12:
        vec = vec * np.exp(-1j * np.angle(vec[0]))
    return np.real(vec)


def test_hhl_matches_classical_solution():
    """HHL solution direction agrees with numpy.linalg.solve on a known system."""
    x_classical = np.linalg.solve(A_WELL, B_VEC)
    x_classical_hat = x_classical / np.linalg.norm(x_classical)

    x_quantum = _solve_hhl(A_WELL, B_VEC, T, NC, C, TILDE_LAMBDAS)

    # Compare normalized vectors with the global phase fixed.
    x_quantum_real = _align_phase(x_quantum)
    np.testing.assert_allclose(x_quantum_real, x_classical_hat, atol=1e-6)


def test_hhl_fidelity_is_near_unity():
    """Fidelity against the classical solution is ~1 for the well-conditioned case."""
    x_classical = np.linalg.solve(A_WELL, B_VEC)
    x_quantum = _solve_hhl(A_WELL, B_VEC, T, NC, C, TILDE_LAMBDAS)
    assert hhl.fidelity(x_quantum, x_classical) == pytest.approx(1.0, abs=1e-6)


def test_quantum_solution_is_normalized():
    """The extracted solution vector has unit norm."""
    x_quantum = _solve_hhl(A_WELL, B_VEC, T, NC, C, TILDE_LAMBDAS)
    assert np.linalg.norm(x_quantum) == pytest.approx(1.0, abs=1e-9)


def test_circuit_register_size():
    """The circuit uses nc clock qubits plus a system and an ancilla qubit."""
    qc = hhl.build_hhl_circuit(A_WELL, B_VEC, T, NC, C, TILDE_LAMBDAS)
    assert qc.num_qubits == NC + 2


def test_build_rejects_unsupported_b():
    """A right-hand side other than (1/sqrt(2))(1,1)^T is rejected clearly."""
    bad_b = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="only supports"):
        hhl.build_hhl_circuit(A_WELL, bad_b, T, NC, C, TILDE_LAMBDAS)


def test_ancilla_rotation_rejects_zero_eigenvalue():
    """A tilde_lambda that aliased to 0 must be rejected, not silently rotated."""
    qc = hhl.build_hhl_circuit(A_WELL, B_VEC, T, NC, C, TILDE_LAMBDAS)
    with pytest.raises(ValueError, match="aliases to zero"):
        hhl.apply_ancilla_rotation(qc, 0, [2, 3, 4], tilde_lambda=0, C=C, nc=NC)


def test_fidelity_is_phase_and_scale_invariant():
    """Fidelity ignores global phase and overall scale of the classical vector."""
    x_quantum = np.array([0.6, 0.8], dtype=complex)
    x_classical = np.array([0.6, 0.8])
    base = hhl.fidelity(x_quantum, x_classical)
    scaled = hhl.fidelity(x_quantum, 5.0 * x_classical)
    phased = hhl.fidelity(np.exp(1j * 0.7) * x_quantum, x_classical)
    assert base == pytest.approx(1.0, abs=1e-12)
    assert scaled == pytest.approx(base, abs=1e-12)
    assert phased == pytest.approx(base, abs=1e-12)
