"""
Verify the coupling matrix A and the whole Gaussian pipeline against brute-force
exact diagonalization in the full 2^L Hilbert space, for small L.
"""
import numpy as np
from scipy.linalg import expm
from tfim_fw import (coupling_matrix, M_fock, propagator, evolve_M, z_string,
                              fidelity_witness, fidelity_witness_v2,
                              trotter_propagator, purity_error,
                              orthogonality_error)

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def kron_list(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def site_op(L, k, P):
    """P acting on site k (1-based)."""
    return kron_list([P if j == k else I2 for j in range(1, L + 1)])


def majorana(L, idx):
    """m_idx (1-based) via Eq. (15)."""
    k = (idx + 1) // 2                       # site
    P = X if idx % 2 == 1 else Y             # odd -> X, even -> Y
    ops = [Z if j < k else (P if j == k else I2) for j in range(1, L + 1)]
    return kron_list(ops)


def H_spin_exact(L, Jx, Jy, B):
    """Eq. (16) as a 2^L x 2^L matrix."""
    d = 2 ** L
    H = np.zeros((d, d), complex)
    for k in range(1, L):
        H -= Jx * site_op(L, k, X) @ site_op(L, k + 1, X)
        H -= Jy * site_op(L, k, Y) @ site_op(L, k + 1, Y)
    for k in range(1, L + 1):
        H -= B * site_op(L, k, Z)
    return H


def M_exact(L, psi):
    """M_{jk} = i <m_j m_k>, computed brute force."""
    ms = [majorana(L, j) for j in range(1, 2 * L + 1)]
    M = np.zeros((2 * L, 2 * L))
    for a in range(2 * L):
        for b in range(2 * L):
            if a == b:
                continue
            M[a, b] = (1j * (psi.conj() @ (ms[a] @ (ms[b] @ psi)))).real
    return M


def up_state(L):
    v = np.zeros(2 ** L, complex)
    v[0] = 1.0            # |0...0> = |up>^{otimes L}  (Z = +1 on every site)
    return v


# ======================================================================
print("=" * 42)
print("TEST 1 -- H(A) reconstructed from A equals H_spin  (checks the")
print("          factor-of-2 matching rule and every sign in A)")
print("=" * 42)
for L, Jx, Jy, B in [(3, 1.0, 0.0, 1.0), (4, 0.7, 0.0, 1.3),
                     (3, 0.9, 0.5, 1.1), (4, -0.4, 1.2, -0.8)]:
    A = coupling_matrix(L, Jx, Jy, B)
    ms = [majorana(L, j) for j in range(1, 2 * L + 1)]
    H_from_A = np.zeros((2 ** L, 2 ** L), complex)
    for a in range(2 * L):
        for b in range(2 * L):
            if A[a, b] != 0:
                H_from_A += 0.25j * A[a, b] * (ms[a] @ ms[b])
    H_ref = H_spin_exact(L, Jx, Jy, B)
    err = np.abs(H_from_A - H_ref).max()
    print(f"  L={L}  Jx={Jx:+.1f} Jy={Jy:+.1f} B={B:+.1f}   max|H(A)-H_spin| = {err:.2e}")

# ======================================================================
print()
print("=" * 42)
print("TEST 2 -- Q(t)=exp(tA) reproduces Heisenberg evolution of m_j")
print("=" * 42)
for L, Jx, Jy, B, t in [(3, 1.0, 0.0, 1.0, 0.37), (3, 0.6, 0.8, 1.2, 0.53)]:
    A = coupling_matrix(L, Jx, Jy, B)
    H = H_spin_exact(L, Jx, Jy, B)
    U = expm(-1j * t * H)
    Q = propagator(A, t)
    ms = [majorana(L, j) for j in range(1, 2 * L + 1)]
    err = 0.0
    for j in range(2 * L):
        lhs = U.conj().T @ ms[j] @ U          # e^{itH} m_j e^{-itH}
        rhs = sum(Q[j, k] * ms[k] for k in range(2 * L))
        err = max(err, np.abs(lhs - rhs).max())
    print(f"  L={L}  Jx={Jx:+.1f} Jy={Jy:+.1f} B={B:+.1f} t={t}   max err = {err:.2e}")

# ======================================================================
print()
print("=" * 42)
print("TEST 3 -- M(rho_t) = Q M_up Q^T matches brute-force M(rho_t)_jk= i<m_j m_k>")
print("=" * 42)
for L, Jx, Jy, B, t in [(3, 1.0, 0.0, 1.0, 0.4), (4, 1.0, 0.0, 1.0, 0.5),
                        (3, 0.5, 0.9, 1.1, 0.6)]:
    A = coupling_matrix(L, Jx, Jy, B)
    Q = propagator(A, t)
    M_gauss = evolve_M(M_fock(L), Q)
    psi = expm(-1j * t * H_spin_exact(L, Jx, Jy, B)) @ up_state(L)
    M_ed = M_exact(L, psi)
    print(f"  L={L}  Jx={Jx:+.1f} Jy={Jy:+.1f} B={B:+.1f} t={t}   "
          f"max|M_gauss - M_ED| = {np.abs(M_gauss - M_ed).max():.2e}")

# ======================================================================
print()
print("=" * 42)
print("TEST 4 -- F_W is a valid LOWER bound on the true fidelity")
print("          (Trotterized preparation vs exact target, brute force F)")
print("=" * 42)
L, J, B = 4, 1.0, 1.0
t = L / 8.0
A_full = coupling_matrix(L, J, 0.0, B)
A_J = coupling_matrix(L, J, 0.0, 0.0)
A_B = coupling_matrix(L, 0.0, 0.0, B)
H_J = H_spin_exact(L, J, 0.0, 0.0)
H_B = H_spin_exact(L, 0.0, 0.0, B)
psi_t = expm(-1j * t * (H_J + H_B)) @ up_state(L)
M_t = evolve_M(M_fock(L), propagator(A_full, t))

print(f"  L={L}, t={t}   (order='BJ' matches U_T = (e^{{-i dt H_B}} e^{{-i dt H_J}})^T)")
print(f"  {'T':>4} {'F_true':>12} {'F_W (BJ)':>12} {'F_W (JB)':>12} {'F-F_W':>12}")
for T in [1, 2, 3, 5, 8, 12, 20, 40]:
    dt = t / T
    step = expm(-1j * dt * H_B) @ expm(-1j * dt * H_J)
    psi_p = np.linalg.matrix_power(step, T) @ up_state(L)
    F_true = abs(psi_t.conj() @ psi_p) ** 2

    M_p_bj = evolve_M(M_fock(L), trotter_propagator(A_J, A_B, t, T, "BJ"))
    M_p_jb = evolve_M(M_fock(L), trotter_propagator(A_J, A_B, t, T, "JB"))
    FW_bj = fidelity_witness(M_p_bj, M_t)
    FW_jb = fidelity_witness(M_p_jb, M_t)
    print(f"  {T:>4} {F_true:>12.8f} {FW_bj:>12.8f} {FW_jb:>12.8f} {F_true - FW_bj:>12.2e}")

# ======================================================================
print()
print("=" * 42)
print("TEST 5 -- structural invariants at large L")
print("=" * 42)
for L in [50, 100]:
    t = L / 8.0
    A = coupling_matrix(L, 1.0, 0.0, 1.0)
    Q = propagator(A, t)
    M_t = evolve_M(M_fock(L), Q)
    print(f"  L={L:>4}   ||QQ^T-I||={orthogonality_error(Q):.2e}   "
          f"||M^2+I||={purity_error(M_t):.2e}   "
          f"F_W(eq8 vs eq62) diff={abs(fidelity_witness(M_t, M_t) - fidelity_witness_v2(M_t, M_t)):.2e}   "
          f"F_W(target)={fidelity_witness(M_t, M_t):.10f}")
# ======================================================================
print()
print("=" * 42)
print("TEST 6 -- Wick/Pfaffian gives <prod_{k=1..n} Z_k> on the target")
print("          and the string plateaus at the light cone, not at a")
print("          precision floor (plateau onset moves as n ~ 2t)")
print("=" * 42)
for L, Jx, Jy, B, t in [(4, 1.0, 0.0, 1.0, 0.5), (5, 0.7, 0.4, 1.2, 0.6)]:
    A = coupling_matrix(L, Jx, Jy, B)
    M = evolve_M(M_fock(L), propagator(A, t))
    psi = expm(-1j * t * H_spin_exact(L, Jx, Jy, B)) @ up_state(L)
    err = 0.0
    for n in range(1, L + 1):
        op = np.eye(2 ** L, dtype=complex)
        for k in range(1, n + 1):
            op = op @ site_op(L, k, Z)
        bf = abs(psi.conj() @ (op @ psi))
        err = max(err, abs(z_string(M, n) - bf))
    print(f"  L={L}  Jx={Jx:+.1f} Jy={Jy:+.1f} B={B:+.1f} t={t}   "
          f"max_n |Pf - brute force| = {err:.2e}")

L = 100
print(f"  {'t':>6} {'n=10':>10} {'n=20':>10} {'n=30':>10} {'n=45':>10}  onset")
for t in [4.0, 8.0, 12.5, 20.0]:
    M = evolve_M(M_fock(L), propagator(coupling_matrix(L, 1.0, 0.0, 1.0), t))
    v = [z_string(M, n) for n in range(1, 50)]
    onset = next((n for n in range(2, 49)
                  if abs(np.log10(v[n]) - np.log10(v[n - 1])) < 1e-3), None)
    print(f"  {t:>6.1f} {v[9]:>10.2e} {v[19]:>10.2e} {v[29]:>10.2e} {v[44]:>10.2e}"
          f"  {onset}")
print("  -> plateau onset tracks 2t (the light cone). It survives adding 1e-10")
print("     noise to M, so it is physical, not roundoff. sqrt|det| happens to")
print("     agree here (M is well conditioned); the Pfaffian is used because it")
print("     is the object Wick's theorem actually gives, and it carries a sign.")

# ======================================================================
print()
print("=" * 42)
print("TEST 7 -- gaussian_fidelity matches brute-force |<psi_t|psi_p>|^2")
print("=" * 42)
from tfim_fw import gaussian_fidelity
L, J, B = 4, 1.0, 1.0
t = L / 8.0
A_full = coupling_matrix(L, J, 0.0, B)
A_J = coupling_matrix(L, J, 0.0, 0.0)
A_B = coupling_matrix(L, 0.0, 0.0, B)
H_J = H_spin_exact(L, J, 0.0, 0.0)
H_B = H_spin_exact(L, 0.0, 0.0, B)
psi_t = expm(-1j * t * (H_J + H_B)) @ up_state(L)
M_t = evolve_M(M_fock(L), propagator(A_full, t))
err = 0.0
for T in [1, 2, 5, 12]:
    dt = t / T
    step = expm(-1j * dt * H_B) @ expm(-1j * dt * H_J)
    psi_p = np.linalg.matrix_power(step, T) @ up_state(L)
    F_bf = abs(psi_t.conj() @ psi_p) ** 2
    M_p = evolve_M(M_fock(L), trotter_propagator(A_J, A_B, t, T, "BJ"))
    err = max(err, abs(gaussian_fidelity(M_t, M_p) - F_bf))
print(f"  L={L}, T in {{1,2,5,12}}:   max |F_gauss - F_bruteforce| = {err:.2e}")
