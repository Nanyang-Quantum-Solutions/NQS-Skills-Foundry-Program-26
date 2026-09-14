"""
Fidelity witnesses for pure fermionic Gaussian states  [arXiv:1703.03152]
Applied to the transverse-field XY / Ising chain, Eq. (16).

Conventions (all from the paper):
  Jordan-Wigner, Eq. (15):
      m_{2k-1} = (prod_{j<k} Z_j) X_k
      m_{2k}   = (prod_{j<k} Z_j) Y_k
  Hamiltonian, Eq. (18):
      H(A) = (i/4) sum_{j,k} A_{jk} m_j m_k ,   A = -A^T real
  Propagator, Eq. (19):
      m_j(t) = e^{itH} m_j e^{-itH} = sum_k Q_{jk}(t) m_k ,   Q(t) = exp(tA)
  Covariance matrix, Eq. (4):
      M_{jk}(rho) = (i/2) tr([m_j,m_k] rho)   ->   i <m_j m_k>  for j != k
  Transformation, Eq. (54):
      M(U rho U^dag) = Q M(rho) Q^T

Majorana indices are 1..2L in the paper; arrays here are 0-indexed.
"""

import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------------------
# 1.  Coupling matrix  A  for  H_spin  (Eq. 16)
# ----------------------------------------------------------------------
def coupling_matrix(L, Jx=0.0, Jy=0.0, B=0.0):
    """
    Antisymmetric 2L x 2L coupling matrix A of

        H = -sum_k (Jx_k X_k X_{k+1} + Jy_k Y_k Y_{k+1}) - sum_k B_k Z_k

    Nonzero entries (paper's 1-based Majorana indices):
        A[2k-1, 2k]   = +2 B_k     k = 1..L      
        A[2k,   2k+1] = +2 Jx_k    k = 1..L-1    
        A[2k-1, 2k+2] = -2 Jy_k    k = 1..L-1    

    Jx, Jy, B may be scalars or length-(L-1)/(L-1)/L arrays.
    """
    Jx = np.broadcast_to(np.atleast_1d(np.asarray(Jx, float)), (L - 1,))
    Jy = np.broadcast_to(np.atleast_1d(np.asarray(Jy, float)), (L - 1,))
    B = np.broadcast_to(np.atleast_1d(np.asarray(B, float)), (L,))

    A = np.zeros((2 * L, 2 * L))

    # field:  (2k-1, 2k) -> 0-based (2k-2, 2k-1)
    for k in range(1, L + 1):
        A[2 * k - 2, 2 * k - 1] = 2.0 * B[k - 1]

    # XX:     (2k, 2k+1) -> 0-based (2k-1, 2k)
    for k in range(1, L):
        A[2 * k - 1, 2 * k] = 2.0 * Jx[k - 1]

    # YY:     (2k-1, 2k+2) -> 0-based (2k-2, 2k+1)
    for k in range(1, L):
        A[2 * k - 2, 2 * k + 1] = -2.0 * Jy[k - 1]

    return A - A.T          # impose antisymmetry


# ----------------------------------------------------------------------
# 2.  Reference covariance matrix  (Eq. 17 / Eq. 53)
# ----------------------------------------------------------------------
def M_fock(L, omega=None):
    """
    Covariance matrix of the Fock state |omega>, Eq. (53):
        M_omega = direct_sum_k (1 - 2 omega_k) [[0,-1],[1,0]]
    omega=None  ->  omega = 0  ->  |up>^{otimes L}, which is Eq. (17).
    """
    omega = np.zeros(L, int) if omega is None else np.asarray(omega, int)
    M = np.zeros((2 * L, 2 * L))
    for k in range(L):
        s = 1.0 - 2.0 * omega[k]
        M[2 * k, 2 * k + 1] = -s
        M[2 * k + 1, 2 * k] = +s
    return M


# ----------------------------------------------------------------------
# 3.  Propagators
# ----------------------------------------------------------------------
def propagator(A, t):
    """Q(t) = exp(t A) in SO(2L), Eq. (19)."""
    return expm(t * A)


def propagator_pairwise(A, t):
    """
    Closed-form exp(tA) when A is a *disjoint* set of 2x2 antisymmetric
    blocks (true for A(B) and, in the Ising case Jy=0, for A(J)).
    Each pair (a,b) with A[a,b]=w rotates by angle w*t:
        [[cos, sin], [-sin, cos]]
    Falls back to expm if the pairing assumption is violated.
    """
    n = A.shape[0]
    Q = np.eye(n)
    used = np.zeros(n, bool)
    for a in range(n):
        if used[a]:
            continue
        nz = np.flatnonzero(np.abs(A[a]) > 1e-14)
        nz = nz[~used[nz]]
        if len(nz) == 0:
            used[a] = True
            continue
        if len(nz) > 1:
            return expm(t * A)          # not a disjoint pairing
        b = nz[0]
        if np.count_nonzero(np.abs(A[b]) > 1e-14) > 1:
            return expm(t * A)
        th = A[a, b] * t
        c, s = np.cos(th), np.sin(th)
        Q[a, a] = c; Q[a, b] = s
        Q[b, a] = -s; Q[b, b] = c
        used[a] = used[b] = True
    return Q


def trotter_propagator(A_J, A_B, t, T, order="JB"):
    """
    Mode-space propagator of the digital simulation with T first-order
    Trotter-Suzuki steps,  dt = t/T.

    order="JB":  Q_T = (exp(dt A_J) exp(dt A_B))^T   <- paper's written form
    order="BJ":  Q_T = (exp(dt A_B) exp(dt A_J))^T

    See the note in the accompanying discussion: the two orderings are both
    valid first-order Trotterizations and agree to O(dt^2); which one matches
    U_T = (e^{-i dt H_B} e^{-i dt H_J})^T depends on the composition
    convention for Q.  Both are provided so the choice is explicit.
    """
    dt = t / T
    QJ = propagator_pairwise(A_J, dt)
    QB = propagator_pairwise(A_B, dt)
    step = QJ @ QB if order == "JB" else QB @ QJ
    return np.linalg.matrix_power(step, T)


# ----------------------------------------------------------------------
# 4.  Covariance matrices and the witness
# ----------------------------------------------------------------------
def evolve_M(M0, Q):
    """M -> Q M Q^T,  Eq. (54)."""
    return Q @ M0 @ Q.T


def fidelity_witness(M_p, M_t):
    """F_W = 1 + (1/4) tr[(M_p - M_t)^T M_t],  Eq. (8)."""
    return 1.0 + 0.25 * np.trace((M_p - M_t).T @ M_t)


def fidelity_witness_v2(M_p, M_t):
    """Equivalent form, Eq. (62):  F_W = 1 - L/2 - (1/4) tr[M_p M_t]."""
    L = M_p.shape[0] // 2
    return 1.0 - L / 2.0 - 0.25 * np.trace(M_p @ M_t)


def gaussian_fidelity(M_t, M_p):
    """
    Exact |<psi_t|psi_p>|^2 for two *pure* Gaussian states, from their
    covariance matrices alone:

        F = |det( (M_t + M_p) / 2 )|^{1/2}

    Sanity: M_p = M_t gives |det M_t|^{1/2} = |Pf(M_t)| = 1; M_p = -M_t gives 0.
    Only valid when both states are pure (M^2 = -1); for a mixed or non-Gaussian
    preparation there is no such shortcut and F must come from the full state.
    """
    return float(np.sqrt(abs(np.linalg.det((M_t + M_p) / 2.0))))


def l1_norm(M_t):
    """|M(rho_t)| = sum_{(j,k) in Omega} |M_jk|, strict upper triangle, Eq. (10)."""
    return np.abs(np.triu(M_t, 1)).sum()


def omega_set_size(M_t, tol=1e-12):
    """|Omega| = number of nonzero entries with 1 <= j < k <= 2L."""
    return int((np.abs(np.triu(M_t, 1)) > tol).sum())


# ----------------------------------------------------------------------
# 5.  Wick / Pfaffian:  <prod_k Z_k> on a Gaussian state
# ----------------------------------------------------------------------
def pfaffian(A):
    """
    Pf(A) for a real antisymmetric matrix, by Parlett-Reid tridiagonalization
    with partial pivoting (Wimmer, Alg. 923 -- the PFAPACK routine the paper
    cites as [67]).  Zero for odd dimension.  This is the object Wick's
    theorem gives, and unlike sqrt(|det A|) it carries the sign.
    """
    A = np.array(A, float)              # copy; the algorithm works in place
    n = A.shape[0]
    if n % 2 == 1:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        # pivot the largest entry of column k onto the superdiagonal
        kp = k + 1 + int(np.abs(A[k + 1:, k]).argmax())
        if kp != k + 1:
            A[[k + 1, kp], k:] = A[[kp, k + 1], k:]
            A[k:, [k + 1, kp]] = A[k:, [kp, k + 1]]
            pf = -pf                    # each transposition flips the sign
        if A[k, k + 1] == 0.0:
            return 0.0
        pf *= A[k, k + 1]
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            col = A[k + 2:, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, col) - np.outer(col, tau)
    return pf


def z_string(M, n):
    """
    |<prod_{k=1}^n Z_k>| on the Gaussian state with covariance matrix M.

    Z_k = -i m_{2k-1} m_{2k}, so the string is a product of the first 2n
    Majoranas and Wick's theorem gives |Pf(M_{1..2n})|  (paper, App. A.c).
    """
    return abs(pfaffian(M[:2 * n, :2 * n]))


# ----------------------------------------------------------------------
# 6.  Diagnostics
# ----------------------------------------------------------------------
def purity_error(M):
    """||M^2 + I||_inf .  Zero iff M is a pure Gaussian covariance matrix."""
    return np.abs(M @ M + np.eye(M.shape[0])).max()


def orthogonality_error(Q):
    """||Q Q^T - I||_inf .  Zero iff Q in O(2L)."""
    return np.abs(Q @ Q.T - np.eye(Q.shape[0])).max()
