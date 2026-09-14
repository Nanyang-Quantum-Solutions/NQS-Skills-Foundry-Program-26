"""
Importance-sampling estimation of the fidelity witness  [arXiv:1703.03152, §8].

The protocol, per shot:

  1. draw a Majorana pair  (j,k)  from   P_jk = |M_jk(rho_t)| / |M(rho_t)|
  2. measure  i m_j m_k  on the preparation, obtaining  beta = +-1
  3. record   X = 2 |M(rho_t)| * beta * sgn(M_jk(rho_t))

Then  E[X] = tr[M(rho_p)^T M(rho_t)]  and  F_W = 1 - L/2 + E[X]/4.

------------------------------------------------------------------------------
The observation this module is built around
------------------------------------------------------------------------------
X takes only the two values +-2|M(rho_t)|.  So X^2 is *deterministic*, the
estimator is a single biased coin, and everything about its statistics is
closed-form:

    p+ = 1/2 + E[X] / (4|M_t|)            probability of the + outcome
    Var[X] = 4|M_t|^2 - E[X]^2            exact, no sampling needed
    F*_W  = 1 - L/2 + |M_t|(2 p_hat - 1)/2        affine in a binomial count


"""
import numpy as np
from scipy.stats import binom

from tfim_fw import l1_norm


# ----------------------------------------------------------------------
# 1.  The sampling distribution over Omega
# ----------------------------------------------------------------------
class Plan:
    """Flattened measurement plan built from the target covariance matrix.

    Holds the strict-upper-triangle support Omega, the sampling probabilities
    P_jk, and the signs of M_t there. This is the object an experimentalist
    would carry to the device: it is a function of the *target* alone.
    """

    __slots__ = ("j", "k", "p", "sign", "norm", "L")

    def __init__(self, M_t, tol=1e-12):
        n = M_t.shape[0]
        self.L = n // 2
        j, k = np.triu_indices(n, 1)
        v = M_t[j, k]
        keep = np.abs(v) > tol
        self.j, self.k, v = j[keep], k[keep], v[keep]
        self.norm = np.abs(v).sum()
        self.p = np.abs(v) / self.norm
        self.sign = np.sign(v)

    def __len__(self):
        return self.j.size


# ----------------------------------------------------------------------
# 2.  Closed-form statistics  (no sampling)
# ----------------------------------------------------------------------
def expected_X(M_p, M_t):
    """E[X] = tr[M_p^T M_t]  -- the quantity the sampler estimates."""
    return float(np.trace(M_p.T @ M_t))


def witness_from_X(X, L):
    """F_W = 1 - L/2 + X/4.  Maps a mean of X (or a sample mean) to the witness."""
    return 1.0 - L / 2.0 + X / 4.0


def p_plus(M_p, M_t, norm=None):
    """Probability of the + outcome. The estimator is Bernoulli(p_plus)."""
    nrm = l1_norm(M_t) if norm is None else norm
    return 0.5 + expected_X(M_p, M_t) / (4.0 * nrm)


def variance_X(M_p, M_t, norm=None):
    """Var[X] = 4|M_t|^2 - E[X]^2, exactly. X^2 is deterministic."""
    nrm = l1_norm(M_t) if norm is None else norm
    return 4.0 * nrm ** 2 - expected_X(M_p, M_t) ** 2


def stderr_witness(M_p, M_t, N, norm=None):
    """Standard error of F*_W after N shots: sqrt(Var[X]/N)/4."""
    return np.sqrt(variance_X(M_p, M_t, norm) / N) / 4.0


# ----------------------------------------------------------------------
# 3.  Shot budgets -- three levels of sharpness
# ----------------------------------------------------------------------
def shots_hoeffding(norm_M_t, eps, delta):
    """Paper Theorem 2: N <= ceil( ln(2/delta) |M_t|^2 / (2 eps^2) ).
    """
    return int(np.ceil(np.log(2.0 / delta) * norm_M_t ** 2 / (2.0 * eps ** 2)))


def shots_exact(norm_M_t, eps, delta, p=0.5, hi=None):
    """Smallest N with P(|p_hat - p| > eta) <= delta under the exact binomial.

    This is the true sample complexity of the estimator; Hoeffding bounds it
    from above. Bisection on the exact binomial tail.
    """
    eta = eps / norm_M_t

    def fails(N):
        lo_k = np.floor((p - eta) * N)
        hi_k = np.ceil((p + eta) * N) - 1
        return binom.cdf(lo_k, N, p) + binom.sf(hi_k, N, p)

    lo = 1
    hi = shots_hoeffding(norm_M_t, eps, delta) if hi is None else hi
    if fails(hi) > delta:                       # should not happen; be safe
        return hi
    while lo < hi:                              # smallest N that already works
        mid = (lo + hi) // 2
        if fails(mid) <= delta:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ----------------------------------------------------------------------
# 4.  The sampler
# ----------------------------------------------------------------------
def sample_X(plan, M_p, N, rng):
    """N draws of X, following the protocol exactly.

    beta is drawn from the Born rule P(beta|j,k) = (1 + beta M_jk(rho_p))/2,
    which is what a device would return. No device is needed to *test the
    estimator*: this samples the same distribution the hardware would.
    """
    idx = rng.choice(plan.j.size, size=N, p=plan.p)
    m = M_p[plan.j[idx], plan.k[idx]]
    beta = np.where(rng.random(N) < (1.0 + m) / 2.0, 1.0, -1.0)
    return 2.0 * plan.norm * beta * plan.sign[idx]


# ----------------------------------------------------------------------
# 5.  Baseline: uniform sampling over Omega (what NOT to do)
# ----------------------------------------------------------------------
def sample_X_uniform(plan, M_t, M_p, N, rng):
    """Unbiased too, but weights every pair equally: X = 2|Omega| beta M_t[j,k].

    Var ratio to importance sampling is |Omega| * sum(M_t^2) / |M_t|^2 >= 1 by
    Cauchy-Schwarz, with equality only if every |M_jk| is the same.
    """
    idx = rng.choice(plan.j.size, size=N)
    vt = M_t[plan.j[idx], plan.k[idx]]
    m = M_p[plan.j[idx], plan.k[idx]]
    beta = np.where(rng.random(N) < (1.0 + m) / 2.0, 1.0, -1.0)
    return 2.0 * plan.j.size * beta * vt


def variance_ratio_uniform(plan, M_t):
    """Var_uniform / Var_importance in the large-|M_t| limit (E[X]^2 negligible)."""
    sq = (M_t[plan.j, plan.k] ** 2).sum()
    return plan.j.size * sq / plan.norm ** 2


# ----------------------------------------------------------------------
# 6.  Hardware-faithful readout: Majorana pair -> Pauli string -> bitstring
# ----------------------------------------------------------------------
def pauli_string(j, k, L):
    """Pauli string for  i m_j m_k,  with 1-based Majorana indices j < k.

    Returns (sign, ops) where ops maps 1-based site -> 'X'/'Y'/'Z', and the
    observable is  sign * prod_site ops[site].  Derived from Eq. (15); the
    Z-string between the endpoint sites is what survives JW cancellation.

        j odd,  k odd   ->  + Y_a (Z..Z) X_b
        j odd,  k even  ->  + Y_a (Z..Z) Y_b
        j even, k odd   ->  - X_a (Z..Z) X_b
        j even, k even  ->  - X_a (Z..Z) Y_b
        same site       ->  - Z_a
    """
    a, b = (j + 1) // 2, (k + 1) // 2
    if a == b:                                   # i m_{2a-1} m_{2a} = -Z_a
        return -1.0, {a: "Z"}
    end_j = "Y" if j % 2 == 1 else "X"
    end_k = "X" if k % 2 == 1 else "Y"
    sign = 1.0 if j % 2 == 1 else -1.0
    ops = {a: end_j, b: end_k}
    for s in range(a + 1, b):
        ops[s] = "Z"
    return sign, ops


def measure_beta(psi, j, k, L, rng, shots):
    """Sample beta for i m_j m_k the way a device would, from a statevector.

    Rotates the endpoint operators into the Z basis with single-qubit gates
    (H for X, H S^dag for Y), samples full Z-basis bitstrings, and reads beta
    off as the parity over the string's support. Exercises the JW dictionary,
    every sign, and the parity readout -- the parts a covariance-matrix
    shortcut silently assumes.
    """
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    HSdag = H @ np.array([[1, 0], [0, -1j]], complex)

    sign, ops = pauli_string(j, k, L)
    state = psi.reshape([2] * L)
    for site, P in ops.items():
        if P == "Z":
            continue
        R = H if P == "X" else HSdag
        state = np.moveaxis(np.tensordot(R, state, axes=([1], [site - 1])), 0, site - 1)

    probs = np.abs(state.reshape(-1)) ** 2
    probs /= probs.sum()
    draws = rng.choice(probs.size, size=shots, p=probs)
    bits = ((draws[:, None] >> np.arange(L - 1, -1, -1)[None, :]) & 1)
    support = np.array(sorted(ops)) - 1
    return sign * (-1.0) ** bits[:, support].sum(axis=1)
