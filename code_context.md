# Fidelity witnesses for pure fermionic Gaussian states — working context

Reproduction of **arXiv:1703.03152** (Gluza, Kliesch, Eisert, Aolita),
*Fidelity witnesses for fermionic quantum simulations*.

Target: Figure 2 — certification of a sudden quench in a critical
transverse-field Ising chain.

Files:

| File | Contents |
|---|---|
| `tfim_fw.py` | Core library. Coupling matrix, propagators, covariance matrices, witness, Pfaffian, exact Gaussian overlap. |
| `sampling.py` | Importance-sampling estimator of `F_W`: measurement plan, closed-form variance, shot budgets (Theorem 2 / exact binomial), uniform baseline, qubit-level readout simulator. |
| `verify.py` | Correctness tests against brute-force exact diagonalization (small `L`). |
| `figures.py` | Reproduces Fig. 2 — prints the numbers and writes `figures/fig2*.png`. |
| `sampling_experiments.py` | Six sampling experiments — prints everything, writes `figures/fig3*.png`. Run all or by number. |
| `plotstyle.py` | Shared matplotlib palette/style. |

Run `python3 verify.py` after **any** change to `tfim_fw.py`. Every test should
return errors at machine precision (`~1e-16`) except Test 4 (an inequality) and
Test 6's second half (a physical plateau, not an error).

---

## 1. Conventions — read before touching anything

Sign conventions are the single most common source of bugs here. These follow
the paper exactly and are verified numerically.

**Jordan–Wigner (paper Eq. 15).** Majorana index `j = 1..2L`, site index
`k = 1..L`:

```
m_{2k-1} = (prod_{j<k} Z_j) X_k          # odd  index -> X
m_{2k}   = (prod_{j<k} Z_j) Y_k          # even index -> Y
```

**Occupation convention (derived, not assumed).** From the above,
`i m_{2k-1} m_{2k} = -Z_k`, and matching to `2 n_k - 1` gives

```
n_k = (1 - Z_k)/2        =>   |up> (Z = +1)  is  the EMPTY mode
```

Therefore `|up>^{otimes L}` is the fermionic vacuum, `omega = 0`, and its
covariance matrix is Eq. (17). If you ever flip this, every sign downstream
breaks.

**Hamiltonian (Eq. 18).** `H(A) = (i/4) sum_{j,k} A_{jk} m_j m_k`, with
`A` real antisymmetric `2L x 2L`.

**Matching rule (the factor of 2).** A term `i*c*m_a*m_b` in `H` corresponds to
`A[a,b] = 2c`, `A[b,a] = -2c`. The factor 2 comes from the double sum counting
`(a,b)` and `(b,a)`; the `1/4` in Eq. (18) is what turns `4c` into `2c`.

**Propagator (Eq. 19).** `m_j(t) = e^{itH} m_j e^{-itH} = sum_k Q_{jk}(t) m_k`
with `Q(t) = expm(t*A)`, real, in `SO(2L)`. No `i` in the exponent.

**Covariance matrix (Eq. 4).** `M_{jk}(rho) = (i/2) tr([m_j,m_k] rho)`, which for
`j != k` is `i <m_j m_k>`. Real, antisymmetric. Note the `i`: the bare product
`m_j m_k` is anti-Hermitian, so `<m_j m_k>` is imaginary and the `i` makes `M`
real. Any formula equating a real quantity to `<m_j m_k>` without an `i` has
lost a factor.

**Transformation (Eq. 54).** `M(U rho U^dag) = Q M(rho) Q^T`.

**Array indexing.** Paper uses 1-based Majorana indices `1..2L`; NumPy arrays are
0-based. `A_{jk}` (paper) is `A[j-1, k-1]` (code).

---

## 2. The coupling matrix (derived, verified)

For the XY chain in a transverse field (paper Eq. 16):

```
H_spin = -sum_{k=1}^{L-1} (Jx_k X_k X_{k+1} + Jy_k Y_k Y_{k+1})
         -sum_{k=1}^{L}   B_k Z_k
```

The three Jordan–Wigner identities (Z-strings cancel in all three):

```
Z_k            = -i m_{2k-1} m_{2k}          # separation 1, same site
X_k X_{k+1}    = -i m_{2k}   m_{2k+1}        # separation 1, across bond
Y_k Y_{k+1}    = +i m_{2k-1} m_{2k+2}        # separation 3
```

Applying the matching rule gives the nonzero entries of `A` (1-based indices):

```
A[2k-1, 2k]   = +2 B_k      k = 1..L
A[2k,   2k+1] = +2 Jx_k     k = 1..L-1
A[2k-1, 2k+2] = -2 Jy_k     k = 1..L-1
```

plus antisymmetry. Implemented in `coupling_matrix(L, Jx, Jy, B)`.

### Structure worth exploiting

For the Ising case (`Jy = 0`), `A` is a **dimerized Majorana chain** — tridiagonal
with alternating couplings:

```
[m1 --2B-- m2] --2J-- [m3 --2B-- m4] --2J-- [m5 --2B-- m6] --2J-- ...
  site 1                 site 2                site 3
```

Consequences:

- `A(B)` couples the disjoint pairs `(1,2),(3,4),...`
- `A(J)` couples the disjoint pairs `(2,3),(4,5),...`
- Neither self-overlaps, so **both Trotter-layer exponentials are closed-form
  2x2 rotations** — no `expm` needed. `propagator_pairwise()` detects this and
  falls back to `expm` when `Jy != 0` breaks it.

---

## 3. Physics reference (verified numerically)

Single-particle energies: eigenvalues of `A` are `+-i*eps_k`, `eps_k >= 0`.
Extract with

```python
eps = np.sort(np.abs(np.linalg.eigvals(A).imag))[::2]   # ascending, one per pair
```

The `[::2]` matters — eigenvalues come in `+-` pairs so the sorted absolute
values are doubled.

**Normal form.** `H = sum_k eps_k (n_k - 1/2)` where `n_k` are occupations of
*delocalized normal modes* (not lattice sites). All `2^L` many-body energies
follow from the `L` numbers `eps_k`; verified to `4e-14` at `L = 8`.

**Gap.** `Delta_spin = min_k eps_k`. For the translation-invariant ring,

```
eps(q)     = 2 sqrt(J^2 + B^2 - 2 J B cos q)
Delta_bulk = 2 |J - B|
```

Derivation: two-Majorana unit cell gives a `2x2` Bloch matrix
`[[0, z], [-conj(z), 0]]` with `z(q) = 2B - 2J exp(-i q)`; eigenvalues
`+-i|z|`. Geometrically `z(q)` traces a circle of radius `2J` centred at `2B`,
and `eps = |z|` is the distance to the origin.

| regime | circle vs origin | winding | phase | OBC lowest `eps` |
|---|---|---|---|---|
| `J > B` | encircles | 1 | ferromagnetic / topological | `~exp(-L/xi)` (edge Majoranas) |
| `J = B` | passes through | — | **critical** | `~pi*v/(2L)` |
| `J < B` | excludes | 0 | paramagnetic / trivial | `2(B-J)` |

with `xi = 1 / ln(J/B)` and `v = 2J`.

**Careful:** in the ordered phase with open boundaries, `min_k eps_k` is the
*edge-mode splitting*, not `2|J-B|`. Both are correct; they answer different
questions. `min_k eps_k` is always the true gap of the finite open chain.

**Light cone.** Quasiparticle pairs separate at `+-v`, so sites at distance `d`
become correlated at `t = d/(2v) = d/(4J)`. Measured front position `~ 4 J t`.
The paper's `t = L/8` with `J = 1` therefore puts the front at `d = L/2`, which
is exactly what the Fig. 2 caption claims.

**Three different "gaps" — do not conflate:**

| symbol | meaning | value here |
|---|---|---|
| `Delta_witness` | gap of `n^(omega)` in `W = 1 - H/Delta` | **always 1** |
| `Delta_spin` | spectral gap of `H_spin` | `2\|J-B\|` (bulk) |
| `eps_k` | single-particle spectrum of `A` | `Delta_spin = min_k eps_k` |

The witness's *soundness* does not depend on `Delta_spin` at all. The spin gap
only affects the *cost* of certification, through the correlation structure of
`M(rho_t)`.

---

## 4. The Figure 2 pipeline

```
Q(t) = expm(t * A(J,B))                       # exact continuous evolution
M(rho_t) = Q(t) @ M_up @ Q(t).T               # target,  computed classically

Q_T = (expm(dt*A_B) @ expm(dt*A_J))^T         # Trotterized digital simulation
M(rho_p) = Q_T @ M_up @ Q_T.T                 # preparation

F_W = 1 + (1/4) * tr[(M_p - M_t).T @ M_t]     # Eq. (8)
    = 1 - L/2 - (1/4) * tr[M_p @ M_t]         # Eq. (62), equivalent
```

with `M_up = M_fock(L)` = Eq. (17), `dt = t/T`, and the Fig. 2 setup
`J = B = 1`, `t = L/8`.

Trotter error plays the role of "noise": coherent, tunable via `T`, and exactly
computable classically because both states are Gaussian. Note this is a
**simulation of the protocol**, not a hardware run — no shot noise, no readout
error, no device noise. Only the *systematic* behaviour of the witness is shown.

### KNOWN AMBIGUITY — Trotter ordering

The paper writes `U_T = (e^{-i dt H_B} e^{-i dt H_J})^T` but
`Q_T = (e^{dt A_J} e^{dt A_B})^T` — **opposite order**.

This is not obviously a typo: under `C_U(m) = U^dag m U`, composing `U = U2 U1`
gives `Q = Q2 Q1`, so `U_T` as written maps to
`Q_T = (e^{dt A_B} e^{dt A_J})^T`.

Both orderings are valid first-order Trotterizations agreeing to `O(dt^2)`, so
both are implemented: `trotter_propagator(..., order="BJ" | "JB")`.

**Test 4 in `verify.py` settles it empirically.** At `L = 4`, comparing against
brute-force `|<psi_t|psi_p>|^2`, the `BJ` ordering tracks the true fidelity far
more tightly (`0.53` vs `0.21` at `T = 1`). `BJ` is the default in `figure2.py`.
Still worth rechecking Appendix A's composition convention directly.

---

## 5. Verification status

`verify.py` — against brute-force exact diagonalization in the full `2^L` space,
including `Jy != 0` and negative couplings:

| Test | Checks | Result |
|---|---|---|
| 1 | `H(A) == H_spin` | `1e-16` — pins every sign in `A` and the factor-of-2 rule |
| 2 | `e^{itH} m_j e^{-itH} == sum_k Q_jk m_k` | `1e-16` — Lemma 3 |
| 3 | `Q M_up Q^T == i<m_j m_k>` | `1e-16` — Eqs. (54) + (17) |
| 4 | `F_W <= F_true` for `T = 1..40` | holds; gap `0.25 -> 1e-4` |
| 5 | `QQ^T = I`, `M^2 = -I` at `L = 50, 100` | `1e-13` |

`figure2.py` — against the paper's reported numbers:

| Paper | Reproduced |
|---|---|
| `\|M(rho_t)\| ~ 2.11 * L^1.42` | `2.35 * L^1.403` (all entries) / `1.17 * L^1.403` (upper triangle) |
| `N <= O(L^2.84)` | `O(L^2.81)` |
| wavefront at `d ~ L/2` | `\|M\|` falls `0.12 -> 1e-12` between `d = 50` and `d = 70` (`L = 100`) |
| `L = 60` needs `T >~ 40` | crosses `F_W = 0` between `T = 40` and `50` |

**Open discrepancy:** the exponent matches to 1%, but the prefactor only matches
if the paper sums over **all** entries of `M` rather than the strict upper
triangle (`2.35` vs `2.11`). Eq. (10) is written as a sum over `Omega` with
`j < k`, which would give `1.17`. Unresolved — does not affect the `O(L^2.8)`
scaling. Could also be a different `L` range in their fit.

---

## 6. Invariants to assert in any new code

```python
purity_error(M)         # ||M^2 + I||_inf   == 0  for pure Gaussian states
orthogonality_error(Q)  # ||Q Q^T - I||_inf == 0  for Q in SO(2L)
fidelity_witness(M_t, M_t) == 1.0           # condition (i)
```

`M` must stay real and antisymmetric throughout. Since `Q` is orthogonal, purity
is preserved *exactly* — any violation means a bug in `A` or in the
exponentiation, not accumulated numerical error.

Useful limit check: as `T -> inf`, `Q_T -> Q(t)` and `F_W -> 1`.

---

## 7. Performance notes

- `L = 400` (i.e. `800 x 800`) runs in seconds; `expm` for `Q(t)` dominates.
- Beyond `L ~ 1000`: `i*A` is Hermitian, so diagonalize once and use
  `expm(t*A) = Re[V @ diag(exp(-i*t*d)) @ V.conj().T]`. Reusable across `t`.
- `trotter_propagator` uses `np.linalg.matrix_power` (repeated squaring,
  `O(log T)` matmuls), so large `T` is cheap.
- Everything is `O(L^3)` per evaluation. The Hilbert space is `2^L` and is never
  touched outside `verify.py`.

---

## 8. Next steps (not yet implemented)

**(a) Sampling layer — DONE.** `sampling.py` + `sampling_experiments.py`.
Findings, all verified numerically:

- The estimator is **one biased coin**: `X in {+-2|M_t|}`, so `X^2` is
  deterministic, `p+ = 1/2 + E[X]/(4|M_t|)`, and
  `Var[X] = 4|M_t|^2 - E[X]^2` **exactly** (closed form, no sampling).
  `F*_W` is affine in a binomial proportion.
- Hence **Theorem 2 is precisely Hoeffding on a Bernoulli**; its looseness is
  the standard slack `2 ln(2/delta)/z^2` = 1.92x at `delta = 0.05`, flat in
  `L` and in the state. The *exact* sample complexity is a binomial tail
  problem (`shots_exact`), sitting at ~52% of Theorem 2.
- Importance sampling beats uniform sampling over `Omega` by only
  `|Omega| sum(M_t^2)/|M_t|^2 ~ L^0.23` in variance (3.3x at L=100) — real
  but mild; the exponential saving is the mode-space basis itself.
- Qubit-level readout (JW string -> single-qubit rotations -> bitstring
  parity) validated against the covariance matrix, all signs included
  (`pauli_string`, `measure_beta`; exact at L=6).
- Certification test (notes §9) runs end to end: soundness held in every
  trial; the acceptance guarantee engages **only** when the declared
  `n_perp_max` has margin over the true demand (Trotter demand is `2+O(dt)`,
  reaching 4.3 at T=2, so declaring exactly 2.0 voids the guarantee).

**(a') Original plan for reference.** Everything so far computes `F_W`
*exactly* from `M(rho_p)`. The paper's Theorem 2 concerns estimating it from
finite shots.

- Sampling distribution: `P_{j,k} = |M_{jk}(rho_t)| / |M(rho_t)|` over `Omega`
- Per shot: draw `(j,k)`, measure `i m_j m_k` on `rho_p` giving `beta = +-1` with
  `P(beta) = (1 + beta * M_{jk}(rho_p)) / 2`, record
  `X = 2 |M(rho_t)| * beta * sign(M_{jk}(rho_t))`
- `E[X] = tr[M(rho_p)^T M(rho_t)]`, then feed into Eq. (8)
- Check Theorem 2 empirically:
  `N <= ceil( ln(2/delta) * |M(rho_t)|^2 / (2 eps^2) )`
- Worth comparing against the Appendix E grouping scheme, which is
  logarithmically worse in the bound (`ln(2|Omega|/delta)` union bound) but
  yields all of `M(rho_p)` as a byproduct

Note: no quantum device needed — `beta` is sampled classically from the known
`M(rho_p)`. This tests the *estimator*, not the hardware.

**(b) Plots. DONE.** `figures.py` writes panels (a) covariance image,
(b) |M| scaling, (c) witness-vs-T to `figures/`, individually and as a
combined `fig2.png`, and still prints the tables. The paper's z-string panel
was dropped as a figure by choice; `z_string`/`pfaffian` stay in the library,
covered by Test 6. Figure convention: |M(rho_t)| sums ALL 4L^2 entries
(= 2x the Eq.-10 j<k sum, which `l1_norm`/`sampling.py` keep internally);
measured fit 2.35 L^1.403, matching the notes' 2.35 L^1.40.

**(c) Top-right panel. DONE.** `pfaffian()` in `tfim_fw.py` is Parlett-Reid with
partial pivoting (the PFAPACK algorithm the paper cites as [67]); `z_string(M, n)`
gives `|<prod_{k=1}^n Z_k>| = |Pf(M_{1..2n})|`. Verified against brute force to
`3e-16` (Test 6).

Finding: the string decays as `10^(-0.27 n)` and then **plateaus** — at the
paper's `t = L/8 = 12.5` it flattens at `1.5e-7` from `n ~ 27`. The plateau is
*physical, not a precision floor*: its onset tracks `2t` (the light cone —
sites beyond it are still `|up>`, contributing factor 1), it is unchanged by
chopping `M` below `1e-10` or by adding `1e-10` noise, and `sqrt(|det|)` agrees
with the Pfaffian throughout because `M` is well conditioned. Use the Pfaffian
anyway — it is the object Wick's theorem gives and it carries a sign — but do
not expect it to buy precision here.

**(d) Realistic noise.** Replace Trotter error with an actual noise channel.
Caution: depolarizing noise makes `rho_p` **non-Gaussian**, so `M(rho_p)` no
longer characterises it. Eq. (8) remains a valid bound (the derivation never
used Wick's theorem on `rho_p` — `n^(omega)` is only quadratic), but you can no
longer compute `F_true` from covariance matrices alone. Restrict `F_true`
cross-checks to small `L` with exact density matrices.

**(e) Systematic vs statistical error.** `F_W <= F` always; the systematic gap
`F - F_W = tr[A rho_p] >= 0` does not shrink with shot count. Driving the
statistical `eps` far below that gap wastes budget. Quantify the gap for the
Trotter case at small `L` where `F_true` is computable.
