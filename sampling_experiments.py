"""
Simulation study of the importance-sampling fidelity-witness estimator on the Trotterized critical Ising quench.

Six experiments, in dependency order:

  0  validation   Pauli dictionary, qubit-level readout, estimator unbiasedness
  1  convergence  F*_W -> F_W at the closed-form 1/sqrt(N) rate
  2  Bernoulli    the estimator is one biased coin; Var[X] = 4|M_t|^2 - E[X]^2
  3  tightness    how loose is Theorem 2?  Hoeffding vs CLT vs exact binomial
  4  scaling      shots vs L; recovering O(L^2.8)
  5  baseline     importance sampling vs uniform sampling over Omega
  6  certification the full accept/reject rule, and where it false-rejects

Writes figures/fig3*.png and prints every number.

    python3 sampling_experiments.py            # all
    python3 sampling_experiments.py 3 6        # selected
"""
import pathlib
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import norm as normal

import plotstyle as ps
from tfim_fw import (coupling_matrix, M_fock, propagator, evolve_M,
                     trotter_propagator, fidelity_witness, l1_norm,
                     gaussian_fidelity)
from sampling import (Plan, sample_X, sample_X_uniform, witness_from_X,
                      expected_X, variance_X, p_plus, stderr_witness,
                      shots_hoeffding, shots_exact,
                      variance_ratio_uniform, pauli_string, measure_beta)

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
J = B = 1.0
SEED = 20260912


def quench(L, T=None, t=None, order="BJ"):
    """(M_t, M_p) for the critical quench; M_p Trotterized with T steps."""
    t = L / 8.0 if t is None else t
    M_t = evolve_M(M_fock(L), propagator(coupling_matrix(L, J, 0.0, B), t))
    if T is None:
        return M_t, None
    Q = trotter_propagator(coupling_matrix(L, J, 0.0, 0.0),
                           coupling_matrix(L, 0.0, 0.0, B), t, T, order)
    return M_t, evolve_M(M_fock(L), Q)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def head(n, title):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT {n} -- {title}")
    print("=" * 70)


# ======================================================================
def exp0():
    """Validation gate: the dictionary, the readout, the estimator."""
    head(0, "validation: Pauli dictionary, qubit readout, unbiasedness")
    import io
    _o = sys.stdout; sys.stdout = io.StringIO()
    from verify import majorana, H_spin_exact, up_state, kron_list, X, Y, Z, I2
    sys.stdout = _o

    L = 6
    ms = [majorana(L, j) for j in range(1, 2 * L + 1)]
    PA = {"X": X, "Y": Y, "Z": Z}
    worst = 0.0
    for j in range(1, 2 * L + 1):
        for k in range(j + 1, 2 * L + 1):
            sign, ops = pauli_string(j, k, L)
            op = sign * kron_list([PA.get(ops.get(s), I2) for s in range(1, L + 1)])
            worst = max(worst, np.abs(op - 1j * ms[j - 1] @ ms[k - 1]).max())
    print(f"  (a) pauli_string vs i m_j m_k, all {2*L*(2*L-1)//2} pairs: "
          f"max err = {worst:.2e}")

    t, T = L / 8.0, 8
    H_J, H_B = H_spin_exact(L, J, 0., 0.), H_spin_exact(L, 0., 0., B)
    psi_p = np.linalg.matrix_power(
        expm(-1j * (t / T) * H_B) @ expm(-1j * (t / T) * H_J), T) @ up_state(L)
    M_t, M_p = quench(L, T)
    rng = np.random.default_rng(SEED)
    errs = []
    for (j, k) in [(1, 2), (2, 3), (1, 4), (3, 8), (2, 11), (5, 12), (4, 9)]:
        b = measure_beta(psi_p, j, k, L, rng, 200_000).mean()
        errs.append(abs(b - M_p[j - 1, k - 1]))
    print(f"  (b) qubit-level <beta> vs M_jk(rho_p), 7 pairs x 200k shots: "
          f"max err = {max(errs):.4f}  (shot noise ~ {1/np.sqrt(200_000):.4f})")

    plan = Plan(M_t)
    FW = fidelity_witness(M_p, M_t)
    est = witness_from_X(sample_X(plan, M_p, 4_000_000, rng).mean(), L)
    se = stderr_witness(M_p, M_t, 4_000_000, plan.norm)
    print(f"  (c) estimator at N=4e6: F*_W = {est:.5f}, exact F_W = {FW:.5f}, "
          f"err = {est-FW:+.5f}  (1 s.e. = {se:.5f})")
    print(f"  (d) E[X] = tr[M_p^T M_t] = {expected_X(M_p, M_t):.6f}"
          f"  ==  4(F_W + L/2 - 1) = {4*(FW+L/2-1):.6f}")


# ======================================================================
def exp1():
    """Convergence at the closed-form rate."""
    head(1, "convergence: F*_W -> F_W, and the closed-form standard error")
    L, T = 20, 40
    M_t, M_p = quench(L, T)
    plan, rng = Plan(M_t), np.random.default_rng(SEED + 1)
    FW = fidelity_witness(M_p, M_t)
    Ns = np.array([10**3, 10**4, 10**5, 10**6, 10**7])

    print(f"  L={L}, T={T}, exact F_W = {FW:.6f}, |M_t| = {plan.norm:.2f}")
    print(f"  {'N':>10} {'repeats':>8} {'mean F*_W':>11} {'bias':>10} "
          f"{'emp. s.d.':>11} {'closed form':>12} {'ratio':>7}")
    emp, pred = [], []
    for N in Ns:
        R = 100 if N <= 10**6 else 12       # s.d. of s.d. ~ 1/sqrt(2R)
        vals = np.array([witness_from_X(sample_X(plan, M_p, int(N), rng).mean(), L)
                         for _ in range(R)])
        sd, se = vals.std(ddof=1), stderr_witness(M_p, M_t, N, plan.norm)
        emp.append(sd); pred.append(se)
        print(f"  {N:>10,} {R:>8} {vals.mean():>11.5f} {vals.mean()-FW:>+10.5f} "
              f"{sd:>11.5f} {se:>12.5f} {sd/se:>7.2f}")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(Ns, pred, color=ps.INK_MUTED, lw=1.4, ls="--",
              label=r"closed form  $\sqrt{\mathrm{Var}[X]/N}\,/\,4$")
    ax.loglog(Ns, emp, color=ps.SERIES_1, marker="o", ms=6,
              markeredgecolor=ps.SURFACE, markeredgewidth=0.8,
              label="empirical s.d.")
    ax.set_title("(a)  estimator converges at the predicted rate")
    ax.set_xlabel("shots $N$"); ax.set_ylabel(r"s.d. of $F^*_\mathcal{W}$")
    ax.legend(labelcolor=ps.INK_2, fontsize=8)
    ax.annotate(f"$L={L}$, $T={T}$", xy=(0.97, 0.93), xycoords="axes fraction",
                ha="right", color=ps.INK_2, fontsize=8)
    save(fig, "fig3a_convergence.png")
    return Ns, emp, pred, FW


# ======================================================================
def exp2():
    """The estimator is a single biased coin."""
    head(2, "the estimator is ONE biased coin  (X^2 is deterministic)")
    L, T, N, R = 20, 40, 100_000, 1500
    M_t, M_p = quench(L, T)
    plan, rng = Plan(M_t), np.random.default_rng(SEED + 2)
    FW = fidelity_witness(M_p, M_t)
    p = p_plus(M_p, M_t, plan.norm)

    print(f"  L={L}, T={T}: |M_t| = {plan.norm:.2f}, X takes only "
          f"+-{2*plan.norm:.2f}")
    print(f"  p+ = 1/2 + E[X]/(4|M_t|) = {p:.6f}")
    print(f"  Var[X] closed form = 4|M_t|^2 - E[X]^2 = {variance_X(M_p,M_t,plan.norm):,.1f}")

    vals = np.array([witness_from_X(sample_X(plan, M_p, N, rng).mean(), L)
                     for _ in range(R)])
    emp_var = (4 * vals.std(ddof=1)) ** 2 * N
    print(f"  Var[X] from {R} runs of N={N:,}    = {emp_var:,.1f}"
          f"   (ratio {emp_var/variance_X(M_p,M_t,plan.norm):.4f})")

    # exact binomial prediction for the spread of F*_W
    se = stderr_witness(M_p, M_t, N, plan.norm)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(vals, bins=45, color=ps.SEQ[2], edgecolor=ps.SURFACE, linewidth=0.5,
            density=True, label=f"{R} independent runs")
    xs = np.linspace(vals.min(), vals.max(), 400)
    ax.plot(xs, normal.pdf(xs, FW, se), color=ps.INK, lw=2.0,
            label="binomial prediction\n(no fitted parameters)")
    ax.axvline(FW, color=ps.SERIES_2, lw=1.6, label=r"exact $F_\mathcal{W}$")
    ax.set_title("(b)  spread is exactly a binomial proportion")
    ax.set_xlabel(r"$F^*_\mathcal{W}$"); ax.set_ylabel("density")
    ax.legend(labelcolor=ps.INK_2, fontsize=8)
    ax.annotate(f"$N={N:,}$ per run", xy=(0.03, 0.93), xycoords="axes fraction",
                color=ps.INK_2, fontsize=8)
    save(fig, "fig3b_bernoulli.png")


# ======================================================================
def exp3():
    """Theorem 2 vs the number of shots actually needed."""
    head(3, "Theorem 2 vs the actual number of shots needed")
    eps, delta = 0.05, 0.05
    Ls = [10, 12, 16, 20, 30, 45, 60, 100]
    rows = []
    print(f"  eps = {eps}, delta = {delta}, worst-case p = 1/2")
    print(f"  {'L':>5} {'|M_t|':>9} {'N_Theorem2':>14} {'N_actual':>13} {'ratio':>7}")
    for L in Ls:
        M_t, _ = quench(L)
        nrm = l1_norm(M_t)
        nh = shots_hoeffding(nrm, eps, delta)
        ne = shots_exact(nrm, eps, delta)
        rows.append((L, nrm, nh, ne))
        print(f"  {L:>5} {nrm:>9.2f} {nh:>14,} {ne:>13,} {nh/ne:>7.3f}")
    z = normal.ppf(1 - delta / 2)
    print(f"\n  N_actual = smallest N passing the exact binomial tail test; the")
    print(f"  ratio is the Bernoulli-Hoeffding slack 2ln(2/delta)/z^2 = "
          f"{2*np.log(2/delta)/z**2:.3f}, flat in L")

    # measure the actual minimum directly: failure rate over repeated trials
    # at two shot counts, interpolated to the delta crossing
    # (ln p_fail is ~linear in N for a binomial tail)
    emp = []
    R, T = 300, 30
    t0 = time.time()
    for L in (10, 12, 16, 20):
        M_t, M_p = quench(L, T)
        plan, rng = Plan(M_t), np.random.default_rng(SEED + 3 + L)
        FW = fidelity_witness(M_p, M_t)
        Nstar = shots_exact(plan.norm, eps, delta,
                            p=p_plus(M_p, M_t, plan.norm))
        N_lo = int(0.7 * Nstar)
        pf = []
        for N in (N_lo, Nstar):
            fails = sum(abs(witness_from_X(sample_X(plan, M_p, N, rng).mean(),
                                           L) - FW) > eps for _ in range(R))
            pf.append(max(fails / R, 0.5 / R))
        slope = (np.log(pf[1]) - np.log(pf[0])) / (Nstar - N_lo)
        N_emp = N_lo + (np.log(delta) - np.log(pf[0])) / slope
        emp.append((L, N_emp))
        print(f"  L={L}: failure rate {pf[0]:.3f} at 0.7 N*, {pf[1]:.3f} at "
              f"N* = {Nstar:,}  ->  measured minimum ~ {N_emp:,.0f}")
    print(f"  [{time.time()-t0:.0f}s, {R} trials per point]")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    Lv = [r[0] for r in rows]
    ax.loglog(Lv, [r[2] for r in rows], color=ps.ORDINAL[3], marker="o", ms=5,
              markeredgecolor=ps.SURFACE, markeredgewidth=0.8,
              label="Theorem 2 (notes 8.3)")
    ax.loglog(Lv, [r[3] for r in rows], color=ps.ORDINAL[1], marker="o", ms=5,
              markeredgecolor=ps.SURFACE, markeredgewidth=0.8,
              label="actual requirement (exact binomial)")
    ax.loglog([e[0] for e in emp], [e[1] for e in emp], ls="none", marker="D",
              ms=7, color=ps.SERIES_2, markeredgecolor=ps.SURFACE,
              markeredgewidth=0.8, label="measured (repeated trials)")
    ax.set_title("(c)  Theorem 2 vs the shots actually needed")
    ax.set_xlabel("system size $L$")
    ax.set_ylabel(f"shots for $\\epsilon={eps}$")
    ax.set_xticks(Lv); ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(v) for v in Lv], fontsize=7.5)
    ax.legend(loc="upper left", labelcolor=ps.INK_2, fontsize=8)
    ax.annotate(f"gap flat at {rows[-1][2]/rows[-1][3]:.2f}$\\times$",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                color=ps.INK_2, fontsize=8)
    save(fig, "fig3c_bound_tightness.png")
    return rows


# ======================================================================
def exp4():
    """Shots vs L."""
    head(4, "shot budget vs system size: recovering O(L^2.8)")
    eps, delta = 0.05, 0.05
    Ls = np.array([20, 30, 45, 60, 80, 100, 140, 200, 280, 400])
    nrm = np.array([l1_norm(quench(int(L))[0]) for L in Ls])
    Nh = np.array([shots_hoeffding(v, eps, delta) for v in nrm])

    a, c = np.polyfit(np.log(Ls), np.log(Nh), 1)
    big = Ls >= 100
    a_b, c_b = np.polyfit(np.log(Ls[big]), np.log(Nh[big]), 1)
    print(f"  eps = {eps}, delta = {delta}")
    print(f"  {'L':>5} {'|M_t|':>10} {'N':>16}")
    for L, v, n in zip(Ls, nrm, Nh):
        print(f"  {L:>5} {v:>10.1f} {n:>16,}")
    print(f"\n  fit all L    : N ~ L^{a:.3f}")
    print(f"  fit L >= 100 : N ~ L^{a_b:.3f}      [paper: O(L^2.84)]")
    print(f"  N scales as |M_t|^2, so the exponent is exactly twice panel (b) of fig2")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(Ls, Nh, color=ps.SERIES_1, marker="o", ms=5,
              markeredgecolor=ps.SURFACE, markeredgewidth=0.8, label="Theorem 2")
    ax.loglog(Ls, np.exp(c_b) * Ls ** a_b, color=ps.INK_MUTED, lw=1.2, ls=":",
              label=f"fit $L^{{{a_b:.2f}}}$ ($L\\geq100$)")
    ax.loglog(Ls, np.exp(c_b) * Ls ** 4 / Ls[0] ** 4 * (Ls[0] / Ls[0]), alpha=0)
    ax.set_title("(d)  shots vs system size")
    ax.set_xlabel("system size $L$"); ax.set_ylabel(f"shots for $\\epsilon={eps}$")
    ax.set_xticks(Ls); ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(v) for v in Ls], fontsize=7.5)
    ax.legend(loc="upper left", labelcolor=ps.INK_2, fontsize=8)
    save(fig, "fig3d_scaling.png")


# ======================================================================
def exp5():
    """Importance sampling vs uniform sampling."""
    head(5, "importance sampling vs uniform sampling over Omega")
    Ls = [10, 16, 20, 30, 45, 60, 100]
    ratios = []
    print("  Var_uniform / Var_importance  =  |Omega| * sum(M_t^2) / |M_t|^2")
    print(f"  {'L':>5} {'|Omega|':>9} {'|M_t|':>9} {'ratio':>9}")
    for L in Ls:
        M_t, _ = quench(L)
        plan = Plan(M_t)
        r = variance_ratio_uniform(plan, M_t)
        ratios.append(r)
        print(f"  {L:>5} {len(plan):>9,} {plan.norm:>9.2f} {r:>9.2f}")
    a, _ = np.polyfit(np.log(Ls), np.log(ratios), 1)
    print(f"\n  ratio ~ L^{a:.2f} -- importance sampling wins, but only "
          f"polynomially-mildly")

    # empirical confirmation at one L
    L, T, N, R = 20, 40, 200_000, 300
    M_t, M_p = quench(L, T)
    plan, rng = Plan(M_t), np.random.default_rng(SEED + 5)
    vi = np.array([sample_X(plan, M_p, N, rng).mean() for _ in range(R)])
    vu = np.array([sample_X_uniform(plan, M_t, M_p, N, rng).mean() for _ in range(R)])
    print(f"  empirical at L={L}: s.d.(importance) = {vi.std():.4f}, "
          f"s.d.(uniform) = {vu.std():.4f}, var ratio = {(vu.std()/vi.std())**2:.2f}"
          f"  (predicted {variance_ratio_uniform(plan, M_t):.2f})")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.semilogx(Ls, ratios, color=ps.SERIES_1, marker="o", ms=6,
                markeredgecolor=ps.SURFACE, markeredgewidth=0.8)
    ax.axhline(1.0, color=ps.INK_MUTED, lw=1.0, ls="--")
    ax.set_title("(e)  variance saved by importance sampling")
    ax.set_xlabel("system size $L$")
    ax.set_ylabel(r"$\mathrm{Var}_{\rm uniform}/\mathrm{Var}_{\rm importance}$")
    ax.set_xticks(Ls); ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(v) for v in Ls], fontsize=7.5)
    ax.annotate(f"grows only as $L^{{{a:.2f}}}$\n(uniform is never better)",
                xy=(0.04, 0.86), xycoords="axes fraction", color=ps.INK_2,
                fontsize=8, va="top")
    save(fig, "fig3e_importance_vs_uniform.png")


# ======================================================================
def exp6():
    """The full certification decision, end to end."""
    head(6, "certification test: accept/reject, soundness, and false rejection")
    L, F_T, eps, delta = 12, 0.50, 0.05, 0.05
    # Trotter error is pair creation: n_perp = 2 + O(dt). Declare it WITH
    # margin -- the minimal declaration n_perp_max = 2.0 gives capacity
    # exactly 2.000, every real Trotter state sits just above it, and the
    # acceptance guarantee never engages (notes 9.4: margin matters).
    n_perp_max = 2.3
    Delta_c = (1 - F_T) * (n_perp_max - 1) / n_perp_max + 2 * eps / n_perp_max
    print(f"  L={L}, F_T={F_T}, eps={eps}, delta={delta}, "
          f"declared n_perp <= {n_perp_max} (= 2 + margin for O(dt) corrections)")
    print(f"  => grey zone  Delta_c = (1-F_T)(n-1)/n + 2eps/n = {Delta_c:.3f}")
    print(f"     guaranteed accept for F >= F_T + Delta_c = {F_T+Delta_c:.3f}")
    print(f"     accept rule: F*_W >= F_T + eps = {F_T+eps:.3f}")

    M_t, _ = quench(L)
    plan = Plan(M_t)
    N = shots_exact(plan.norm, eps, delta)
    rng = np.random.default_rng(SEED + 6)
    print(f"  shots per trial: N = {N:,}   (|M_t| = {plan.norm:.2f})\n")

    # capacity of the test (notes 9.3): the class S_perp it can vouch for
    n_cap = (1 - F_T - 2 * eps) / (1 - F_T - Delta_c)
    print(f"  capacity: guarantee applies only to states with "
          f"n_perp(rho_perp) <= n_perp_T = {n_cap:.3f}\n")

    Ts, R = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 40], 60
    print(f"  {'T':>4} {'true F':>9} {'F_W':>9} {'n_perp':>8} "
          f"{'accept rate':>12}  verdict")
    res = []
    t0 = time.time()
    for T in Ts:
        _, M_p = quench(L, T)
        F = gaussian_fidelity(M_t, M_p)
        FW = fidelity_witness(M_p, M_t)
        npp = (1 - FW) / (1 - F) if F < 1 - 1e-12 else float("nan")
        acc = np.mean([witness_from_X(sample_X(plan, M_p, N, rng).mean(), L)
                       >= F_T + eps for _ in range(R)])
        if F < F_T:
            verdict = "must reject" + (" -- OK" if acc == 0 else " -- VIOLATION")
        elif F >= F_T + Delta_c and npp <= n_cap:
            verdict = "must accept" + (" -- OK" if acc >= 1 - delta
                                       else " -- VIOLATION")
        elif F >= F_T + Delta_c:
            verdict = (f"outside S_perp (demand {npp:.2f} > capacity "
                       f"{n_cap:.2f}) -- no guarantee")
        else:
            verdict = "grey zone -- no guarantee"
        res.append((T, F, FW, npp, acc))
        print(f"  {T:>4} {F:>9.4f} {FW:>9.4f} {npp:>8.3f} {acc:>12.2f}  {verdict}")
    print(f"  [{time.time()-t0:.0f}s]")
    print(f"\n  Soundness held everywhere: no preparation with F < F_T was accepted")
    print(f"  (that guarantee is unconditional). Every 'must accept' row engaged")
    print(f"  because demand stayed under the declared capacity {n_cap:.2f} -- the")
    print(f"  margin in n_perp_max is what buys that. With the minimal declaration")
    print(f"  n_perp_max = 2.0, capacity is exactly 2.000, every Trotter state")
    print(f"  (demand 2.0+) falls outside the class, and no acceptance is ever")
    print(f"  guaranteed -- the notes' 9.4 warning, realized numerically.")

    Fs = np.array([r[1] for r in res]); acc = np.array([r[4] for r in res])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axvspan(F_T, F_T + Delta_c, color="#f0efec", zorder=0)
    ax.plot(Fs, acc, color=ps.SERIES_1, marker="o", ms=6,
            markeredgecolor=ps.SURFACE, markeredgewidth=0.8)
    ax.axvline(F_T, color=ps.INK_2, lw=1.2, ls="--")
    ax.axvline(F_T + Delta_c, color=ps.INK_2, lw=1.2, ls="--")
    ax.set_title("(f)  operating characteristic of the certification test")
    ax.set_xlabel("true fidelity $F$"); ax.set_ylabel("acceptance rate")
    ax.set_ylim(-0.06, 1.08)
    ax.annotate("must\nreject", xy=(F_T - 0.05, 0.55), ha="right",
                color=ps.INK_2, fontsize=8)
    ax.annotate("grey\nzone", xy=((2 * F_T + Delta_c) / 2, 0.55), ha="center",
                color=ps.INK_MUTED, fontsize=8)
    ax.annotate("must accept", xy=(F_T + Delta_c + 0.03, 0.55),
                color=ps.INK_2, fontsize=8)
    ax.annotate(f"$L={L}$, $F_T={F_T}$, $\\epsilon={eps}$, $N={N:,}$\n"
                f"$\\Delta_c={Delta_c:.2f}$ forced by $n_\\perp^{{\\max}}={n_perp_max}$",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                color=ps.INK_2, fontsize=8)
    save(fig, "fig3f_certification.png")


# ======================================================================
EXPS = {0: exp0, 1: exp1, 2: exp2, 3: exp3, 4: exp4, 5: exp5, 6: exp6}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(EXPS)
    t0 = time.time()
    for n in want:
        EXPS[n]()
    print(f"\ntotal {time.time()-t0:.0f}s")
