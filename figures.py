"""
Reproduce Figure 2 of arXiv:1703.03152 -- certification of a sudden quench in a
critical transverse-field Ising chain.

Fig. 2 is the paper's only figure with numerical content (Fig. 1 is a hand-drawn
schematic of the witness geometry). Reproduced panels:

  (a) |M_{jk}(rho_t)| as an image   -- the correlation wavefront
  (b) |M(rho_t)| vs L               -- the sample-complexity scaling
  (c) F_W vs Trotter steps T        -- the witness in action

(The paper's z-string panel is not reproduced -- the machinery lives in
tfim_fw.z_string, tested in Test 6.)  |M(rho_t)| here sums the modulus of ALL
4L^2 entries, the convention of the working notes and of the paper's plot; by
antisymmetry it is exactly twice the j<k sum of Eq. (10), which the sampling
code uses internally.
Each panel is written to figures/ on its own, plus a combined fig2.png.
Numbers are also printed, so the script stays useful without opening the images.

    python3 figures.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps
from tfim_fw import (coupling_matrix, M_fock, propagator, evolve_M,
                     trotter_propagator, fidelity_witness, l1_norm,
                     omega_set_size, purity_error)

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

J = B = 1.0                      # critical
LS_SCALING = [20, 30, 45, 60, 80, 100, 140, 200, 280, 400]
LS_WITNESS = [15, 30, 45, 60]    # paper's Fig. 2 bottom-right panel
TS = [10, 20, 30, 40, 50, 60, 80, 100]
L_MAP = 100                      # paper's Fig. 2 top-left


def target_M(L, t=None):
    """Covariance matrix of the exact continuous-time quenched state."""
    t = L / 8.0 if t is None else t
    return evolve_M(M_fock(L), propagator(coupling_matrix(L, J, 0.0, B), t))


# ======================================================================
# (a) covariance matrix image -- the correlation wavefront
# ======================================================================
def panel_a(ax):
    M = target_M(L_MAP)
    im = ax.imshow(np.abs(M), cmap=ps.BLUE_CMAP, vmin=0.0, vmax=0.5,
                   interpolation="nearest")
    ax.set_title("(a)  $|M_{jk}(\\rho_t)|$,  $L=100$,  $t=L/8$")
    ax.set_xlabel("Majorana index $k$")
    ax.set_ylabel("Majorana index $j$")
    ax.grid(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=ps.INK_MUTED)
    cb.set_label("$|M_{jk}|$", color=ps.INK_2)

    # guide lines at site distance L/2 -- i.e. 2*(L/2) Majorana indices off the
    # diagonal -- which is where the caption says the wavefront has reached
    n, off = 2 * L_MAP, L_MAP
    ax.plot([off, n - 1], [0, n - 1 - off], color=ps.INK_2, lw=1.0, ls="--")
    ax.plot([0, n - 1 - off], [off, n - 1], color=ps.INK_2, lw=1.0, ls="--")
    ax.text(n - 6, 30, "dashed: site\ndistance $L/2$", color=ps.INK_2, fontsize=8,
            ha="right", va="top")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    return M


def wavefront_table(M, L):
    print("\n(a)  correlation wavefront   (max |M| at site distance d, L=%d)" % L)
    print(f"  {'d':>5} {'max |M|':>12}")
    for d in [0, 10, 20, 30, 40, 45, 50, 55, 60, 70]:
        v = max(abs(M[2 * k, 2 * (k + d) + 1]) for k in range(L - d))
        print(f"  {d:>5} {v:>12.3e}")
    print("  -> edge near d = L/2 = %d, as the Fig. 2 caption states" % (L // 2))


# ======================================================================
# (b) |M(rho_t)| vs L -- the sample-complexity scaling
# ======================================================================
def panel_b(ax):
    Ls = np.array(LS_SCALING)
    norms, omegas = [], []
    for L in Ls:
        M = target_M(L)
        norms.append(2 * l1_norm(M))    # ALL 4L^2 entries; antisymmetry makes
        omegas.append(omega_set_size(M) / (2 * L ** 2))   # this 2x the j<k sum
    norms = np.array(norms)

    a, c = np.polyfit(np.log(Ls), np.log(norms), 1)       # notes: 2.35 L^1.40

    ax.loglog(Ls, norms, color=ps.SERIES_1, marker="o", ms=5,
              markeredgecolor=ps.SURFACE, markeredgewidth=0.8,
              label="measured $|M(\\rho_t)|$")
    ax.loglog(Ls, 2.11 * Ls ** 1.42, color=ps.INK_MUTED, lw=1.2, ls=":",
              label="paper's fit  $2.11\\,L^{1.42}$")
    ax.set_title("(b)  $|M(\\rho_t)|$ vs system size")
    ax.set_xlabel("system size $L$")
    ax.set_ylabel("$|M(\\rho_t)|$")
    ax.set_xticks(LS_SCALING)                       # plain labels; the default
    ax.set_xticks([], minor=True)                   # log ticks collide here
    ax.set_xticklabels([str(L) for L in LS_SCALING], fontsize=7.5)
    ax.legend(loc="upper left", fontsize=7.5, labelcolor=ps.INK_2)
    ax.annotate(f"fit:  ${np.exp(c):.2f}\\,L^{{{a:.3f}}}$"
                f"   (notes: $2.35\\,L^{{1.40}}$)"
                f"\n$\\Rightarrow\\ \\mathcal{{N}} \\sim O(L^{{{2*a:.2f}}})$",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                color=ps.INK_2, fontsize=8)
    return Ls, norms, omegas, (a, c)


# ======================================================================
# (c) F_W vs Trotter steps -- the witness in action
# ======================================================================
def panel_c_witness(ax):
    rows = {}
    for i, L in enumerate(LS_WITNESS):
        t = L / 8.0
        A_J = coupling_matrix(L, J, 0.0, 0.0)
        A_B = coupling_matrix(L, 0.0, 0.0, B)
        M_t = target_M(L, t)
        # order="BJ" is the ordering that matches the paper's U_T; verified to
        # machine precision against the Heisenberg action of U_T (see notes).
        fw = [fidelity_witness(evolve_M(M_fock(L),
                                        trotter_propagator(A_J, A_B, t, T, "BJ")), M_t)
              for T in TS]
        rows[L] = fw
        ax.plot(TS, fw, color=ps.ORDINAL[i], marker="o", ms=5,
                markeredgecolor=ps.SURFACE, markeredgewidth=0.8, label=f"$L={L}$")
        # direct-label where the curves are still well separated in T; at the
        # right edge they have all converged and the labels would overprint
        j = next(k for k, v in enumerate(fw) if v > 0.35)
        dx, ha = (9, "left") if j == 0 else (-9, "right")
        ax.annotate(f"$L={L}$", xy=(TS[j], fw[j]), xytext=(dx, 9),
                    textcoords="offset points", color=ps.ORDINAL[i], fontsize=8,
                    ha=ha, bbox=dict(fc=ps.SURFACE, ec="none", pad=1.2))

    lo = -0.25
    ax.axhspan(lo, 0.0, color="#f0efec", zorder=0)
    ax.axhline(0.0, color=ps.INK_MUTED, lw=1.0)
    ax.set_title("(c)  $F_\\mathcal{W}$ vs Trotter steps")
    ax.set_xlabel("Trotter steps $T$")
    ax.set_ylabel("$F_\\mathcal{W}$")
    ax.set_ylim(lo, 1.08)                    # paper's range; curves dive to -29
    ax.set_xlim(TS[0] - 3, TS[-1] + 4)       # at small T, off the bottom
    ax.legend(loc="lower right", fontsize=7.5, labelcolor=ps.INK_2)
    ax.annotate("$F_\\mathcal{W}<0$: witness carries no information",
                xy=(TS[0] - 1, -0.19), color=ps.INK_MUTED, fontsize=8)
    return rows


# ======================================================================
def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(pathlib.Path.cwd())}")


if __name__ == "__main__":
    print("=" * 64)
    print("Figure 2 of arXiv:1703.03152   (J = B = 1, t = L/8)")
    print("=" * 64)

    # --- individual panels ---
    fig, ax = plt.subplots(figsize=(5.0, 4.2)); M_map = panel_a(ax)
    save(fig, "fig2a_covariance_matrix.png")

    fig, ax = plt.subplots(figsize=(5.0, 4.0)); Ls, norms, omegas, fit = panel_b(ax)
    save(fig, "fig2b_norm_scaling.png")

    fig, ax = plt.subplots(figsize=(5.0, 4.0)); fw_rows = panel_c_witness(ax)
    save(fig, "fig2c_witness_vs_trotter.png")

    # --- combined row ---
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    panel_a(axes[0]); panel_b(axes[1]); panel_c_witness(axes[2])
    fig.suptitle("Certification of a sudden quench in a critical Ising chain "
                 "($J=B=1$, $t=L/8$)", x=0.06, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig2.png")

    # ------------------------------------------------------------------
    # numbers, so the script is still useful without opening the images
    # ------------------------------------------------------------------
    wavefront_table(M_map, L_MAP)

    print("\n(b)  |M(rho_t)| vs L   (all 4L^2 entries)")
    print(f"  {'L':>5} {'|M(rho_t)|':>12} {'|Omega|/2L^2':>13} {'purity err':>12}")
    for L, nm, om in zip(Ls, norms, omegas):
        print(f"  {L:>5} {nm:>12.2f} {om:>13.3f}"
              f" {purity_error(target_M(int(L))):>12.1e}")
    a, c = fit
    print(f"  fit   : {np.exp(c):.2f} L^{a:.3f}   -> N ~ O(L^{2*a:.2f})")
    print("  notes : 2.35 L^1.40")
    print("  paper : 2.11 L^1.42  (their fit; same data, larger-L window)")

    print("\n(c)  F_W vs T")
    print(f"  {'T':>5}" + "".join(f"{'L=' + str(L):>12}" for L in LS_WITNESS))
    for i, T in enumerate(TS):
        print(f"  {T:>5}" + "".join(f"{fw_rows[L][i]:>12.4f}" for L in LS_WITNESS))
    print("  -> monotone in T; larger L needs larger T; negative = uninformative")
