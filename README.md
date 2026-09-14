# Fidelity witnesses for fermionic Gaussian states

My numerical companion to **arXiv:1703.03152** (Gluza–Kliesch–Eisert–Aolita),
built while working through the paper. The benchmark is theirs: quench
|↑…↑⟩ under the critical transverse-field Ising chain, then certify the result
with the fidelity witness `F_W = 1 − ⟨misplaced fermions⟩ ≤ F`. I first
reproduce their Figure 2, then go past the paper and simulate the
importance-sampling estimator itself — the part they only bound.

Everything is `2L×2L` covariance-matrix algebra; the `2^L` Hilbert space
appears only in brute-force cross-checks. Derivations and conventions live in
my working notes — this file is just a map.

```
pip install -r requirements.txt
python verify.py                  # 7 tests vs exact diagonalization, ~1e-15
python figures.py                 # fig2*  (reproducing the paper, seconds)
python sampling_experiments.py    # fig3*  (testing the estimator, ~5 min)
```

## The plots

**fig2 — what certification costs** (paper's Fig. 2)

- `fig2a` — target correlations `|M_jk|` fill a light cone half the chain
  wide; normalized, this image *is* the measurement schedule.
- `fig2b` — `|M(ρ_t)| ≈ 2.35 L^1.40`, confirming my notes' fit (paper:
  `2.11 L^1.42`) ⇒ shots `∝ L^2.8`.
- `fig2c` — witness vs Trotter steps: `F_W ≈ 2F − 1`, so it dies exactly at
  `F = ½`; larger chains need finer Trotterization.

**fig3 — does the estimator deliver it** (simualtions from the paper's importance sampling results)

- `fig3a` — converges to `F_W` at the closed-form `1/√N` rate.
- `fig3b` — its spread matches a zero-parameter binomial curve: each shot is
  one biased coin flip (`X = ±2|M_t|`).
- `fig3c` — Theorem 2 vs shots actually needed: repeated-trial measurements
  land on the exact-binomial curve, a flat 1.92× below the bound.
- `fig3d` — shot budget `∝ L^2.8`, as claimed.
- `fig3e` — importance vs uniform sampling: only an `~L^0.2` variance win;
  the exponential saving is the mode-space *basis*, not the weighting.
- `fig3f` — the full accept/reject test: a bad state is never accepted, with
  no assumptions; guaranteed acceptance needs the declared noise character
  `n_⊥^max` to carry margin over reality.

## Notes

- Added my notion notes (in pdf format) which is based on my reading of the paper. Please note I use claude Opus 5.0 for verifying my notes and formatting it. 