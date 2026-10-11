#!/usr/bin/env python
"""R2 reference: the Su & Olson (1997) non-equilibrium transport benchmark, solved semi-analytically.

Problem (Su & Olson, Ann. Nucl. Energy 24, 1035 (1997); definition as restated in arXiv:2011.01543 Sec. III):
infinite homogeneous slab, gray, purely absorbing (c_s = 0), C_v = alpha T^3, eps = 4a/alpha, z = sigma x,
tau = eps c sigma t, isotropic source S = 1 for |z| <= x0 = 0.5 and tau <= tau0 = 10. In these units
    eps d(psi_n)/dtau + mu_n d(psi_n)/dz + psi_n = (V + S)/2,     U = sum_n wh_n psi_n  (sum wh = 2),
    dV/dtau = U - V,
U = radiation energy density / (a T_H^4), V = a T^4 / (a T_H^4).

Method (independent of the code): Fourier series on a periodic box of length Lp >> the wave extent; for each wave
number k the ordinates + V form a constant-coefficient linear ODE y' = M_k y + s_k (source on) solved EXACTLY in tau by
the matrix exponential of the augmented matrix [[M, s], [0, 0]] (scipy.linalg.expm); after tau0 the source is off.
Angular set: any (mu_n, wh_n) -- the code's own slab cosines (to isolate its spatial/time error) or Gauss-Legendre
S_N (to measure the angular error). Spatial: spectral; returns CELL AVERAGES on a given grid (exact sinc factor), so
it compares like for like with a finite-volume code and does not suffer Gibbs at |z| = x0 in the averages.
"""
import numpy as np
from scipy.linalg import expm


def gauss_slab(n):
    mu, w = np.polynomial.legendre.leggauss(n)
    return mu, w            # sum w = 2


def code_slab(nzeta=4, npsi=8):
    """The code's product grid (radiation_sn_angles.hpp BuildAngularGrid) seen by a slab along x1: mu = n_x."""
    m, wm = np.polynomial.legendre.leggauss(nzeta)
    dphi = 2 * np.pi / npsi
    mu, w = [], []
    for iz in range(nzeta):
        s = np.sqrt(max(0.0, 1 - m[iz] ** 2))
        for ip in range(npsi):
            phi = (ip + 0.5) * dphi
            mu.append(s * np.cos(phi))
            w.append(wm[iz] * dphi / (2 * np.pi))     # 4 pi normalisation -> sum = 2
    return np.array(mu), np.array(w)


def solve(taus, zc, h, mu, wh, eps=1.0, x0=0.5, tau0=10.0, Lp=80.0, nk=4096):
    """Cell averages of U and V on cells centred at zc with width h, at each tau in taus. Returns (U, V) [ntau, nz]."""
    N = len(mu)
    ks = 2 * np.pi * np.arange(nk + 1) / Lp
    U = np.zeros((len(taus), len(zc)))
    V = np.zeros((len(taus), len(zc)))
    for ik, k in enumerate(ks):
        Sk = (1.0 / Lp) * (2 * np.sin(k * x0) / k if k > 0 else 2 * x0)       # coefficient of e^{ikz}
        M = np.zeros((N + 1, N + 1), complex)
        M[:N, :N] = np.diag(-(1.0 + 1j * k * mu)) / eps
        M[:N, N] = 0.5 / eps
        M[N, :N] = wh
        M[N, N] = -1.0
        s = np.zeros(N + 1, complex)
        s[:N] = 0.5 * Sk / eps
        A = np.zeros((N + 2, N + 2), complex)
        A[:N + 1, :N + 1] = M
        A[:N + 1, N + 1] = s
        y0 = np.zeros(N + 2, complex)
        y0[N + 1] = 1.0
        # state at tau0 (source on from 0)
        y_t0 = expm(A * tau0) @ y0
        avg = (np.sin(k * h / 2) / (k * h / 2)) if k > 0 else 1.0
        phase = np.exp(1j * k * zc) * avg
        fac = 1.0 if k == 0 else 2.0                                          # +k and -k (real, even solution)
        for it, t in enumerate(taus):
            y = (expm(A * t) @ y0) if t <= tau0 else np.concatenate([expm(M * (t - tau0)) @ y_t0[:N + 1], [0.0]])
            Uk = wh @ y[:N]
            Vk = y[N]
            U[it] += fac * np.real(Uk * phase)
            V[it] += fac * np.real(Vk * phase)
    return U, V


if __name__ == "__main__":
    # self-check: convergence in Lp / nk and the S_N angular convergence, at a few (z, tau)
    zc = np.array([0.01, 0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0])
    taus = [0.1, 1.0, 3.16228, 10.0]
    for name, (mu, w) in (("S8", gauss_slab(8)), ("S32", gauss_slab(32)), ("code4x8", code_slab())):
        U, V = solve(taus, zc, 1e-3, mu, w)
        print(name)
        for it, t in enumerate(taus):
            print(f"  tau={t:8.5f} U " + " ".join(f"{u:.5f}" for u in U[it]) + " | V " + " ".join(f"{v:.5f}" for v in V[it]))
