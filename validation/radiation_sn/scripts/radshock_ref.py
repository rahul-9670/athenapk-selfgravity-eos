#!/usr/bin/env python
"""R8b reference: steady gray NON-EQUILIBRIUM-DIFFUSION radiative shock (the Lowrie & Edwards 2008, Shock Waves 18, 129
model: Eddington factor 1/3, Fick flux), solved here independently as a phase-plane problem in (E, F).

Units = the AthenaPK deck's code units: upstream rho0 = 1, sound speed a0 = 1 (p0 = 1/gamma, code temperature
T = p/rho, T0 = 1/gamma), E = a_code T^4 with a_code = P0 / T0^4 (so E0 = P0 = a_R T0^4 / (rho0 a0^2), LE08's P0),
c_code = C = c/a0, absorption per MASS kappa (sigma = kappa rho; LE08 use a constant sigma per length -- the per-mass
choice matches the code's `constant` opacity model; at rho0 = 1 the upstream values coincide). Shock frame, upstream
u0 = M0 > 0 flowing in +x.
Conservation integrals (m = rho u, Pi, Phi from the upstream equilibrium):
  full   (O(u/c) mixed-frame diffusion, LE08):  m u + p + E/3 = Pi,  m (h + u^2/2) + (4/3) u E + F = Phi,
                                               F' + [(4/3) u E]' = u (E/3)' - C sigma (E - a T^4)
  static (what the code solves: radiation force off, no Doppler/advection of radiation):
                                               m u + p = Pi,  m (h + u^2/2) + F = Phi,  F' = -C sigma (E - a T^4)
  both:  E' = -3 sigma F / C;  h = gamma/(gamma-1) p/rho.
Given (E, F), u solves a quadratic (two roots: supersonic upstream branch, subsonic downstream branch); T = p/rho.
Upstream (E0, 0) and downstream (E1, 0) equilibria are saddles; the precursor is the unstable manifold of the upstream
point integrated in +x on the supersonic root, the relaxation region the manifold of the downstream point integrated in
-x on the subsonic root. Subcritical shocks: the two curves cross in the (E, F) plane at the embedded hydrodynamic shock
(E, F continuous, u jumps between the roots); x = 0 there.
usage: radshock_ref.py check     (prints the downstream state vs the closed-form RH system and the profile landmarks)
"""
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

GAM = 5.0 / 3.0


class Shock:
    def __init__(self, M0, P0, C, kappa, full=True, gam=GAM):
        self.g, self.M0, self.P0, self.C, self.k, self.full = gam, M0, P0, C, kappa, full
        self.T0 = 1.0 / gam
        self.a = P0 / self.T0 ** 4
        self.m = M0
        p0 = 1.0 / gam
        E0 = P0
        self.E0 = E0
        self.Pi = self.m * M0 + p0 + (E0 / 3.0 if full else 0.0)
        self.Phi = self.m * (gam / (gam - 1) * p0 + 0.5 * M0 ** 2) + ((4.0 / 3.0) * M0 * E0 if full else 0.0)

    def disc(self, E, F):
        g, m = self.g, self.m
        A = m * (0.5 - g / (g - 1))
        B = g / (g - 1) * (self.Pi - (E / 3.0 if self.full else 0.0)) + ((4.0 / 3.0) * E if self.full else 0.0)
        return A, B, B * B - 4 * A * (F - self.Phi)

    def roots(self, E, F):
        A, B, D = self.disc(E, F)
        if D < 0:
            return None
        s = np.sqrt(D)
        r = sorted([(-B + s) / (2 * A), (-B - s) / (2 * A)])
        return r            # [subsonic (smaller u), supersonic (larger u)]

    def state(self, E, F, branch):
        r = self.roots(E, F)
        if r is None:
            return None
        u = r[branch]
        p = self.Pi - self.m * u - (E / 3.0 if self.full else 0.0)
        rho = self.m / u
        return u, rho, p / rho

    def rhs(self, x, y, branch):
        self.ncall = getattr(self, "ncall", 0) + 1
        if self.ncall > 400000:
            raise RuntimeError("integration did not finish in 4e5 rhs calls (stiff/singular manifold)")
        E, F = y
        st = self.state(E, F, branch)
        if st is None:
            return [np.nan, np.nan]
        u, rho, T = st
        sig = self.k * rho
        S = self.C * sig * (E - self.a * T ** 4)
        dE = -3.0 * sig * F / self.C
        if not self.full:
            return [dE, -S]
        h = 1e-7 * max(abs(E), 1e-30)
        hf = 1e-7 * max(abs(F), abs(self.Phi) * 1e-6)
        uE = (self.state(E + h, F, branch)[0] - self.state(E - h, F, branch)[0]) / (2 * h)
        uF = (self.state(E, F + hf, branch)[0] - self.state(E, F - hf, branch)[0]) / (2 * hf)
        dF = (-S - dE * (u + (4.0 / 3.0) * E * uE)) / (1.0 + (4.0 / 3.0) * E * uF)
        return [dE, dF]

    def downstream(self):
        """Downstream equilibrium: F = 0, E1 = a T1^4 on the subsonic root (scan + bisection in E)."""
        def f(E):
            st = self.state(E, 0.0, 0)
            return st[2] - (E / self.a) ** 0.25
        Es = self.E0 * np.logspace(0.01, 3, 4000)
        v = [f(E) if self.roots(E, 0.0) is not None else np.nan for E in Es]
        for i in range(len(Es) - 1):
            if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] * v[i + 1] < 0:
                return brentq(f, Es[i], Es[i + 1], xtol=1e-15 * Es[i], rtol=1e-14)
        raise RuntimeError("no downstream equilibrium")

    def manifold(self, E, branch, direction, xspan, eps=1e-7):
        y0 = np.array([E, 0.0])
        J = np.zeros((2, 2))
        for j in range(2):
            d = np.zeros(2)
            d[j] = 1e-6 * (E if j == 0 else self.Phi * 1e-3)
            J[:, j] = (np.array(self.rhs(0, y0 + d, branch)) - np.array(self.rhs(0, y0 - d, branch))) / (2 * d[j])
        lam, V = np.linalg.eig(J)
        i = int(np.argmax(lam)) if direction > 0 else int(np.argmin(lam))
        v = np.real(V[:, i])
        v = v if v[0] * (1 if branch == 1 else -1) > 0 else -v      # E increases into the shock from upstream
        start = y0 + eps * E * v / abs(v[0])
        f = lambda x, y: self.rhs(x, y, branch)
        D0 = self.disc(*y0)[2]
        sonic = lambda x, y: self.disc(*y)[2] - 1e-6 * D0          # stop before the sonic point (no root beyond)
        sonic.terminal = True
        Emax = 2.0 * max(self.E0, getattr(self, "E1", E))
        past = lambda x, y: (y[0] - Emax) if branch == 1 else (y[0] - 1.001 * self.E0)   # left the shock's E range
        past.terminal = True
        sol = solve_ivp(f, (0.0, direction * xspan), start, method="LSODA", rtol=1e-11, atol=[1e-14 * E, 1e-14 * E],
                        dense_output=True, events=[sonic, past], max_step=xspan / 5000)
        return sol

    def solve(self, xspan=0.05):
        self.E1 = self.downstream()
        up = self.manifold(self.E0, 1, +1, xspan)
        dn = self.manifold(self.E1, 0, -1, xspan)
        # crossing: F_up(E) = F_dn(E) on the common E range
        Eu, Fu = up.y
        Ed, Fd = dn.y
        lo, hi = max(Eu.min(), Ed.min()), min(Eu.max(), Ed.max())
        grid = np.linspace(lo, hi, 20001)
        ou, od = np.argsort(Eu), np.argsort(Ed)
        dF = np.interp(grid, Eu[ou], Fu[ou]) - np.interp(grid, Ed[od], Fd[od])
        k = np.nonzero(np.sign(dF[1:]) != np.sign(dF[:-1]))[0]
        if k.size == 0:
            raise RuntimeError("no embedded-shock crossing (continuous shock or failed integration)")
        k = k[0]
        Es = grid[k] - dF[k] * (grid[k + 1] - grid[k]) / (dF[k + 1] - dF[k])
        xu = np.interp(Es, Eu[ou], up.t[ou])
        xd = np.interp(Es, Ed[od], dn.t[od])
        self.up, self.dn, self.Es, self.xu, self.xd = up, dn, Es, xu, xd
        return self

    def profile(self, x):
        """T_gas, T_rad, rho at positions x (shock at x = 0; upstream x < 0)."""
        out = np.zeros((3, len(x)))
        for n, xx in enumerate(x):
            if xx < 0:
                xs = self.xu + xx
                if xs <= self.up.t[0]:
                    E, F, br = self.E0, 0.0, 1
                else:
                    E, F = self.up.sol(xs)
                    br = 1
            else:
                xs = self.xd + xx
                if xs >= self.dn.t[0]:
                    E, F, br = self.E1, 0.0, 0
                else:
                    E, F = self.dn.sol(xs)
                    br = 0
            u, rho, T = self.state(E, F, br)
            out[:, n] = T, (E / self.a) ** 0.25, rho
        return out


def rh_closed_form(M0, P0, gam=GAM):
    """Downstream (rho1, T1/T0) from the LE08 jump conditions (FML17 eqs. as quoted: T normalised by T0, full model)."""
    def eqs(T1):
        # momentum quadratic in rho1: 3 T rho^2 + [g P0 (T^4 - 1) - 3 (g M^2 + 1)] rho + 3 g M^2 = 0
        a, b, c = 3 * T1, gam * P0 * (T1 ** 4 - 1) - 3 * (gam * M0 ** 2 + 1), 3 * gam * M0 ** 2
        d = b * b - 4 * a * c
        if d < 0:
            return None
        return [(-b + s * np.sqrt(d)) / (2 * a) for s in (1, -1)]
    def resid(T1):
        best = None
        for r in eqs(T1) or []:
            e = (6 * T1 - 3 * (gam - 1) * M0 ** 2 - 6 - 8 * (gam - 1) * P0) * r ** 2 + 8 * (gam - 1) * P0 * T1 ** 4 * r \
                + 3 * (gam - 1) * M0 ** 2
            if r > 1.0001 and (best is None or abs(e) < abs(best[0])):
                best = (e, r)
        return best
    Ts = np.linspace(1.0001, 20, 200000)
    vals = [resid(T) for T in Ts]
    for i in range(len(Ts) - 1):
        if vals[i] and vals[i + 1] and vals[i][0] * vals[i + 1][0] < 0:
            T1 = brentq(lambda T: resid(T)[0], Ts[i], Ts[i + 1], xtol=1e-14)
            return resid(T1)[1], T1
    raise RuntimeError


if __name__ == "__main__":
    if sys.argv[1] == "check":
        for M0 in (1.2, 2.0, 3.0):
            for full in (True, False):
                s = Shock(M0, 1e-4, 1.0e3, 577.35, full=full)
                E1 = s.downstream()
                st = s.state(E1, 0.0, 0)
                line = f"M0 {M0} {'full  ' if full else 'static'}: rho1 {st[1]:.8f} T1/T0 {st[2] / s.T0:.8f}"
                if full:
                    r1, T1 = rh_closed_form(M0, 1e-4)
                    line += f" | closed-form RH: rho1 {r1:.8f} T1/T0 {T1:.8f}"
                try:
                    s.solve()
                    xs = np.linspace(-0.02, 0.02, 4001)
                    T, Tr, rho = s.profile(xs)
                    i0 = np.argmin(np.abs(xs))
                    line += (f" | Tmax/T0 {T.max() / s.T0:.5f} theta(0)/T0 {Tr[i0] / s.T0:.5f} T_p/T0 "
                             f"{s.state(s.Es, np.interp(s.Es, *[s.up.y[0], s.up.y[1]]), 1)[2] / s.T0:.5f}")
                except Exception as e:  # noqa: BLE001
                    line += f" | profile: {e}"
                print(line)
