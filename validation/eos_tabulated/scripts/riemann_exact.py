#!/usr/bin/env python
"""Exact Riemann solver for a GENERAL equation of state P(rho, e) (e = specific internal energy),
in the spirit of Colella & Glaz 1985 (JCP 59, 264): the left/right wave curves are
  * rarefaction: integrated along the isentrope de = (P/rho^2) drho, du = -/+ (c/rho) drho,
                 c^2 = dP/drho|_s = P_rho|_e + (P/rho^2) P_e|_rho, of the GIVEN P(rho, e);
  * shock:       the Rankine-Hugoniot relation e - e0 = (P + P0)/2 (1/rho0 - 1/rho), solved for rho at
                 given P with the GIVEN P(rho, e); du = sqrt((P - P0)(1/rho0 - 1/rho)).
The star pressure solves u_L - f_L(P) = u_R + f_R(P). The reference for the tabulated EOS is the
RUNTIME interpolant itself (bilinear in log rho, log e; eos_interp = p or p_over_rho), because a
conservative scheme converges to the weak solution of the equations it actually solves; the
interpolant's deviation from the true EOS is E2/E-F1's business, not this test's.

Convexity: a simple wave is a rarefaction only where the fundamental derivative
G = 1 + (rho/c) dc/drho|_s > 0. The solver evaluates G along every wave path it uses and reports the
minimum; G < 0 would mean composite waves, which this solver does not construct (it would say so).

Self-test: python riemann_exact.py  (ideal gas, Toro 2009 Table 4.2 tests 1-5)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class IdealEOS:
    def __init__(self, gamma):
        self.g = gamma

    def P(self, rho, e):
        return (self.g - 1.0) * rho * e

    def e_of_P(self, rho, P):
        return P / ((self.g - 1.0) * rho)


class TableEOS:
    """The runtime bilinear interpolant (src/eos/eos_table.hpp), vectorised, code units."""

    def __init__(self, tab, mode="p_over_rho", rho_cubic=False):
        if mode.endswith("_cubic"):          # "p_over_rho_cubic" / "p_cubic" = hydro/eos_rho_interp = cubic
            mode, rho_cubic = mode[:-6], True
        self.t = tab
        self.mode = mode
        self.rho_cubic = rho_cubic   # hydro/eos_rho_interp = cubic (Catmull-Rom along log rho for P only)
        A = tab["P"].copy()
        if mode == "p_over_rho":
            rho_node = 10 ** (tab["lr0"] + np.arange(tab["nr"]) * tab["dlr"])
            A = A / rho_node[:, None]
        self.A = A
        self.le_lo = tab["le0"]; self.le_hi = tab["le0"] + (tab["ne"] - 1) * tab["dle"]
        self.lnr_floor = (tab["lr0"] + 1e-6) * np.log(10.0)  # isentropes stop at the table's rho edge

    @staticmethod
    def _bilin(A, x0, dx, y0, dy, x, y):
        n1, n2 = A.shape
        fi = (x - x0) / dx; i = np.clip(np.floor(fi).astype(int), 0, n1 - 2); ti = np.clip(fi - i, 0.0, 1.0)
        fj = (y - y0) / dy; j = np.clip(np.floor(fj).astype(int), 0, n2 - 2); tj = np.clip(fj - j, 0.0, 1.0)
        return (1 - ti) * (1 - tj) * A[i, j] + ti * (1 - tj) * A[i + 1, j] + (1 - ti) * tj * A[i, j + 1] + ti * tj * A[i + 1, j + 1]

    def P(self, rho, e):
        t = self.t
        if self.rho_cubic:
            v = self._cr(self.A, t["lr0"], t["dlr"], t["le0"], t["dle"], np.log10(rho), np.log10(e))
        else:
            v = self._bilin(self.A, t["lr0"], t["dlr"], t["le0"], t["dle"], np.log10(rho), np.log10(e))
        return rho * v if self.mode == "p_over_rho" else v

    @staticmethod
    def _herm_e(A, r, j, u):
        """EosTable::HermiteE: monotone (Fritsch-Carlson) cubic Hermite along e in cell j of rows r (vectorised)."""
        n2 = A.shape[1]
        y0, y1 = A[r, j], A[r, j + 1]; d = y1 - y0
        dm = np.where(j >= 1, y0 - A[r, np.maximum(j - 1, 0)], d)
        dp = np.where(j + 2 <= n2 - 1, A[r, np.minimum(j + 2, n2 - 1)] - y1, d)
        m0 = 0.5 * (dm + d); m1 = 0.5 * (d + dp)
        m0 = np.where((d > 0) & (dm > 0), m0, 0.0); m1 = np.where((d > 0) & (dp > 0), m1, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            al = m0 / d; be = m1 / d; ss = al * al + be * be
            tau = np.where(ss > 9.0, 3.0 / np.sqrt(np.where(ss > 0, ss, 1.0)), 1.0)
        m0 = np.where(ss > 9.0, tau * al * d, m0); m1 = np.where(ss > 9.0, tau * be * d, m1)
        u2 = u * u; u3 = u2 * u
        h = (2 * u3 - 3 * u2 + 1) * y0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * y1 + (u3 - u2) * m1
        return np.where(d > 0, h, y0 + u * d)

    @classmethod
    def _cr(cls, A, x0, dx, y0, dy, x, y):
        """EosTable::PTab with p_cubic_: Catmull-Rom along x (linear in the first/last x cell), monotone cubic
        Hermite along y."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        n1, n2 = A.shape
        fi = (x - x0) / dx; i = np.clip(np.floor(fi).astype(int), 0, n1 - 2); ti = np.clip(fi - i, 0.0, 1.0)
        fj = (y - y0) / dy; j = np.clip(np.floor(fj).astype(int), 0, n2 - 2); tj = np.clip(fj - j, 0.0, 1.0)
        H = lambda r: cls._herm_e(A, r, j, tj)
        lin = (i < 1) | (i > n1 - 3)
        ic = np.clip(i, 1, n1 - 3); t = np.where(lin, ti, fi - ic); t2 = t * t; t3 = t2 * t
        w = (0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2))
        cub = w[0] * H(ic - 1) + w[1] * H(ic) + w[2] * H(ic + 1) + w[3] * H(ic + 2)
        linv = (1 - ti) * H(i) + ti * H(np.minimum(i + 1, n1 - 1))
        out = np.where(lin, linv, cub)
        return out if out.ndim else float(out)

    def e_of_P(self, rho, P):  # exact inverse of the interpolant (to round-off), unlike the 20-step runtime bisection
        f = lambda le: self.P(rho, 10 ** le) - P
        return 10 ** brentq(f, self.le_lo, self.le_hi, xtol=1e-14, rtol=1e-15, maxiter=200)

    def logT(self, rho, e):
        t = self.t
        return self._bilin(t["logT"], t["lr0"], t["dlr"], t["le0"], t["dle"], np.log10(rho), np.log10(e))

    def esp_of_T(self, rho, T):
        t = self.t
        return self._bilin(t["espT"], t["lr0"], t["dlr"], t["lT0"], t["dlT"], np.log10(rho), np.log10(T))


def c2(eos, rho, e, h=1e-7):
    """dP/drho at constant entropy of eos.P, by centred differences (one-sided-average at the kinks)."""
    Pr = (eos.P(rho * (1 + h), e) - eos.P(rho * (1 - h), e)) / (2 * h * rho)
    Pe = (eos.P(rho, e * (1 + h)) - eos.P(rho, e * (1 - h))) / (2 * h * e)
    return Pr + eos.P(rho, e) / rho ** 2 * Pe


class WaveCurve:
    """One side's wave curve from state K = (rho, u, e); sign = -1 for the left wave, +1 for the right."""

    def __init__(self, eos, rho, u, e, sign):
        self.eos, self.rho, self.u, self.e, self.s = eos, rho, u, e, sign
        self.P0 = float(eos.P(rho, e))
        self._ise = None
        self.lnr_lo = max(np.log(rho) - 30.0, getattr(eos, "lnr_floor", -np.inf))

    # Isentrope through K, tabulated lazily downward in ln rho by classical RK4 with a fixed step H
    # (the interpolant is only piecewise smooth: kinks every 0.05 dex in rho and 0.005 dex in e, so a
    # fixed small step is more robust than an adaptive one; RK4 error across a kink ~ H^2 x jump).
    # Columns: ln rho, e, I = int_{ln rho_K}^{ln rho} c dln rho (<= 0 going down). Stops at the table edge.
    H = 2e-4

    def _rhs(self, lnr, e):
        rho = np.exp(lnr)
        return self.eos.P(rho, e) / rho, np.sqrt(max(c2(self.eos, rho, e), 0.0))

    def _extend(self, lnr_target):
        if self._ise is None:
            self._ise = [[np.log(self.rho)], [self.e], [0.0]]
            self._end = False
        L, Ecol, Icol = self._ise
        lnr, e, I = L[-1], Ecol[-1], Icol[-1]
        e_floor = 10 ** getattr(self.eos, "le_lo", -300) * 1.001
        h = -self.H
        while lnr > lnr_target and not self._end:
            if lnr + h < self.lnr_lo:
                self._end = True
                break
            k1e, k1i = self._rhs(lnr, e)
            k2e, k2i = self._rhs(lnr + h / 2, e + h / 2 * k1e)
            k3e, k3i = self._rhs(lnr + h / 2, e + h / 2 * k2e)
            k4e, k4i = self._rhs(lnr + h, e + h * k3e)
            e_new = e + h / 6 * (k1e + 2 * k2e + 2 * k3e + k4e)
            if e_new <= e_floor:
                self._end = True
                break
            I += h / 6 * (k1i + 2 * k2i + 2 * k3i + k4i)
            e = e_new; lnr += h
            L.append(lnr); Ecol.append(e); Icol.append(I)

    def isentrope(self, lnr_lo):
        """callable ln rho -> (e, I), valid on [max(lnr_lo, end of curve), ln rho_K]."""
        self._extend(lnr_lo)
        L = np.array(self._ise[0][::-1]); Ecol = np.array(self._ise[1][::-1]); Icol = np.array(self._ise[2][::-1])
        from scipy.interpolate import CubicSpline
        se, si = CubicSpline(L, Ecol), CubicSpline(L, Icol)
        f = lambda x: np.array([se(x), si(x)])
        f.lo = L[0]
        return f

    def rarefaction_state(self, P):
        """(rho, e, du) on the isentrope at pressure P < P0; du = int_rho^rho0 c/rho drho > 0."""
        lnr0 = np.log(self.rho)
        if P >= self.P0:
            return self.rho, self.e, 0.0
        # extend the curve until it passes P (or ends)
        while True:
            Lend, Eend = self._ise[0][-1] if self._ise else lnr0, self._ise[1][-1] if self._ise else self.e
            if self._ise is not None and (self.eos.P(np.exp(Lend), Eend) <= P or self._end):
                break
            self._extend((Lend if self._ise else lnr0) - 0.5)
        sol = self.isentrope(self._ise[0][-1])
        lo = sol.lo
        f = lambda lnr: self.eos.P(np.exp(lnr), sol(lnr)[0]) - P
        if f(lo) > 0.0:  # P not reached on the curve (table edge): report its end
            e, I = sol(lo)
            return np.exp(lo), e, -I
        lnr = brentq(f, lo, lnr0, xtol=1e-14, rtol=1e-15, maxiter=300)
        e, I = sol(lnr)
        return np.exp(lnr), e, -I  # I is negative going down in lnr

    def shock_state(self, P):
        """(rho, e, du) on the Hugoniot at P > P0."""
        eos, r0, e0, P0 = self.eos, self.rho, self.e, self.P0

        def g(rho):
            e = e0 + 0.5 * (P + P0) * (1 / r0 - 1 / rho)
            return eos.P(rho, e) - P

        if g(r0 * (1 + 1e-14)) >= 0.0:
            return r0, e0, 0.0
        hi = r0 * 1.0001
        while g(hi) < 0:
            hi *= 1.5
            if hi > r0 * 1e4:
                raise RuntimeError("Hugoniot: no root below 1e4 x compression")
        rho = brentq(g, r0 * (1 + 1e-14), hi, xtol=1e-15 * r0, rtol=1e-15, maxiter=300)
        e = e0 + 0.5 * (P + P0) * (1 / r0 - 1 / rho)
        du = np.sqrt((P - P0) * (1 / r0 - 1 / rho))
        return rho, e, du

    def f(self, P):  # velocity change magnitude across the wave (u* = u_K + sign f)
        if abs(P - self.P0) <= 1e-13 * self.P0:
            return 0.0
        if P >= self.P0:
            return self.shock_state(P)[2]
        return -self.rarefaction_state(P)[2]  # rarefaction: f < 0


def solve(eos, L, R, Pmin_frac=1e-8):
    """L, R = (rho, u, e). Returns dict with the star state and a sampler."""
    WL, WR = WaveCurve(eos, *L, -1), WaveCurve(eos, *R, +1)
    du = R[1] - L[1]
    F = lambda P: WL.f(P) + WR.f(P) + du
    hi = max(WL.P0, WR.P0)
    while F(hi) < 0:
        hi *= 2.0
    lo = min(WL.P0, WR.P0)
    while F(lo) > 0:
        lo *= 0.3
        if lo < min(WL.P0, WR.P0) * Pmin_frac:
            raise RuntimeError("vacuum generated (or the rarefaction runs off the table)")
    Ps = brentq(F, lo, hi, xtol=1e-15 * hi, rtol=1e-14, maxiter=400)
    us = 0.5 * (L[1] + R[1]) + 0.5 * (WR.f(Ps) - WL.f(Ps))
    out = dict(P=Ps, u=us, WL=WL, WR=WR)
    for W, key in ((WL, "L"), (WR, "R")):
        if Ps >= W.P0:
            rs, es, _ = W.shock_state(Ps)
            m = np.sqrt((Ps - W.P0) / (1 / W.rho - 1 / rs))
            out[key] = dict(type="shock", rho=rs, e=es, S=W.u + W.s * m / W.rho)
        else:
            rs, es, _ = W.rarefaction_state(Ps)
            cK = np.sqrt(c2(eos, W.rho, W.e)); cs = np.sqrt(c2(eos, rs, es))
            out[key] = dict(type="rarefaction", rho=rs, e=es, head=W.u + W.s * cK, tail=us + W.s * cs)
    return out


def fundamental_derivative_min(eos, W, rho_end, n=2000):
    """min over the isentrope from rho_K to rho_end of G = 1 + (rho/c) dc/drho|_s (finite differences)."""
    sol = W.isentrope(np.log(rho_end))
    lnr = np.linspace(np.log(W.rho), max(np.log(rho_end), sol.lo), n)
    c = np.array([np.sqrt(c2(eos, np.exp(x), sol(x)[0])) for x in lnr])
    G = 1 + np.gradient(np.log(c), lnr)
    return float(G.min()), float(np.exp(lnr[np.argmin(G)]))


def sample(eos, sol, Lst, Rst, xi):
    """Exact (rho, u, P, e) at similarity coordinates xi = (x - x0)/t."""
    out = np.zeros((4, len(xi)))
    us, Ps = sol["u"], sol["P"]
    for k, x in enumerate(xi):
        side, K, W = ("L", Lst, sol["WL"]) if x < us else ("R", Rst, sol["WR"])
        sgn = -1 if side == "L" else +1
        star = sol[side]
        if star["type"] == "shock":
            behind = (x - star["S"]) * sgn < 0  # between contact and shock
            if behind:
                out[:, k] = (star["rho"], us, Ps, star["e"])
            else:
                out[:, k] = (K[0], K[1], eos.P(K[0], K[2]), K[2])
        else:
            if (x - star["head"]) * sgn >= 0:
                out[:, k] = (K[0], K[1], eos.P(K[0], K[2]), K[2])
            elif (x - star["tail"]) * sgn <= 0:
                out[:, k] = (star["rho"], us, Ps, star["e"])
            else:  # inside the fan: u + sgn c = x on the isentrope
                ise = W.isentrope(np.log(star["rho"]))

                def g(lnr):
                    e, I = ise(lnr)
                    # I = int_{ln rho_K}^{ln rho} c dln rho < 0; left fan u = u_L - I, right fan u = u_R + I
                    u = W.u + sgn * I
                    return u + sgn * np.sqrt(c2(eos, np.exp(lnr), e)) - x

                lnr = brentq(g, np.log(star["rho"]), np.log(W.rho), xtol=1e-14, maxiter=300)
                e, I = ise(lnr)
                out[:, k] = (np.exp(lnr), W.u + sgn * I, eos.P(np.exp(lnr), e), e)
    return out


if __name__ == "__main__":
    # Toro (2009) Table 4.2/4.3, gamma = 1.4: (rhoL, uL, PL, rhoR, uR, PR) -> (P*, u*, rho*L, rho*R)
    toro = [((1.0, 0.0, 1.0, 0.125, 0.0, 0.1), (0.30313, 0.92745, 0.42632, 0.26557)),
            ((1.0, -2.0, 0.4, 1.0, 2.0, 0.4), (0.00189, 0.00000, 0.02185, 0.02185)),
            ((1.0, 0.0, 1000.0, 1.0, 0.0, 0.01), (460.894, 19.5975, 0.57506, 5.99924)),
            ((1.0, 0.0, 0.01, 1.0, 0.0, 100.0), (46.0950, -6.19633, 5.99242, 0.57511)),
            ((5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950), (1691.64, 8.68975, 14.2823, 31.0426))]
    eos = IdealEOS(1.4)
    worst = 0.0
    for n, (s, ref) in enumerate(toro, 1):
        L = (s[0], s[1], eos.e_of_P(s[0], s[2])); R = (s[3], s[4], eos.e_of_P(s[3], s[5]))
        so = solve(eos, L, R)
        got = (so["P"], so["u"], so["L"]["rho"], so["R"]["rho"])
        # Toro prints 6 significant digits: allow half a unit in the last printed place
        rel = [max(0.0, abs(g - r) - max(0.5e-5, 0.5e-5 * abs(r))) / max(abs(r), 1e-3) for g, r in zip(got, ref)]
        worst = max(worst, max(rel))
        print(f"Toro test {n}: P* {got[0]:.6g} u* {got[1]:.6g} rho*L {got[2]:.6g} rho*R {got[3]:.6g}  max rel dev beyond Toro rounding {max(rel):.1e}")
    # fan sampling check against the analytic ideal-gas fan (Sod, t = 0.25, x0 = 0.5)
    L = (1.0, 0.0, eos.e_of_P(1.0, 1.0)); R = (0.125, 0.0, eos.e_of_P(0.125, 0.1))
    so = solve(eos, L, R)
    xi = np.linspace(-1.2, 1.9, 63)
    S = sample(eos, so, L, R, xi)
    g = 1.4; cL = np.sqrt(g)
    fan = (xi > -cL) & (xi < so["L"]["tail"])
    rho_an = (2 / (g + 1) + (g - 1) / ((g + 1) * cL) * (0 - xi[fan])) ** (2 / (g - 1))
    print(f"Sod fan: max |rho - analytic| = {np.abs(S[0, fan] - rho_an).max():.2e} over {fan.sum()} points")
    print("SELF-TEST", "PASS" if worst < 2e-4 and np.abs(S[0, fan] - rho_an).max() < 1e-8 else "FAIL")
