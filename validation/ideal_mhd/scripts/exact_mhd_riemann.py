#!/usr/bin/env python
"""Exact solver for the 1D ideal-MHD Riemann problem, gamma-law gas, Bx != 0, REGULAR solutions
(fast, rotational, slow waves on each side + contact; no compound/intermediate waves). ideal_mhd I3 (D1 (a)).

Units: Heaviside-Lorentz (AthenaPK): P_mag = B^2/2, v_A = B/sqrt(rho).
Formulation (in the spirit of Torrilhon 2002; Takahashi & Yamada 2014 -- the approach, not their code):
  unknowns  z = (b_L1, b_R1, b_*, theta_*): |B_t| behind the left / right fast wave, |B_t| at the contact, and
            the common direction of B_t at the contact. The rotational discontinuities turn B_t (|B_t| fixed)
            from the fast-wave direction to theta_*; fast and slow waves keep the direction (coplanar).
  equations u, v_y, v_z, P continuous across the contact (B_t continuous by construction).
Fast/slow SHOCK with given downstream b2 (upstream state 1, V = 1/rho, m = rho (u - s) signed):
  induction + tangential momentum  m^2 (b2 V2 - b1 V1) = Bx^2 (b2 - b1)
  normal momentum                  m^2 (V2 - V1) + (P2 - P1) + (b2^2 - b1^2)/2 = 0
  energy (MHD Hugoniot)            e2 - e1 + (P1 + P2)/2 (V2 - V1) + (V2 - V1)(b2 - b1)^2 / 4 = 0,  e = P V/(g-1)
  root V2 in ((g-1)/(g+1) V1, V1); fast: m^2 V >= Bx^2 on both sides, slow: m^2 V <= Bx^2 on both sides.
RAREFACTION: RK4 in b along the primitive eigenvector of lambda = u + c (c signed, < 0 for left-going):
  du/db = (c^2 - ca^2)/(b c), drho = rho du / c, dP = rho a^2 du / c, dv_t = -Bx db / (rho c).
ROTATIONAL: v_t2 = v_t1 + kappa (B_t2 - B_t1)/sqrt(rho), kappa = sign(Bx) (left-going) / -sign(Bx) (right-going),
  from the Alfven eigenvectors in pgen linear_wave_mhd.cpp (columns 1 and 5).
Admissible shock root: inside ((g-1)/(g+1) V1, V1), on the m^2 > 0 side of the pole V2 = b1 V1/b2, |f| ~ 0.
Self-checks (selftest()): every shock satisfies F(W2) - F(W1) = s (U2 - U1) with the FULL conservative MHD flux
to 1e-10 relative; the entropy P/rho^g is constant through every rarefaction fan; contact residual < 1e-11.
"""
import numpy as np
from scipy.optimize import fsolve, brentq


class State:
    __slots__ = ("rho", "u", "vt", "P", "Bt")

    def __init__(self, rho, u, vt, P, Bt):
        self.rho, self.u, self.vt, self.P, self.Bt = float(rho), float(u), np.array(vt, float), float(P), np.array(Bt, float)

    def copy(self):
        return State(self.rho, self.u, self.vt, self.P, self.Bt)

    def prim(self, Bx):
        return np.array([self.rho, self.u, self.vt[0], self.vt[1], self.P, Bx, self.Bt[0], self.Bt[1]])


class Riemann:
    def __init__(self, WL, WR, Bx, gamma, nfan=400):
        self.WL, self.WR, self.Bx, self.g, self.nfan = WL, WR, float(Bx), float(gamma), nfan

    # ---- wave speeds --------------------------------------------------------------------------------
    def speeds(self, rho, P, b):
        a2 = self.g * P / rho
        ca2 = self.Bx**2 / rho
        bt2 = b * b / rho
        s = a2 + ca2 + bt2
        d = np.sqrt(max(s * s - 4 * a2 * ca2, 0.0))
        return a2, ca2, 0.5 * (s + d), max(0.5 * (s - d), 0.0)

    # ---- fast / slow wave ---------------------------------------------------------------------------
    def fs_wave(self, W1, that, b2, fam, sig):
        """W1 -> W2 through a fast (fam 'f') or slow ('s') wave of direction sig (-1 left-going, +1 right-going),
        downstream |B_t| = b2 along that. Returns (W2, piece)."""
        b1 = float(W1.Bt @ that)
        shock = (b2 > b1) if fam == "f" else (b2 < b1)
        if abs(b2 - b1) < 1e-14:
            return W1.copy(), ("none",)
        return self.shock(W1, that, b1, b2, fam, sig) if shock else self.raref(W1, that, b1, b2, fam, sig)

    def shock(self, W1, that, b1, b2, fam, sig):
        g, Bx = self.g, self.Bx
        V1, P1 = 1.0 / W1.rho, W1.P

        def parts(V2):
            m2 = Bx**2 * (b2 - b1) / (b2 * V2 - b1 * V1)
            P2 = P1 - 0.5 * (b2**2 - b1**2) - m2 * (V2 - V1)
            return m2, P2

        def f(V2):
            m2, P2 = parts(V2)
            return P2 * V2 / (g - 1) - P1 * V1 / (g - 1) + 0.5 * (P1 + P2) * (V2 - V1) + 0.25 * (V2 - V1) * (b2 - b1) ** 2

        Vmin = (g - 1) / (g + 1) * V1
        # m^2 > 0 needs b2 V2 - b1 V1 of the sign of b2 - b1: for a fast shock (b2 > b1) V2 > b1 V1 / b2. f changes
        # sign across that POLE of m^2 without a zero (found 2026-10-12: a "second root" with m^2 = -8e10), so the
        # bracket starts above it and every root must also make |f| small.
        if b2 > b1:
            Vmin = max(Vmin, b1 * V1 / b2)
        grid = Vmin + (V1 - Vmin) * (1 - np.logspace(-12, 0, 4000)[::-1] + 1e-12)
        grid = np.unique(np.clip(grid, Vmin * (1 + 1e-12), V1 * (1 - 1e-12)))
        fv = np.array([f(v) for v in grid])
        cand = []
        for i in np.nonzero(np.sign(fv[:-1]) * np.sign(fv[1:]) < 0)[0]:
            try:
                V2 = brentq(f, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15)
            except ValueError:
                continue
            m2, P2 = parts(V2)
            if m2 <= 0 or P2 <= 0 or abs(f(V2)) > 1e-10 * (P1 * V1 + abs(P2) * V2):
                continue
            ok = (m2 * V1 >= Bx**2 * (1 - 1e-9) and m2 * V2 >= Bx**2 * (1 - 1e-9)) if fam == "f" else \
                 (m2 * V1 <= Bx**2 * (1 + 1e-9) and m2 * V2 <= Bx**2 * (1 + 1e-9))
            if ok:
                cand.append((V2, m2, P2))
        if len(cand) != 1:
            raise ValueError(f"{fam}-shock b {b1:.6g}->{b2:.6g}: {len(cand)} admissible roots")
        V2, m2, P2 = cand[0]
        m = -sig * np.sqrt(m2)          # left-going: s < u1 -> m > 0
        s = W1.u - m * V1
        W2 = State(1.0 / V2, W1.u + m * (V2 - V1), W1.vt + that * Bx * (b2 - b1) / m, P2, W1.Bt + that * (b2 - b1))
        return W2, ("shock", s, W1, W2)

    def raref(self, W1, that, b1, b2, fam, sig):
        Bx, n = self.Bx, self.nfan
        y = np.array([W1.rho, W1.u, float(W1.vt @ that), W1.P])
        vperp = W1.vt - that * float(W1.vt @ that)

        def rhs(b, y):
            rho, u, vt, P = y
            a2, ca2, cf2, cs2 = self.speeds(rho, P, b)
            c = sig * np.sqrt(cf2 if fam == "f" else cs2)
            du = (c * c - ca2) / (b * c)
            return np.array([rho * du / c, du, -Bx / (rho * c), rho * a2 * du / c]), c

        path = []
        bs = np.linspace(b1, b2, n + 1)
        for k in range(n + 1):
            _, c = rhs(bs[k], y)
            path.append((y[1] + c, y.copy(), bs[k]))
            if k == n:
                break
            h = bs[k + 1] - bs[k]
            k1 = rhs(bs[k], y)[0]
            k2 = rhs(bs[k] + h / 2, y + h / 2 * k1)[0]
            k3 = rhs(bs[k] + h / 2, y + h / 2 * k2)[0]
            k4 = rhs(bs[k] + h, y + h * k3)[0]
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        mk = lambda yy, bb: State(yy[0], yy[1], vperp + that * yy[2], yy[3], W1.Bt + that * (bb - b1))
        fan = [(lam, mk(yy, bb)) for lam, yy, bb in path]
        return fan[-1][1], ("fan", fan)

    def rotate(self, W1, theta, sig):
        bt = np.linalg.norm(W1.Bt)
        Bt2 = bt * np.array([np.cos(theta), np.sin(theta)])
        kap = -sig * np.sign(self.Bx)
        W2 = State(W1.rho, W1.u, W1.vt + kap * (Bt2 - W1.Bt) / np.sqrt(W1.rho), W1.P, Bt2)
        return W2, ("shock", W1.u + sig * abs(self.Bx) / np.sqrt(W1.rho), W1, W2)

    # ---- one side -----------------------------------------------------------------------------------
    def side(self, W, b1, bstar, theta, sig):
        that0 = W.Bt / np.linalg.norm(W.Bt)
        W1, p1 = self.fs_wave(W, that0, b1, "f", sig)
        W2, p2 = self.rotate(W1, theta, sig)
        tstar = np.array([np.cos(theta), np.sin(theta)])
        W3, p3 = self.fs_wave(W2, tstar, bstar, "s", sig)
        return W3, [p1, p2, p3]

    def residual(self, z):
        bL, bR, bs, th = z
        try:
            L, _ = self.side(self.WL, bL, bs, th, -1)
            R, _ = self.side(self.WR, bR, bs, th, +1)
        except ValueError:
            return np.full(4, 1e3)
        sc = max(abs(self.WL.u), abs(self.WR.u), 1.0)
        return np.array([(L.u - R.u) / sc, (L.vt[0] - R.vt[0]) / sc, (L.vt[1] - R.vt[1]) / sc,
                         (L.P - R.P) / max(self.WL.P, self.WR.P)])

    def solve(self, z0):
        z, info, ier, msg = fsolve(self.residual, z0, full_output=True, xtol=1e-14)
        self.z = z
        self.res = np.abs(self.residual(z)).max()
        L, pl = self.side(self.WL, z[0], z[2], z[3], -1)
        R, pr = self.side(self.WR, z[1], z[2], z[3], +1)
        self.starL, self.starR, self.pl, self.pr = L, R, pl, pr
        return z, self.res

    # ---- sampling at xi = x/t -----------------------------------------------------------------------
    def sample(self, xi):
        """primitive vector (rho, u, vy, vz, P, Bx, By, Bz) at xi = (x - x0)/t."""
        # left waves in order, contact, right waves in reverse order (with before/after swapped)
        seq = list(self.pl)
        seq.append(("shock", 0.5 * (self.starL.u + self.starR.u), self.starL, self.starR))
        for p in reversed(self.pr):
            if p[0] == "shock":
                seq.append(("shock", p[1], p[3], p[2]))
            elif p[0] == "fan":
                seq.append(("fan", list(reversed(p[1]))))
        cur = self.WL
        for p in seq:
            if p[0] == "none":
                continue
            if p[0] == "shock":
                if xi < p[1]:
                    return cur.prim(self.Bx)
                cur = p[3]
            else:
                fan = p[1]
                lam = np.array([q[0] for q in fan])
                if xi < lam[0]:
                    return cur.prim(self.Bx)
                if xi <= lam[-1]:
                    k = min(np.searchsorted(lam, xi), len(lam) - 1)
                    k = max(k, 1)
                    w = (xi - lam[k - 1]) / (lam[k] - lam[k - 1]) if lam[k] > lam[k - 1] else 1.0
                    return (1 - w) * fan[k - 1][1].prim(self.Bx) + w * fan[k][1].prim(self.Bx)
                cur = fan[-1][1]
        return cur.prim(self.Bx)

    def waves(self):
        out = []
        for side, ps in (("L", self.pl), ("R", self.pr)):
            for name, p in zip(("fast", "rot", "slow"), ps):
                if p[0] == "shock":
                    out.append((side, name, "jump", p[1]))
                elif p[0] == "fan":
                    out.append((side, name, "fan", p[1][0][0], p[1][-1][0]))
        out.append(("C", "contact", "jump", self.starL.u))
        return out

    # ---- self-test ----------------------------------------------------------------------------------
    def flux(self, W):
        g, Bx = self.g, self.Bx
        rho, u, (vy, vz), P, (By, Bz) = W.rho, W.u, W.vt, W.P, W.Bt
        B2 = Bx**2 + By**2 + Bz**2
        v2 = u * u + vy * vy + vz * vz
        E = P / (g - 1) + 0.5 * rho * v2 + 0.5 * B2
        vB = u * Bx + vy * By + vz * Bz
        U = np.array([rho, rho * u, rho * vy, rho * vz, E, By, Bz])
        F = np.array([rho * u, rho * u * u + P + 0.5 * B2 - Bx**2, rho * u * vy - Bx * By, rho * u * vz - Bx * Bz,
                      (E + P + 0.5 * B2) * u - Bx * vB, By * u - Bx * vy, Bz * u - Bx * vz])
        return U, F

    def selftest(self):
        worst_rh, worst_s = 0.0, 0.0
        for p in self.pl + self.pr:
            if p[0] == "shock":
                U1, F1 = self.flux(p[2])
                U2, F2 = self.flux(p[3])
                r = np.abs((F2 - F1) - p[1] * (U2 - U1)).max() / max(np.abs(F1).max(), np.abs(U1).max(), 1.0)
                worst_rh = max(worst_rh, r)
            elif p[0] == "fan":
                S = np.array([q[1].P / q[1].rho**self.g for q in p[1]])
                worst_s = max(worst_s, np.abs(S / S[0] - 1).max())
        return {"RH_rel": worst_rh, "fan_entropy_rel": worst_s, "contact_res": self.res}


# ---- standard problems (HL units; Gaussian B/sqrt(4 pi) = HL B) ----------------------------------------------
S4P = 1.0 / np.sqrt(4 * np.pi)
PROBLEMS = {
    # Ryu & Jones 1995 Fig. 2a (all seven waves), gamma 5/3, t = 0.2 on [0,1], x0 = 0.5 (from memory: check)
    "RJ2a": dict(gamma=5 / 3, t=0.2, Bx=2 * S4P,
                 L=State(1.08, 1.2, [0.01, 0.5], 0.95, [3.6 * S4P, 2 * S4P]),
                 R=State(1.0, 0.0, [0.0, 0.0], 1.0, [4 * S4P, 2 * S4P])),
    # RJ2a states with a DIVERGING normal flow (u_L = -0.6, u_R = +0.6): rarefactions instead of shocks -- this
    # project's own variant, to exercise the fan integration against the numerical solution (no literature value).
    "RJ2a_div": dict(gamma=5 / 3, t=0.2, Bx=2 * S4P,
                     L=State(1.08, -0.6, [0.01, 0.5], 0.95, [3.6 * S4P, 2 * S4P]),
                     R=State(1.0, 0.6, [0.0, 0.0], 1.0, [4 * S4P, 2 * S4P])),
}


def build(name, nfan=400):
    p = PROBLEMS[name]
    rs = Riemann(p["L"], p["R"], p["Bx"], p["gamma"], nfan=nfan)
    bL, bR = np.linalg.norm(p["L"].Bt), np.linalg.norm(p["R"].Bt)
    thL, thR = np.arctan2(*p["L"].Bt[::-1]), np.arctan2(*p["R"].Bt[::-1])
    best = None
    for fL in (1.0, 1.1, 1.3, 0.9):
        for fR in (1.0, 1.1, 1.3, 0.9):
            for fs in (0.8, 1.0, 1.2):
                z0 = [fL * bL, fR * bR, fs * 0.5 * (bL + bR), 0.5 * (thL + thR)]
                try:
                    z, res = rs.solve(z0)
                except Exception:
                    continue
                if best is None or res < best[1]:
                    best = (z, res)
                if res < 1e-11:
                    break
            if best and best[1] < 1e-11:
                break
        if best and best[1] < 1e-11:
            break
    rs.solve(best[0])
    return rs, p


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "RJ2a"
    rs, p = build(name)
    print(name, "z =", rs.z, "residual", rs.res)
    print("selftest", rs.selftest())
    for w in rs.waves():
        print(" ", w)
    for nm, W in (("L*", rs.starL), ("R*", rs.starR)):
        print(nm, "rho %.6f u %.6f vy %.6f vz %.6f P %.6f By %.6f Bz %.6f" % (W.rho, W.u, *W.vt, W.P, *W.Bt))
