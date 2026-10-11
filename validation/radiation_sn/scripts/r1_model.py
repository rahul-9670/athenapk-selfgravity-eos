"""R1 reference models for the 0-D multigroup gas-radiation relaxation (uniform periodic box: every ordinate = J).

Code units throughout (T = p/rho in code T units, Tk = T * T_unit in K). Ideal gas: e = rho T/(gamma-1).
Opacity: constant kappa (per mass, code) times the per-group Planck multiplier mP_g(Tk) of the analytic
Semenov-class shape phi = S psi + (1-S), psi = min(1, (nu/nu_break)^beta) (radiation_groups.hpp DustOpacityModel).
In 0-D the Rosseland excess and kappa_s are effective scattering with J = I and drop out exactly.

Models
  exact_ode     continuous system  dE_g/dt = chat chi_g (B_g - E_g),  de/dt = -(c/chat) sum_g chat chi_g (B_g - E_g)
                with INDEPENDENT band means (scipy.quad in nu); integrated with Radau at rtol 1e-12.
  code_step     the code's discrete step (SnSeedSource + SnMatterCoupling): one Planck-weighted Fleck factor,
                (1-f) chi_g returned as SAME-GROUP scattering (inert in 0-D), lagged B_g(T^n), kappa_g(T^n):
                  E_g' = (E_g + chat dt f chi_g B_g)/(1 + chat dt f chi_g);  e' = e + c dt sum_g f chi_g (E_g' - B_g)
  imc_step      Fleck & Cummings multifrequency: same f, but the effectively scattered (1-f) chi_g chat E_g' is
                re-emitted with the emission spectrum w_g = chi_g B_g / sum chi B (a cross-group kernel).
                Solved exactly (small dense linear system per step).
Multipliers: 'replica' = the code's own Simpson construction on its log-T table with linear interpolation
(GroupMultsAtT + GroupOpacityTable::Lookup), 'indep' = scipy.quad at the exact T.
"""
import numpy as np
from scipy.integrate import quad, solve_ivp

H_OVER_K = 4.799243e-11  # s K, the code's constant (radiation_groups.hpp)


def planck_cum(x):
    """Fraction of the Planck energy below x = h nu / kT (independent: quad of x^3/(e^x-1) / (pi^4/15))."""
    if x <= 0:
        return 0.0
    if x > 700:
        return 1.0
    v, _ = quad(lambda t: t ** 3 / np.expm1(t), 0.0, x, limit=200, epsabs=0, epsrel=1e-13)
    return v / (np.pi ** 4 / 15.0)


class Shape:
    def __init__(self, beta, nu_break=1e14, Tlo=1400.0, Thi=1600.0):
        self.beta, self.nub, self.Tlo, self.Thi = beta, nu_break, Tlo, Thi

    def surv(self, Tk):
        return 1.0 if Tk <= self.Tlo else (0.0 if Tk >= self.Thi else (self.Thi - Tk) / (self.Thi - self.Tlo))

    def phi(self, x, Tk):
        if self.beta == 0.0:
            return 1.0
        S = self.surv(Tk)
        xb = H_OVER_K * self.nub / Tk
        psi = min(1.0, (x / (xb + 1e-300)) ** self.beta)
        return S * psi + (1.0 - S)


def mults_indep(shape, edges, Tk):
    """Planck multipliers mP_g = <phi>_{B,g}/<phi>_{B,full} and fractions, by quad (x = h nu/kT)."""
    wB = lambda x: x ** 3 / np.expm1(x) if x > 0 else 0.0
    xb = H_OVER_K * shape.nub / Tk
    def band(a, b, f):
        pts = [p for p in (xb,) if a < p < b]
        return quad(f, a, b, points=pts or None, limit=400, epsabs=0, epsrel=1e-12)[0]
    full = band(1e-12, 200.0, lambda x: shape.phi(x, Tk) * wB(x)) / band(1e-12, 200.0, wB)
    ng = len(edges) - 1
    mP, fr = np.zeros(ng), np.zeros(ng)
    for g in range(ng):
        a = H_OVER_K * edges[g] / Tk
        b = min(H_OVER_K * edges[g + 1] / Tk, 200.0)
        a = max(a, 1e-12)
        fr[g] = planck_cum(b if g < ng - 1 else 1e30) - planck_cum(a if g > 0 else 0.0)
        if b <= a or fr[g] < 1e-300:
            mP[g] = 1.0
            continue
        mP[g] = (band(a, b, lambda x: shape.phi(x, Tk) * wB(x)) / band(a, b, wB)) / full
    return mP, fr


class Replica:
    """The code's GroupMultsAtT (Simpson, NQ = 240, X = 80, band cap a+60) on its log10 T table + Lookup."""
    def __init__(self, shape, edges, Tmin=3.0, Tmax=1e6, nT=256):
        self.shape, self.edges, self.ng = shape, edges, len(edges) - 1
        self.lT0 = np.log10(Tmin)
        self.dlT = (np.log10(Tmax) - self.lT0) / (nT - 1)
        self.nT = nT
        self.W = np.array([self._mults(10 ** (self.lT0 + i * self.dlT)) for i in range(nT)]).T  # [g, iT]

    def _phimean(self, a, b, Tk, ross=False, inv=False):
        a = max(a, 1e-8)
        if b <= a:
            return 1.0
        n = 240
        h = (b - a) / n
        num = den = 0.0
        for k in range(n + 1):
            x = a + k * h
            w = 1.0 if k in (0, n) else (4.0 if k % 2 else 2.0)
            ex = np.exp(-x)
            base = np.exp(a - x) * ((x ** 4 / ((1 - ex) * (1 - ex))) if ross else (x ** 3 / (1 - ex)))
            ph = self.shape.phi(x, Tk)
            if inv:
                ph = 1.0 / ph
            num += w * ph * base
            den += w * base
        return num / (den + 1e-300)

    def _mults(self, Tk):
        full = self._phimean(1e-8, 80.0, Tk)
        out = []
        for g in range(self.ng):
            a = H_OVER_K * self.edges[g] / Tk
            b = (a + 60.0) if g == self.ng - 1 else H_OVER_K * self.edges[g + 1] / Tk
            b = min(b, a + 60.0)
            out.append(self._phimean(a, b, Tk) / (full + 1e-300))
        return out

    def __call__(self, Tk):
        return self.mP(Tk), fractions(self.edges, Tk)

    def mP(self, Tk):
        if self.shape.beta == 0.0:
            return np.ones(self.ng)
        f = (np.log10(Tk) - self.lT0) / self.dlT
        i = int(min(max(int(f), 0), self.nT - 2))
        t = min(max(f - i, 0.0), 1.0)
        return (1 - t) * self.W[:, i] + t * self.W[:, i + 1]


def fractions(edges, Tk):
    """Code's PlanckFraction with an independent cumulative (planck_cum)."""
    ng = len(edges) - 1
    return np.array([max(0.0, planck_cum(1e30 if g == ng - 1 else H_OVER_K * edges[g + 1] / Tk)
                     - planck_cum(H_OVER_K * edges[g] / Tk)) for g in range(ng)])


class Problem:
    def __init__(self, rho, kappa, gamma, arad, c, chat, T_unit, edges, shape):
        self.rho, self.kappa, self.gm1, self.a, self.c, self.chat = rho, kappa, gamma - 1.0, arad, c, chat
        self.Tu, self.edges, self.shape, self.ng = T_unit, edges, shape, len(edges) - 1

    def T_of_e(self, e):
        return self.gm1 * e / self.rho

    def coeffs(self, T, mult):
        Tk = T * self.Tu
        mP, fr = mult(Tk)
        chi = self.rho * self.kappa * np.asarray(mP)
        B = self.a * T ** 4 * np.asarray(fr)
        return chi, B, np.asarray(fr)


def exact_ode(P, e0, E0, t_out, rtol=1e-12):
    cache = {}
    def mult(Tk):
        k = round(Tk, 12)
        if k not in cache:
            cache[k] = mults_indep(P.shape, P.edges, Tk)
        return cache[k]
    def rhs(t, y):
        e, E = y[0], y[1:]
        chi, B, _ = P.coeffs(P.T_of_e(e), mult)
        dE = P.chat * chi * (B - E)
        return np.concatenate([[-(P.c / P.chat) * dE.sum()], dE])
    sol = solve_ivp(rhs, (0.0, t_out[-1]), np.concatenate([[e0], E0]), method="Radau", t_eval=t_out,
                    rtol=rtol, atol=1e-14 * (e0 + np.sum(E0)))
    return sol.y  # [1+ng, nt]


def fleck_f(P, T, chi, fr, dt, alpha):
    cab = np.sum(chi * fr) / max(np.sum(fr), 1e-300)
    dBde = 4.0 * P.a * T ** 3 * P.gm1 / P.rho
    S = P.c * dt * cab * dBde
    return 1.0 / (1.0 + alpha * S), S


def code_step(P, e, E, dt, alpha, mult):
    T = P.T_of_e(e)
    chi, B, fr = P.coeffs(T, mult)
    f, S = fleck_f(P, T, chi, fr, dt, alpha)
    En = (E + P.chat * dt * f * chi * B) / (1.0 + P.chat * dt * f * chi)
    en = e + P.c * dt * np.sum(f * chi * (En - B))
    return en, En, S


def imc_step(P, e, E, dt, alpha, mult):
    T = P.T_of_e(e)
    chi, B, fr = P.coeffs(T, mult)
    f, S = fleck_f(P, T, chi, fr, dt, alpha)
    w = chi * B / max(np.sum(chi * B), 1e-300)
    # (E'_g - E_g)/(chat dt) = f chi_g B_g - chi_g E'_g + (1-f) w_g sum_h chi_h E'_h
    ng = len(E)
    A = np.eye(ng) * (1.0 + P.chat * dt * chi) - P.chat * dt * (1.0 - f) * np.outer(w, chi)
    En = np.linalg.solve(A, E + P.chat * dt * f * chi * B)
    en = e + P.c * dt * np.sum(f * chi * (En - B))
    return en, En, S


def march(step, P, e0, E0, dt, nsteps, alpha, mult, every=1):
    e, E = e0, np.array(E0, float)
    out = [(0.0, e, E.copy())]
    for n in range(1, nsteps + 1):
        e, E, _ = step(P, e, E, dt, alpha, mult)
        if n % every == 0 or n == nsteps:
            out.append((n * dt, e, E.copy()))
    return out


class TabOpacity:
    """The v3 opacity table read and interpolated EXACTLY as the code does (OpacityTable::bilin on
    (log10 rho_cgs, log10 T_K) of the cgs kappa x opacity_unit; GroupOpacityTable::Lookup linear in log10 T_K of
    the per-group multipliers). Used as the coefficient source for the tabulated R1 family: there the ODE reference
    shares the code's coefficients, so it isolates the TIME discretisation (O3 tests the interpolation itself)."""
    def __init__(self, path, opac_unit, rho_unit, edges, log_interp=False):
        self.log_interp = log_interp   # R-F5: the code's radiation/opacity_interp = log
        import struct
        b = open(path, "rb").read()
        off = 8
        ver, flags, ng, nr, nT = struct.unpack_from("<5q", b, off); off += 40
        self.lr0, self.dlr, self.lT0, self.dlT = struct.unpack_from("<4d", b, off); off += 7 * 8
        self.edges_file = np.frombuffer(b, "<f8", ng + 1, off); off += (ng + 1) * 8
        n2 = nr * nT
        kP = np.frombuffer(b, "<f8", n2, off).reshape(nr, nT); off += 3 * n2 * 8   # skip kR, ks
        self.mPt = np.frombuffer(b, "<f8", ng * nT, off).reshape(ng, nT)
        self.kP = kP * (opac_unit if (flags & 1) else 1.0)
        self.nr, self.nT, self.ng, self.rho_unit, self.edges = nr, nT, ng, rho_unit, edges
        assert np.allclose(self.edges_file[1:-1], edges[1:-1], rtol=1e-6), self.edges_file

    def kappaP(self, rho_code, Tk):
        fi = (np.log10(rho_code * self.rho_unit) - self.lr0) / self.dlr
        i = min(max(int(fi), 0), self.nr - 2); ti = min(max(fi - i, 0.0), 1.0)
        fj = (np.log10(Tk) - self.lT0) / self.dlT
        j = min(max(int(fj), 0), self.nT - 2); tj = min(max(fj - j, 0.0), 1.0)
        A = np.log(self.kP) if self.log_interp else self.kP
        v = ((1 - ti) * (1 - tj) * A[i, j] + ti * (1 - tj) * A[i + 1, j] + (1 - ti) * tj * A[i, j + 1]
             + ti * tj * A[i + 1, j + 1])
        return np.exp(v) if self.log_interp else v

    def mP(self, Tk):
        f = (np.log10(Tk) - self.lT0) / self.dlT
        i = min(max(int(f), 0), self.nT - 2); t = min(max(f - i, 0.0), 1.0)
        W = np.log(np.maximum(self.mPt, 1e-300)) if self.log_interp else self.mPt
        v = (1 - t) * W[:, i] + t * W[:, i + 1]
        return np.exp(v) if self.log_interp else v


class TabProblem(Problem):
    """Problem whose gray kappa comes from the table at (rho, T): chi_g = rho kappaP(rho,T) mP_g(T)."""
    def __init__(self, tab, rho, gamma, arad, c, chat, T_unit, edges):
        super().__init__(rho, 1.0, gamma, arad, c, chat, T_unit, edges, Shape(0.0))
        self.tab = tab

    def coeffs(self, T, mult=None):
        Tk = T * self.Tu
        fr = fractions(self.edges, Tk)
        chi = self.rho * self.tab.kappaP(self.rho, Tk) * self.tab.mP(Tk)
        return chi, self.a * T ** 4 * fr, fr


def exact_ode_tab(P, e0, E0, t_out, rtol=1e-12):
    def rhs(t, y):
        e, E = y[0], y[1:]
        chi, B, _ = P.coeffs(P.T_of_e(e))
        dE = P.chat * chi * (B - E)
        return np.concatenate([[-(P.c / P.chat) * dE.sum()], dE])
    sol = solve_ivp(rhs, (0.0, t_out[-1]), np.concatenate([[e0], E0]), method="Radau", t_eval=t_out,
                    rtol=rtol, atol=1e-14 * (e0 + np.sum(E0)))
    return sol.y
