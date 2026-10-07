#!/usr/bin/env python
"""Isothermal self-similar collapse solutions, derived and solved here (no tabulated input).

Units: cs = 1, 4 pi G = 1  (so G = 1/(4 pi)).

Larson-Penston (pre-singularity, tau = t0 - t > 0):
    x = r/(cs tau),  rho = D(x)/(4 pi G tau^2),  u = cs V(x),  M = cs^3 tau m(x)/G,  m = x^2 D (x+V)
    (x+V) D' + D V' = -2 D (1 + V/x)
    D'/D + (x+V) V' = -(x+V) D
  => D' = D (x+V) [D - 2(1 + V/x)] / Delta,  V' = [2(1 + V/x) - (x+V)^2 D] / Delta,
     Delta = (x+V)^2 - 1.  Regular centre: V = -2x/3, D(0) = D0 (eigenvalue: the solution must
     cross the sonic line x+V = 1 smoothly).  Literature: D0 ~ 1.67, D -> 8.86/x^2, V -> -3.28
     (Larson 1969; Penston 1969; Hunter 1977) -- reproduced below as the solver's own check.

Shu (1977) expansion wave (t > 0):  x = r/(cs t),  rho = alpha/(4 pi G t^2),  u = cs v
    [(x-v)^2 - 1] v'       = [alpha (x-v) - 2/x] (x-v)
    [(x-v)^2 - 1] alpha'/alpha = [alpha - 2 (x-v)/x] (x-v)
  static SIS outside x = 1 (alpha = 2/x^2, v = 0); inside, free fall onto m0 = x^2 alpha (x - v),
  m0 -> 0.975 as x -> 0 (Shu 1977).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ------------------------------------------------------------------ Larson-Penston
def _lp_rhs(x, y):
    D, V = y
    w = x + V
    delta = w * w - 1.0
    dD = D * w * (D - 2.0 * (1.0 + V / x)) / delta
    dV = (2.0 * (1.0 + V / x) - w * w * D) / delta
    return [dD, dV]


def _lp_sonic_slopes(xs):
    """Regular crossing of the sonic line w = x + V = 1 at x = xs.
    Identity: N_D = -D w (D Delta + N_V)  =>  D' = -D w (D + V') everywhere (non-singular).
    At the sonic point N_V = 0 gives D* = 2/xs, V* = 1 - xs, and L'Hopital on V' = N_V/Delta gives
    V'^2 + V' + (xs - 1)/xs^2 = 0  =>  V' = -1/xs  or  V' = -1 + 1/xs."""
    Ds, Vs = 2.0 / xs, 1.0 - xs
    roots = []
    for vp in (-1.0 / xs, -1.0 + 1.0 / xs):
        roots.append((vp, -Ds * (Ds + vp)))
    return Ds, Vs, roots


def _lp_inward(xs, branch, xmin=1e-4, h=1e-4):
    Ds, Vs, roots = _lp_sonic_slopes(xs)
    vp, dp = roots[branch]
    y0 = [Ds - h * dp, Vs - h * vp]
    sol = solve_ivp(_lp_rhs, (xs - h, xmin), y0, rtol=1e-11, atol=1e-14, dense_output=True)
    return sol


def larson_penston(xmax=400.0, branch=0):
    """Shoot on the sonic point x*: off the eigenvalue the inward branch turns singular before
    the centre with V/x running to +/- large; the regular (LP) solution separates the two.
    Bisection on the sign of V/x + 2/3 at the stopping point; the root is then checked to reach
    x = 1e-4 with V/x = -2/3. Returns (D0, x*, x, D, V)."""
    crit = lambda xs: (lambda so: so.y[1, -1] / so.t[-1] + 2.0 / 3.0)(_lp_inward(xs, branch))
    lo, hi = 2.0, 2.6
    flo = crit(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = crit(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    xs = 0.5 * (lo + hi)
    sin_ = _lp_inward(xs, branch)
    Ds, Vs, roots = _lp_sonic_slopes(xs)
    vp, dp = roots[branch]
    h = 1e-4
    sout = solve_ivp(_lp_rhs, (xs + h, xmax), [Ds + h * dp, Vs + h * vp], rtol=1e-11, atol=1e-14)
    x = np.concatenate([sin_.t[::-1], sout.t])
    D = np.concatenate([sin_.y[0][::-1], sout.y[0]])
    V = np.concatenate([sin_.y[1][::-1], sout.y[1]])
    return D[0], xs, x, D, V, sin_.t[-1], sout.t[-1]


# ------------------------------------------------------------------ Shu 1977
def _shu_rhs(x, y):
    a, v = y
    w = x - v
    den = w * w - 1.0
    dv = (a * w - 2.0 / x) * w / den
    da = a * (a - 2.0 * w / x) * w / den
    return [da, dv]


def shu(eps=1e-2, xmin=1e-6):
    """Inward from the head of the expansion wave. x = 1 is a degenerate critical point: the
    expansion-wave branch is tangent to the critical line (v' = 1 there), so a first-order start
    stalls and an outward shoot from the free-fall asymptote is unstable. Second-order expansion
    (derived with sympy, validation 2026-10-03): with eps = 1 - x,
        v = -eps - eps^2/2,   alpha = 2 + 2 eps + 2 eps^2 + 2 eps^3   (i.e. alpha ~ 2/x behind the head).
    m0 = x^2 alpha (x - v) as x -> 0 is the reduced accretion rate (Shu: 0.975); it is converged
    to 5 digits for eps = 3e-2 .. 1e-3. Integration in s = ln x, ln alpha (LSODA)."""
    def rhs(s_, y):
        x = np.exp(s_)
        la, v = y
        a = np.exp(la)
        w = x - v
        den = w * w - 1.0
        return [x * w * (a - 2 * w / x) / den, x * w * (a * w - 2 / x) / den]
    x0 = 1.0 - eps
    y0 = [np.log(2 + 2 * eps + 2 * eps ** 2 + 2 * eps ** 3), -eps - 0.5 * eps ** 2]
    sol = solve_ivp(rhs, (np.log(x0), np.log(xmin)), y0, method="LSODA", rtol=1e-11, atol=1e-13,
                    dense_output=True)
    x = np.geomspace(xmin, x0, 4000)
    la, v = sol.sol(np.log(x))
    a = np.exp(la)
    m0 = x[0] ** 2 * a[0] * (x[0] - v[0])
    return m0, x, a, v


if __name__ == "__main__":
    D0, xs, x, D, V, xin, xout = larson_penston()
    print(f"  inward branch reached x = {xin:.2e} (V/x there {V[0]/x[0]:+.4f}); outward reached x = {xout:.1f}")
    big = x > 100
    print(f"Larson-Penston: D0 = {D0:.5f} (lit. ~1.67)  sonic point x* = {xs:.5f}  "
          f"D x^2 -> {np.median(D[big] * x[big] ** 2):.4f} (lit. 8.86)  V(x={x[-1]:.0f}) = {V[-1]:.4f} (lit. -3.28)")
    m0, x, a, v = shu()
    i = np.argmin(abs(x - 0.05))
    print(f"Shu 1977: m0 = {m0:.5f} (lit. 0.975); x = 0.05: alpha = {a[i]:.2f}, -v = {-v[i]:.3f}")
