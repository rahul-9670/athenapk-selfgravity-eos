#!/usr/bin/env python
"""A10 reference: steady single-fluid ambipolar C-shock, perpendicular field B = By(x) y, adiabatic gamma, eta_A = Q By^2
(diffusion/ambipolar_coeff = fixed). Shock frame, gas flows +x. Conserved: m = rho v; Pi = m v + P + By^2/2;
F = m v^2/2 + gamma/(gamma-1) P v + v0 B0 By (Poynting flux -E_z By with E_z = -v0 B0 everywhere); induction:
Q By^2 dBy/dx = v By - v0 B0. Given By, momentum + energy give v from a quadratic (root continuous with v0).
Lab set-up (pgen iprob 45): uniform (rho0, +u, P0, B0) hitting a reflecting wall at x = L; the shock runs left at speed V;
shock-frame upstream speed v0 = u + V, downstream v1 = V (gas at rest). V from the ideal-MHD jump conditions.
usage: cshock_ref.py rho0 P0 B0 u Q gamma   -> prints V, downstream state, width, C/J type; importable (profile())."""
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
def consts(rho0, v0, P0, B0, g):
    m = rho0 * v0; Pi = m * v0 + P0 + 0.5 * B0**2; F = 0.5 * m * v0**2 + g / (g - 1) * P0 * v0 + v0 * B0 * B0
    return m, Pi, F
def v_of_B(B, m, Pi, F, v0, B0, g, vprev):
    a = m * (0.5 - g / (g - 1)); b = g / (g - 1) * (Pi - 0.5 * B**2); c = v0 * B0 * B - F
    disc = b * b - 4 * a * c
    if disc < 0: return np.nan
    r = np.array([(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)])
    return r[np.argmin(np.abs(r - vprev))]
def downstream(rho0, v0, P0, B0, g):
    """ideal perpendicular MHD jump: compression r = v0/v1 = B1/B0 (nontrivial root)."""
    m, Pi, F = consts(rho0, v0, P0, B0, g)
    def res(r):
        v1 = v0 / r; B1 = B0 * r; P1 = Pi - m * v1 - 0.5 * B1**2
        return 0.5 * m * v1**2 + g / (g - 1) * P1 * v1 + v0 * B0 * B1 - F
    rs = np.linspace(1.0001, (g + 1) / (g - 1) - 1e-6, 4000); f = [res(r) for r in rs]
    for i in range(len(rs) - 1):
        if np.sign(f[i]) != np.sign(f[i + 1]): return brentq(res, rs[i], rs[i + 1])
    return np.nan
def lab_to_shock(rho0, P0, B0, u, g):
    """V such that the downstream gas is at rest in the lab: v0 - v1 = u with v0 = u + V."""
    def res(V):
        r = downstream(rho0, u + V, P0, B0, g); return (u + V) - (u + V) / r - u if np.isfinite(r) else -1.0
    Vs = np.linspace(1e-3, 50 * u, 3000); f = [res(V) for V in Vs]
    for i in range(len(Vs) - 1):
        if np.sign(f[i]) != np.sign(f[i + 1]): return brentq(res, Vs[i], Vs[i + 1])
    raise RuntimeError('no shock speed')
def profile(rho0, P0, B0, u, Q, g, eps=1e-6, npts=4000):
    """steady profile in the shock frame: x (relative), By, v, rho, P; also r, V, ctype."""
    V = lab_to_shock(rho0, P0, B0, u, g); v0 = u + V; r = downstream(rho0, v0, P0, B0, g); B1 = B0 * r
    m, Pi, F = consts(rho0, v0, P0, B0, g)
    st = {'v': v0}
    def rhs(x, y):
        B = y[0]; v = v_of_B(B, m, Pi, F, v0, B0, g, st['v'])
        if not np.isfinite(v): return [np.nan]
        st['v'] = v; return [(v * B - v0 * B0) / (Q * B * B)]
    ev = lambda x, y: y[0] - (B0 + (1 - 1e-6) * (B1 - B0)); ev.terminal = True
    sol = solve_ivp(rhs, [0, 1e3 * Q * B1**2 / v0], [B0 * (1 + eps)], events=ev, max_step=Q * B0**2 / v0 / 50, rtol=1e-10, atol=1e-14)
    B = sol.y[0]; vv = []; st['v'] = v0
    for b in B: vv.append(v_of_B(b, m, Pi, F, v0, B0, g, st['v'])); st['v'] = vv[-1]
    vv = np.array(vv); ctype = 'C' if np.all(np.isfinite(vv)) and sol.status == 1 else 'J (needs a sub-shock)'
    rho = m / vv; P = Pi - m * vv - 0.5 * B**2
    return dict(x=sol.t, B=B, v=vv, rho=rho, P=P, V=V, v0=v0, r=r, B1=B1, ctype=ctype)
if __name__ == '__main__':
    rho0, P0, B0, u, Q, g = map(float, sys.argv[1:7])
    p = profile(rho0, P0, B0, u, Q, g)
    cs, vA = np.sqrt(g * P0 / rho0), B0 / np.sqrt(rho0)
    i10, i90 = [np.argmin(np.abs(p['B'] - (B0 + f * (p['B1'] - B0)))) for f in (0.1, 0.9)]
    print(f"c_s {cs:.3f} v_A {vA:.3f} c_f {np.hypot(cs, vA):.3f} | V {p['V']:.4f} v0 {p['v0']:.4f} M_f {p['v0']/np.hypot(cs, vA):.3f} "
          f"r {p['r']:.4f} | type {p['ctype']} | 10-90 width {p['x'][i90]-p['x'][i10]:.4f} (Q B0^2/v0 = {Q*B0**2/p['v0']:.4f})")
