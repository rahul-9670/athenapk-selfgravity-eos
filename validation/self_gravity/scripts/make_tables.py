#!/usr/bin/env python
"""Initial-state tables for the profile1d problem generator (literature tests L1-L10).

Columns: coordinate, rho, P [, v]. Every hydrostatic table is checked here: the residual of
dP/dr + rho G M(<r)/r^2 (spherical) or dP/dz + rho dphi/dz (planar) relative to dP/dr must be
< 1e-6, else the script aborts. Output: ../inputs/tables/<name>.txt plus a JSON of the parameters
the analysis needs.
"""
import json
import os

import numpy as np
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.optimize import brentq

OUT = os.path.join(os.path.dirname(__file__), "..", "inputs", "tables")
os.makedirs(OUT, exist_ok=True)
META = {}


def write(name, cols, header, meta):
    np.savetxt(os.path.join(OUT, name + ".txt"), np.column_stack(cols), header=header, fmt="%.16e")
    META[name] = meta
    print(f"  wrote {name}.txt ({len(cols[0])} rows)  {meta}")


def check_spherical(name, r, rho, P, G, r_lo=None, r_hi=None):
    M = 4 * np.pi * cumulative_simpson(rho * r ** 2, x=r, initial=0.0)
    dP = np.gradient(P, r, edge_order=2)
    g = rho * G * M / np.maximum(r, 1e-300) ** 2
    sel = (r > (r_lo if r_lo else r[5])) & (r < (r_hi if r_hi else r[-5]))
    res = np.max(np.abs(dP[sel] + g[sel])) / np.max(np.abs(dP[sel]))
    print(f"  {name}: hydrostatic residual max|dP/dr + rho g| / max|dP/dr| = {res:.2e}")
    assert res < 1e-5, name
    return M


# ------------------------------------------------------------------ L1 polytrope + atmosphere
def polytrope_atmosphere():
    """Mullen, Hanawa & Gammie 2021 Sec. 3.3/4.3: gamma = 2 (n = 1) body, G = M = R = 1, atmosphere
    M(r) = M + Ma (1 - (r/R)^-2), Ma = 1e-2 M  ->  rho_atm = Ma R^2 / (2 pi r^5). Atmosphere pressure
    from hydrostatic integration inward from r_out; the body's pressure is K rho^2 + P_atm(R)
    (a constant offset keeps it hydrostatic and makes P continuous)."""
    G, M, R, Ma = 1.0, 1.0, 1.0, 1e-2
    rhoc = np.pi / 4.0 * M / R ** 3   # M = 4 pi rho_c a^3 * pi, a = R/pi
    K = 2 * np.pi * G * R ** 2 / np.pi ** 2
    r_out = 7.0  # beyond the corner of the [-4R, 4R]^3 box
    r = np.concatenate([np.linspace(0, R, 4001), np.geomspace(R * (1 + 1e-9), r_out, 6000)])
    rho = np.where(r <= R, rhoc * np.sinc(r / R), Ma * R ** 2 / (2 * np.pi * np.maximum(r, R) ** 5))
    Mr = np.where(r <= R, 4 * rhoc * R ** 3 / np.pi ** 2 * (np.sin(np.pi * r / R) - np.pi * r / R * np.cos(np.pi * r / R)),
                  M + Ma * (1 - (R / np.maximum(r, R)) ** 2))
    # P_atm(r) = int_r^inf rho G M(r')/r'^2 dr' in CLOSED FORM (2026-10-03: a cumulative Simpson across the
    # 1e-9-wide interval at the surface discontinuity gave a 100x wrong integration constant, P(R) = 0.028
    # instead of 2.7e-4, and the 3D run faithfully breathed by 45 %):
    #   rho G M/r^2 = (G Ma R^2/2pi) [ (M+Ma) r^-7 - Ma R^2 r^-9 ]
    out = r > R
    ra, rha = r[out], rho[out]
    Patm = lambda rr: G * Ma * R ** 2 / (2 * np.pi) * ((M + Ma) / (6 * rr ** 6) - Ma * R ** 2 / (8 * rr ** 8))
    P_atm = Patm(ra) + 1e-14
    P_R = Patm(R)
    P = np.empty_like(r)
    P[out] = P_atm
    P[~out] = K * rho[~out] ** 2 + P_R
    integrand = rha * G * (M + Ma * (1 - (R / ra) ** 2)) / ra ** 2
    check_spherical("polytrope_atm (body)", r[r < R], rho[r < R], P[r < R], G, r_hi=0.98)
    assert abs(Mr[r <= R][-1] - M) < 1e-6, Mr[r <= R][-1]
    dP = np.gradient(P_atm, ra, edge_order=2)
    res = np.max(np.abs(dP[5:-5] + integrand[5:-5])) / np.max(np.abs(dP[5:-5]))
    jump = abs(P[~out][-1] - P_atm[0]) / P_atm[0]
    print(f"  polytrope_atm (atm): residual {res:.2e}; pressure continuity at R: |P(R-) - P(R+)|/P = {jump:.2e} (P(R) = {P_R:.4e})")
    assert res < 1e-4 and jump < 1e-3
    tau = R / np.sqrt(2 * G * M / R)
    write("L1_polytrope_atm", [r, rho, P], "r rho P   (G=M=R=1, four_pi_G = 4 pi, gamma = 2)",
          {"four_pi_G": 4 * np.pi, "gamma": 2.0, "rhoc": rhoc, "K": K, "Ma": Ma, "tau": tau, "R": R,
           "P_surf": float(P_R)})


# ------------------------------------------------------------------ L2 Spitzer sheet
def spitzer_sheet():
    """Mullen et al. 2021 Sec. 4.1: polytropic sheet P = K rho^Gamma, Gamma = gamma = 1.2, G = K = 1,
    mean density 0.3, period Lz = 4 (Jeans swindle). h + phi = C with h = K Gamma/(Gamma-1) rho^(Gamma-1);
    phi'' = 4 pi G (rho - rho_bar); symmetric about the box centre. C from the mean-density condition."""
    G, K, Gam, rbar, L = 1.0, 1.0, 1.2, 0.3, 4.0
    def rho_of(phi, C):
        h = np.maximum(C - phi, 0.0)
        return (h * (Gam - 1) / (K * Gam)) ** (1 / (Gam - 1))
    def shoot(C, dense=False):
        f = lambda z, y: [y[1], 4 * np.pi * G * (rho_of(y[0], C) - rbar)]
        return solve_ivp(f, (0, L / 2), [0.0, 0.0], rtol=1e-12, atol=1e-14, dense_output=dense)
    def mean_err(C):
        s = shoot(C, True)
        z = np.linspace(0, L / 2, 20001)
        return np.trapezoid(rho_of(s.sol(z)[0], C), z) / (L / 2) - rbar
    C = brentq(mean_err, 1e-6, 50.0, xtol=1e-14)
    s = shoot(C, True)
    z = np.linspace(0, L / 2, 40001)
    phi, dphi = s.sol(z)
    rho = rho_of(phi, C)
    P = K * rho ** Gam
    print(f"  spitzer: C = {C:.10f}  contrast rho_max/rho_min = {rho.max() / rho.min():.3e}  dphi(L/2) = {dphi[-1]:.2e}")
    dP = np.gradient(P, z, edge_order=2)
    res = np.max(np.abs(dP + rho * dphi)) / np.max(np.abs(dP))
    print(f"  spitzer: hydrostatic residual {res:.2e}")
    assert res < 1e-5
    # table over one period on [0, L], sheet centred at L/2 (the profile is even about the centre)
    q = np.concatenate([L / 2 - z[::-1], L / 2 + z[1:]])
    r2 = np.concatenate([rho[::-1], rho[1:]])
    p2 = np.concatenate([P[::-1], P[1:]])
    write("L2_spitzer", [q, r2, p2], "x rho P   (planar, period 4, centred at 2; G = K = 1, Gamma = 1.2)",
          {"four_pi_G": 4 * np.pi, "gamma": Gam, "rho_bar": rbar, "L": L, "contrast": float(rho.max() / rho.min())})


# ------------------------------------------------------------------ L3 n = 5 polytrope
def plummer_polytrope():
    """Hanawa & Mullen 2025 Sec. 4.4: rho = rho_c (1 + r^2/(3 R^2))^(-5/2), P = P_c (1 + r^2/(3R^2))^(-3),
    rho_c = R = 1, 4 pi G = 1. Hydrostatic for P_c = 4 pi G rho_c^2 R^2 / 6 (= 1/6 here; checked)."""
    G = 1 / (4 * np.pi)
    r = np.linspace(0, 30, 300001)
    rho = (1 + r ** 2 / 3) ** -2.5
    Pc = 4 * np.pi * G / 6
    P = Pc * (1 + r ** 2 / 3) ** -3
    check_spherical("n5_polytrope", r, rho, P, G)
    write("L3_n5_polytrope", [r, rho, P], "r rho P  (4 pi G = 1, rho_c = R = 1, P_c = 1/6)",
          {"four_pi_G": 1.0, "gamma": 5 / 3, "Pc": Pc})
    # collapse variant: uniform pressure P(0.9 R) for r < 0.9 R, gamma = 1.3 (Sec. 4.6)
    P2 = np.where(r < 0.9, Pc * (1 + 0.81 / 3) ** -3, P)
    write("L3_n5_collapse", [r, rho, P2], "r rho P  (n=5 polytrope, P uniform inside 0.9 R)",
          {"four_pi_G": 1.0, "gamma": 1.3, "Pc": Pc})


# ------------------------------------------------------------------ L5 Evrard
def evrard():
    """Evrard 1988: M = R = G = 1, rho = M/(2 pi R^2 r) for r < R, u = 0.05 G M / R, gamma = 5/3."""
    r = np.concatenate([np.geomspace(1e-6, 1.0, 20000)])
    rho = 1.0 / (2 * np.pi * r)
    P = (5 / 3 - 1) * rho * 0.05
    write("L5_evrard", [r, rho, P], "r rho P  (G = M = R = 1, four_pi_G = 4 pi, gamma = 5/3, u = 0.05)",
          {"four_pi_G": 4 * np.pi, "gamma": 5 / 3, "u": 0.05})


# ------------------------------------------------------------------ L8 singular isothermal sphere
def sis():
    """Shu 1977: rho = cs^2/(2 pi G r^2) = 2/r^2 (cs = 1, 4 pi G = 1), at rest; core capped at r_core."""
    rc = 0.02
    r = np.geomspace(1e-5, 8.0, 40000)
    # MASS-CONSERVING core (2026-10-04): a core capped at the SIS value 2/rc^2 removes 2/3 of the SIS mass
    # inside rc (0.335 of 0.503); the gas outside, still at SIS pressure, then sees too little gravity
    # (measured ratio 1 - 0.335/(8 pi x), e.g. 0.908 at x = 0.148) and expands. Uniform core 6/rc^2 holds
    # exactly M_SIS(rc) = 8 pi rc; pressure in the core = its density (isothermal, only 1-2 cells).
    rho = np.where(r < rc, 6.0 / rc ** 2, 2.0 / np.maximum(r, rc) ** 2)
    Mc = 4 * np.pi / 3 * rc ** 3 * 6.0 / rc ** 2
    assert abs(Mc - 8 * np.pi * rc) < 1e-12, Mc
    print(f"  SIS core: M(rc) = {Mc:.6f} = 8 pi rc = {8 * np.pi * rc:.6f}")
    write("L8_sis", [r, rho, rho], "r rho P  (cs = 1, 4 pi G = 1; P = rho; mass-conserving uniform core r < 0.02)",
          {"four_pi_G": 1.0, "r_core": rc})


# ------------------------------------------------------------------ L9 Boss-Bodenheimer
def boss_bodenheimer():
    """Boss & Bodenheimer 1979 'standard isothermal test' as in Truelove et al. 1998 / Bate & Burkert
    1997: uniform sphere, alpha = E_th/|W| = 0.26, beta = E_rot/|W| = 0.16, m = 2 perturbation 10 %.
    Units rho0 = cs = 1, 4 pi G = 1: alpha = 7.5 cs^2/R^2 -> R = sqrt(7.5/0.26); beta = Omega^2 -> Omega = 0.4."""
    R = np.sqrt(7.5 / 0.26)
    r = np.linspace(0, R, 2001)
    write("L9_boss_bodenheimer", [r, np.ones_like(r), np.ones_like(r)], "r rho P  (rho0 = cs = 1, 4 pi G = 1)",
          {"four_pi_G": 1.0, "R": R, "Omega": 0.4, "m2_amp": 0.1, "t_ff": float(np.sqrt(3 * np.pi * np.pi / 8.0))})


# ------------------------------------------------------------------ L10 Hennebelle et al. 2011
def hennebelle():
    """Hennebelle et al. 2011 Sec. 2.1: rho = rho_c/(1 + (r/r0)^2), contrast 10 at the edge R = 1.35 pc,
    r0 = 0.22 pc, T = 10 K, 100 Msun; ambient rho_e/10 in pressure equilibrium. Code units: cs(10 K) = 1,
    4 pi G = 1, rho_c = 1 -> lengths in cs/sqrt(4 pi G rho_c); the physical ratio R/r0 = 6.136 sets r0 from
    the contrast: (R/r0)^2 = 9 -> R/r0 = 3 for contrast 10 (their r0 ~ 0.22 pc with R = 1.35 pc gives 6.1,
    which is inconsistent with contrast 10 for this profile; we follow the CONTRAST, R/r0 = 3).
    alpha_th = 0.12 fixes R in code units: E_th/|W| for this profile is computed numerically."""
    G = 1 / (4 * np.pi)
    def alpha_th(R):
        r0 = R / 3.0
        r = np.linspace(0, R, 20001)
        rho = 1.0 / (1 + (r / r0) ** 2)
        M = 4 * np.pi * cumulative_simpson(rho * r ** 2, x=r, initial=0.0)
        Eth = 1.5 * M[-1] * 1.0
        W = -G * np.trapezoid(4 * np.pi * r ** 2 * rho * M / np.maximum(r, 1e-12), r)
        return Eth / abs(W)
    R = brentq(lambda R: alpha_th(R) - 0.12, 1.0, 200.0)
    r0 = R / 3.0
    r = np.linspace(0, R, 4001)
    rho = 1.0 / (1 + (r / r0) ** 2)
    write("L10_hennebelle", [r, rho, rho], "r rho P  (cs = 1, 4 pi G = 1, rho_c = 1; P = rho)",
          {"four_pi_G": 1.0, "R": R, "r0": r0, "rho_e": float(rho[-1]), "alpha_th": 0.12})


# ------------------------------------------------------------------ L6' exact BE spheres, light hot ambient
def be_spheres():
    """Exact isothermal Lane-Emden (cs = 1, 4 pi G = 1, xi = r), rho = f exp(-psi), P = rho; the ambient
    (profile1d rho_out/p_out) is 1/100 of the edge density at the edge pressure: confinement without the
    mass reservoir of collapse_be's static ambient (finding SG-F5)."""
    for xi, f in [(3.0, 1.1), (6.45, 1.0), (9.0, 1.0), (9.0, 1.1)]:
        s_ = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], (1e-6, xi), [1e-12 / 6, 1e-6 / 3],
                       rtol=1e-12, atol=1e-14, dense_output=True)
        r = np.linspace(0, xi, 6001)
        rho = f * np.exp(-s_.sol(np.maximum(r, 1e-6))[0])
        check_spherical(f"BE xi={xi} (f=1 shape)", r, rho / f, rho / f, 1 / (4 * np.pi))
        write(f"L6b_be_xi{xi:g}_f{f:g}", [r, rho, rho], f"r rho P  (exact BE xi_max = {xi}, f = {f}, cs = 1, 4 pi G = 1)",
              {"four_pi_G": 1.0, "xi": xi, "f": f, "rho_edge": float(rho[-1])})


if __name__ == "__main__":
    be_spheres()
    polytrope_atmosphere()
    spitzer_sheet()
    plummer_polytrope()
    evrard()
    sis()
    boss_bodenheimer()
    hennebelle()
    json.dump(META, open(os.path.join(OUT, "tables_meta.json"), "w"), indent=1)
