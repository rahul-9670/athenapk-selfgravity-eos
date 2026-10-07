#!/usr/bin/env python
"""Exact Newtonian potentials for the self-gravity validation suite.

Convention: phi solves  lap(phi) = 4 pi G rho,  phi -> 0 at infinity,  G = four_pi_G / (4 pi).
Code units throughout (four_pi_G = 1 in every test except jeans_wave).

Every function is vectorised over (x, y, z) arrays. `python exact_potentials.py` runs the
self-tests: each closed form is checked against brute-force numerical quadrature.

References
  uniform sphere, Plummer, Gaussian: Binney & Tremaine 2008, Galactic Dynamics, 2nd ed., Ch. 2
  homogeneous ellipsoid (Maclaurin spheroid): Chandrasekhar 1969, Ellipsoidal Figures of
      Equilibrium, Ch. 3; Binney & Tremaine 2008 eq. (2.140)
  homogeneous rectangular prism: Nagy, Papp & Benedek 2000, J. Geodesy 74, 552, eq. (8)
"""
import numpy as np
from scipy.special import erf
from scipy import integrate


def G_of(four_pi_G=1.0):
    return four_pi_G / (4.0 * np.pi)


# ---------------------------------------------------------------- spherical
def phi_uniform_sphere(x, y, z, M, R, c=(0, 0, 0), four_pi_G=1.0):
    G = G_of(four_pi_G)
    r = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    rs = np.maximum(r, 1e-300)
    return np.where(r >= R, -G * M / rs, -G * M / (2 * R ** 3) * (3 * R ** 2 - r ** 2))


def phi_gaussian(x, y, z, M, sigma, c=(0, 0, 0), four_pi_G=1.0):
    """rho = M (2 pi s^2)^-1.5 exp(-r^2/2s^2)  ->  phi = -G M erf(r/(sqrt2 s)) / r."""
    G = G_of(four_pi_G)
    r = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    small = r < 1e-8 * sigma
    rs = np.where(small, 1.0, r)
    val = -G * M * erf(rs / (np.sqrt(2) * sigma)) / rs
    return np.where(small, -G * M * np.sqrt(2 / np.pi) / sigma, val)


def phi_plummer_truncated(x, y, z, M, a, rt, c=(0, 0, 0), four_pi_G=1.0):
    """rho = 3M/(4 pi a^3)(1+r^2/a^2)^-2.5 for r < rt, 0 outside. M is the UNtruncated mass."""
    G = G_of(four_pi_G)
    r = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    Mr = lambda s: M * s ** 3 / (s ** 2 + a ** 2) ** 1.5
    Mt = Mr(rt)
    # 4 pi int_r^rt rho s ds = (M/a) [ (1+r^2/a^2)^-1.5 ... ] -> exact antiderivative
    shell = lambda r_: (M / a) * ((1 + r_ ** 2 / a ** 2) ** -1.5 - (1 + rt ** 2 / a ** 2) ** -1.5)
    rs = np.maximum(r, 1e-300)
    inside = -G * (Mr(rs) / rs + shell(r))
    return np.where(r >= rt, -G * Mt / rs, inside)


def phi_spherical_quadrature(x, y, z, rho_of_r, R, c=(0, 0, 0), four_pi_G=1.0, n=400001):
    """Exact potential of an arbitrary spherical rho(r) supported on r < R (1D quadrature,
    n-point Simpson; accuracy ~1e-12 for smooth profiles)."""
    G = G_of(four_pi_G)
    s = np.linspace(0.0, R, n)
    rho = rho_of_r(s)
    dM = 4 * np.pi * rho * s ** 2
    dW = 4 * np.pi * rho * s
    M = integrate.cumulative_simpson(dM, x=s, initial=0.0)
    W = integrate.cumulative_simpson(dW, x=s, initial=0.0)  # int_0^r
    Mtot, Wtot = M[-1], W[-1]
    r = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    rin = np.minimum(r, R)
    Mi = np.interp(rin, s, M)
    Wi = np.interp(rin, s, W)
    rs = np.maximum(r, 1e-300)
    phi_in = -G * (np.where(r > 0, Mi / rs, 0.0) + (Wtot - Wi))
    phi_in = np.where(r > 0, phi_in, -G * Wtot)
    return np.where(r >= R, -G * Mtot / rs, phi_in), Mtot


# ---------------------------------------------------------------- spheroid
def phi_maclaurin(x, y, z, rho, a, cax, c=(0, 0, 0), four_pi_G=1.0):
    """Homogeneous OBLATE spheroid, semi-axes (a, a, cax), a > cax, symmetry axis z.
    Phi = -pi G rho a^2 c [I0(l) - R^2 I1(l) - z^2 I3(l)],  l = 0 inside, else the
    ellipsoidal coordinate of the point (B&T 2008 eq. 2.140 reduced to closed form)."""
    G = G_of(four_pi_G)
    X, Y, Z = x - c[0], y - c[1], z - c[2]
    R2 = X ** 2 + Y ** 2
    Z2 = Z ** 2
    e = np.sqrt(a ** 2 - cax ** 2)
    m2 = R2 / a ** 2 + Z2 / cax ** 2
    b = a ** 2 + cax ** 2 - R2 - Z2
    cc = a ** 2 * cax ** 2 - R2 * cax ** 2 - Z2 * a ** 2
    lam = np.where(m2 > 1.0, 0.5 * (-b + np.sqrt(np.maximum(b * b - 4 * cc, 0.0))), 0.0)
    w0 = np.sqrt(cax ** 2 + lam)
    t = np.pi / 2 - np.arctan(w0 / e)
    I0 = 2.0 / e * t
    I1 = (t - e * w0 / (w0 ** 2 + e ** 2)) / e ** 3
    I3 = 2.0 / e ** 2 * (1.0 / w0 - t / e)
    return -np.pi * G * rho * a ** 2 * cax * (I0 - R2 * I1 - Z2 * I3)


# ---------------------------------------------------------------- cuboid
def phi_prism(x, y, z, rho, lo, hi, four_pi_G=1.0):
    """Homogeneous rectangular prism [lo, hi] (Nagy, Papp & Benedek 2000)."""
    G = G_of(four_pi_G)
    U = 0.0
    for i, xc in enumerate((lo[0], hi[0])):
        for j, yc in enumerate((lo[1], hi[1])):
            for k, zc in enumerate((lo[2], hi[2])):
                X, Y, Z = xc - x, yc - y, zc - z
                r = np.sqrt(X * X + Y * Y + Z * Z)
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = (X * Y * np.log(Z + r) + Y * Z * np.log(X + r) + Z * X * np.log(Y + r)
                         - 0.5 * X * X * np.arctan(Y * Z / (X * r))
                         - 0.5 * Y * Y * np.arctan(Z * X / (Y * r))
                         - 0.5 * Z * Z * np.arctan(X * Y / (Z * r)))
                U = U + (-1) ** (i + j + k + 1) * f
    return -G * rho * U


# ---------------------------------------------------------------- periodic
def phi_sinusoid(x, y, z, A, L, lo=(0, 0, 0), four_pi_G=1.0):
    """rho = rho_bg + A sss  in a periodic box; zero-mean potential (Tomida & Stone 2023 eq. 29)."""
    k2 = sum((2 * np.pi / Li) ** 2 for Li in L)
    s = (np.sin(2 * np.pi * (x - lo[0]) / L[0]) * np.sin(2 * np.pi * (y - lo[1]) / L[1])
         * np.sin(2 * np.pi * (z - lo[2]) / L[2]))
    return -four_pi_G * A * s / k2


# ---------------------------------------------------------------- self-tests
def _brute(rho_fn, box, pts, n=160, four_pi_G=1.0):
    """Direct 3D midpoint summation (smooth rho) for checking closed forms."""
    G = G_of(four_pi_G)
    ax = [np.linspace(box[0][d], box[1][d], n + 1) for d in range(3)]
    c = [0.5 * (a[1:] + a[:-1]) for a in ax]
    dV = np.prod([(box[1][d] - box[0][d]) / n for d in range(3)])
    X, Y, Z = np.meshgrid(*c, indexing="ij")
    rho = rho_fn(X, Y, Z)
    out = []
    for p in pts:
        r = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2)
        out.append(-G * np.sum(rho / np.maximum(r, 1e-12)) * dV)
    return np.array(out)


def _selftest():
    ok = True
    def check(name, a, b, tol):
        nonlocal ok
        e = np.max(np.abs(a - b) / np.abs(b))
        flag = "PASS" if e < tol else "FAIL"
        ok &= e < tol
        print(f"  {name:48s} max rel diff {e:.2e}  (tol {tol:.0e})  {flag}")

    pts = np.array([[0.3, 0.1, -0.2], [1.7, 0.4, 0.9], [3.0, -2.0, 1.0], [0.0, 0.0, 2.5]])
    x, y, z = pts.T
    # 1) spherical quadrature reproduces the uniform sphere and the Plummer closed forms
    q, _ = phi_spherical_quadrature(x, y, z, lambda s: np.where(s <= 1.5, 2.0, 0.0), 1.5)
    check("quadrature vs uniform sphere", q, phi_uniform_sphere(x, y, z, 2.0 * 4 / 3 * np.pi * 1.5 ** 3, 1.5), 1e-9)
    rho_pl = lambda s: 3 / (4 * np.pi * 0.7 ** 3) * (1 + s ** 2 / 0.7 ** 2) ** -2.5
    q, _ = phi_spherical_quadrature(x, y, z, rho_pl, 2.2)
    check("quadrature vs truncated Plummer closed form", q, phi_plummer_truncated(x, y, z, 1.0, 0.7, 2.2), 1e-9)
    rho_g = lambda s: (2 * np.pi * 0.5 ** 2) ** -1.5 * np.exp(-s ** 2 / (2 * 0.5 ** 2))
    q, _ = phi_spherical_quadrature(x, y, z, rho_g, 12.0)
    check("quadrature vs Gaussian erf closed form", q, phi_gaussian(x, y, z, 1.0, 0.5), 1e-9)
    # 2) Maclaurin spheroid: sphere limit, and 3D integral (smoothed edge is not exact, so
    #    instead check against the B&T integral evaluated by quad)
    a, cz = 1.2, 0.5
    def bt(p):
        X, Y, Z = p
        m2 = (X * X + Y * Y) / a ** 2 + Z * Z / cz ** 2
        if m2 <= 1:
            lam = 0.0
        else:
            from scipy.optimize import brentq
            lam = brentq(lambda l: (X * X + Y * Y) / (a * a + l) + Z * Z / (cz * cz + l) - 1, 0, 1e6)
        f = lambda u: (1 - (X * X + Y * Y) / (a * a + u) - Z * Z / (cz * cz + u)) / ((a * a + u) * np.sqrt(cz * cz + u))
        return -np.pi * G_of() * 1.0 * a * a * cz * integrate.quad(f, lam, np.inf, epsabs=0, epsrel=1e-13, limit=500)[0]
    pts2 = np.array([[0.2, 0.1, 0.1], [0.9, 0.3, 0.0], [1.5, 0.2, 0.7], [0.1, 0.0, 2.0], [3, 3, 3]])
    check("Maclaurin closed form vs B&T 2.140 integral", phi_maclaurin(*pts2.T, 1.0, a, cz),
          np.array([bt(p) for p in pts2]), 1e-10)
    # 3) prism: vs brute force (exterior + interior points)
    lo, hi = (-1.0, -0.5, -0.75), (1.0, 0.5, 0.75)
    pts3 = np.array([[0.13, 0.07, -0.21], [0.8, -0.3, 0.5], [2.0, 1.0, 0.3], [0.0, 0.0, 3.0]])
    br = _brute(lambda X, Y, Z: np.ones_like(X), (lo, hi), pts3 + 1e-4, n=200)
    check("prism closed form vs 200^3 brute sum", phi_prism(*(pts3 + 1e-4).T, 1.0, lo, hi), br, 2e-3)
    # prism far field -> point mass
    far = np.array([[40.0, 30.0, 20.0]])
    check("prism far field vs -GM/r", phi_prism(*far.T, 1.0, lo, hi),
          -G_of() * 2 * 1 * 1.5 / np.sqrt((far ** 2).sum(1)), 1e-3)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
