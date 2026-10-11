#!/usr/bin/env python
"""R8a reference: the time-dependent PLANE-SOURCE benchmark in an infinite, purely and isotropically scattering medium
(sigma_t = sigma_s = 1, particle speed 1, unit isotropic plane pulse at x = 0, t = 0), scalar flux phi(x, t).

Formula (Ganapol 2001, LA-UR-01-1854; as written in Garrett & Hauck 2013, Transp. Theory Stat. Phys. 42, 203, eq. for
the plane source; quoted from memory and therefore VERIFIED here by an independent Monte Carlo, `check_mc`):
  phi(x, t) = e^{-t} / (2t) [1 + t/(4 pi) (1 - eta^2) int_0^pi sec^2(u/2) Re{xi^2 exp(t/2 (1 - eta^2) xi)} du],  |x| < t
  eta = x/t, q = (1 + eta)/(1 - eta), xi(u) = (ln q + i u) / (eta + i tan(u/2)); phi = 0 for |x| > t.
The first term e^{-t}/(2t) is the uncollided flux.
usage: ganapol_ref.py check   (prints the formula's integral and its distance from a 4e7-particle Monte Carlo)
"""
import sys

import numpy as np
from scipy.integrate import quad


def phi(x, t):
    eta = x / t
    if abs(eta) >= 1.0:
        return 0.0
    q = (1.0 + eta) / (1.0 - eta)
    a = 0.5 * t * (1.0 - eta * eta)

    def f(u):
        xi = (np.log(q) + 1j * u) / (eta + 1j * np.tan(0.5 * u))
        return (1.0 / np.cos(0.5 * u) ** 2) * (xi * xi * np.exp(a * xi)).real

    I = quad(f, 0.0, np.pi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    return np.exp(-t) / (2.0 * t) * (1.0 + t / (4.0 * np.pi) * (1.0 - eta * eta) * I)


def phi_collided(x, t):
    return phi(x, t) - (np.exp(-t) / (2.0 * t) if abs(x) < t else 0.0)


def mc(t, n, seed=1, nbin=100):
    """Analog Monte Carlo: isotropic emission at x = 0, exponential flights (sigma = 1), isotropic re-emission, speed 1.
    Returns bin centres, the collided-particle density at time t (per unit x) and its 1-sigma error; the uncollided
    particles are counted separately (their exact density is e^{-t}/(2t) on |x| < t, so only the collided part is binned)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    tt = np.zeros(n)
    mu = rng.uniform(-1, 1, n)
    col = np.zeros(n, bool)
    live = np.ones(n, bool)
    while live.any():
        idx = np.nonzero(live)[0]
        s = rng.exponential(1.0, idx.size)
        dt = np.minimum(s, t - tt[idx])
        x[idx] += mu[idx] * dt
        tt[idx] += dt
        hit = s < (t - tt[idx] + dt) - 1e-15
        hit &= tt[idx] < t
        done = ~hit
        live[idx[done]] = False
        h = idx[hit]
        col[h] = True
        mu[h] = rng.uniform(-1, 1, h.size)
    edges = np.linspace(-t, t, nbin + 1)
    c, _ = np.histogram(x[col], edges)
    w = edges[1] - edges[0]
    return 0.5 * (edges[1:] + edges[:-1]), c / (n * w), np.sqrt(np.maximum(c, 1)) / (n * w), (~col).mean()


def sn_plane(mu, wh, xc, h, ts, W, Lp=8.0, nk=2048):
    """EXACT solution of the discrete-ordinates equations (continuous x and t) for the plane source with the angular set
    (mu, wh) (sum wh = 2): d psi_m/dt + mu_m d psi_m/dx + psi_m = (1/2) sum_n wh_n psi_n, initial psi_m = G(x)/2 (G the
    unit Gaussian of width W). Fourier series on a periodic box Lp, per wave number the exact matrix exponential (via
    the eigen-decomposition); returns CELL AVERAGES (sinc factor) of phi = sum wh psi on cells (xc, h) at each t in ts.
    Same method as su_olson_ref.py: isolates the code's spatial/temporal error from its angular one."""
    N = len(mu)
    out = np.zeros((len(ts), len(xc)))
    for k in 2 * np.pi * np.arange(nk + 1) / Lp:
        Gk = np.exp(-0.5 * (k * W) ** 2) / Lp
        M = np.diag(-(1.0 + 1j * k * mu)).astype(complex) + 0.5 * np.outer(np.ones(N), wh)
        lam, V = np.linalg.eig(M)
        c0 = np.linalg.solve(V, 0.5 * Gk * np.ones(N, complex))
        avg = (np.sin(k * h / 2) / (k * h / 2)) if k > 0 else 1.0
        ph = np.exp(1j * k * xc) * avg * (1.0 if k == 0 else 2.0)
        for it, t in enumerate(ts):
            pk = wh @ (V @ (np.exp(lam * t) * c0))
            out[it] += np.real(pk * ph)
    return out


def point_smoothed(r, t, W, nx=8001):
    """3-D POINT source (unit isotropic pulse) in the same medium, smoothed with the isotropic 3-D Gaussian of width W:
    phi_pt(r, t) = -(1 / (2 pi r)) d/dr [phi_pl * G_W](r, t) -- the plane/point Green's-function relation for an infinite
    homogeneous medium (a plane source is the point source integrated over the plane), which commutes with the
    Gaussian smoothing because the plane integral of a 3-D Gaussian is the 1-D Gaussian of the same width."""
    from scipy.special import ndtr
    x = np.linspace(0.0, t + 6 * W, nx)
    unc = 0.5 * np.exp(-t) / t * (ndtr((x + t) / W) - ndtr((x - t) / W))
    xf = np.linspace(-t, t, 4001)
    cf = np.array([phi_collided(v, t) for v in xf])
    dxf = xf[1] - xf[0]
    col = np.array([np.sum(cf * np.exp(-0.5 * ((v - xf) / W) ** 2)) * dxf / (np.sqrt(2 * np.pi) * W) for v in x])
    pl = unc + col
    d = np.gradient(pl, x)
    pt = -d / (2 * np.pi * np.maximum(x, 1e-30))
    pt[0] = pt[1]
    return np.interp(r, x, pt)


def check_mc():
    t = 1.0
    tot = quad(lambda x: phi_collided(x, t), -t, t, limit=200, epsrel=1e-10)[0] + np.exp(-t)
    print(f"formula: int phi dx at t = 1: {tot:.12f} (exact 1: pure scattering conserves particles)")
    xs, dens, err, unc = mc(t, 40_000_000)
    ref = np.array([quad(lambda x: phi_collided(x, t), a - 0.01, a + 0.01)[0] / 0.02 for a in xs])
    z = (dens - ref) / err
    print(f"MC 4e7 particles: uncollided fraction {unc:.6f} (exact e^-1 = {np.exp(-1):.6f}); collided density vs "
          f"formula (bin averages, 100 bins): max |z| {np.abs(z).max():.2f}, rms z {np.sqrt(np.mean(z * z)):.2f}, max rel "
          f"{np.max(np.abs(dens / ref - 1)):.2e}")
    for a in (0.0, 0.3, 0.6, 0.9, 0.99):
        print(f"  phi({a}, 1) = {phi(a, 1.0):.8f}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check_mc()
