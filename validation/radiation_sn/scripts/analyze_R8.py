#!/usr/bin/env python
"""R8 analysis. R8a: Ganapol plane source at t = 1 (and 0.5) vs scripts/ganapol_ref.py, the reference convolved with the
initial Gaussian (width W, make_decks_R8.py). phi = c E with c_code = 1.
Two references, because a discrete-ordinates code cannot converge POINTWISE to Ganapol at fixed angles: its uncollided
flux is a set of spikes at x = mu_n t instead of the flat top e^{-t}/(2t) (phi(0) gets no uncollided part at all):
 * SN_ref = the EXACT solution of the discrete-ordinates equations with the code's own angular set (ganapol_ref.sn_plane,
   continuous x and t): code vs SN_ref = the code's SPATIAL + TEMPORAL error (L1, max in |x| < 0.9 t);
 * Ganapol: compared in measures that do converge with the angular set -- the Wasserstein-1 distance
   W1 = int |C(x) - C_ref(x)| dx of the cumulative distributions, and the variance <x^2> (exact for the continuum:
   (2/3)(t - 1 + e^{-t}) + W^2). SN_ref's own W1 to Ganapol is the angular error of the set.
Also int phi dx (= 1 exactly: pure scattering, vacuum boundaries outside the front).
usage: analyze_R8.py run | check   -> results/tables/R8_benchmarks.txt
"""
import glob
import os
import sys

import h5py
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ganapol_ref import phi_collided, sn_plane  # noqa: E402
from su_olson_ref import code_slab  # noqa: E402
from make_decks_R8 import DTS, ORDS, W  # noqa: E402
from rad_common import MOD  # noqa: E402

RUNS = os.path.join(MOD, "runs", "R8")
OUT = os.path.join(MOD, "results", "tables", "R8_benchmarks.txt")


def reference(xc, t):
    """phi(x, t) convolved with the unit Gaussian of width W: uncollided flat top analytically, collided numerically."""
    unc = 0.5 * np.exp(-t) / t * (ndtr((xc + t) / W) - ndtr((xc - t) / W))
    xf = np.linspace(-t, t, 4001)
    cf = np.array([phi_collided(x, t) for x in xf])
    dxf = xf[1] - xf[0]
    col = np.array([np.sum(cf * np.exp(-0.5 * ((x - xf) / W) ** 2)) * dxf / (np.sqrt(2 * np.pi) * W) for x in xc])
    return unc + col


def load(d, t):
    for f in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
        with h5py.File(f, "r") as h:
            if abs(float(h["Info"].attrs["Time"]) - t) < 1e-9:
                x = np.array(h["Locations/x"])[0]
                return 0.5 * (x[1:] + x[:-1]), np.array(h["rad.Er"])[0, 0, 0], x[1] - x[0]
    raise FileNotFoundError(d, t)


def point_section():
    """R8c: 3-D point source; per-cell phi vs ganapol_ref.point_smoothed at the cell centre. Metrics: radial W1 of the
    cumulative energy C(r) = sum_{r_i < r} phi_i dV vs the reference's, the shell-mean error (cells binned by r, dr = dx),
    and the RAY-EFFECT spread: rms and max over cells of phi/ref - 1 in r < 0.9 t (the reference is isotropic)."""
    from ganapol_ref import point_smoothed
    from make_decks_R8 import WC
    out = ["R8c Ganapol 3-D point source (pure scattering == steady gray radiative equilibrium): phi = c E"]
    for nm in ("point_o4x8", "point_o8x16", "point_o4x8_step"):
        d = os.path.join(RUNS, nm)
        for t in (0.5, 1.0):
            f = None
            for g in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
                with h5py.File(g, "r") as h:
                    if abs(float(h["Info"].attrs["Time"]) - t) < 1e-9:
                        f = g
            if f is None:
                out.append(f"  {nm} t {t}: no dump")
                continue
            with h5py.File(f, "r") as h:
                E = np.array(h["rad.Er"])
                xs = [np.array(h[f"Locations/{c}"]) for c in "xyz"]
            c = [0.5 * (a[:, 1:] + a[:, :-1]) for a in xs]
            X = np.broadcast_to(c[0][:, None, None, :], E.shape)
            Y = np.broadcast_to(c[1][:, None, :, None], E.shape)
            Z = np.broadcast_to(c[2][:, :, None, None], E.shape)
            dV = float((xs[0][0, 1] - xs[0][0, 0]) ** 3)
            dx = float(xs[0][0, 1] - xs[0][0, 0])
            r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2).ravel()
            ph = E.ravel()
            o = np.argsort(r)
            r, ph = r[o], ph[o]
            ref = point_smoothed(r, t, WC)
            C, Cr = np.cumsum(ph) * dV, np.cumsum(ref) * dV
            w1r = float(np.sum(np.abs(C - Cr)[:-1] * np.diff(r)))
            inn = r < 0.9 * t
            q = ph[inn] / ref[inn] - 1.0
            bins = np.arange(0.0, 0.9 * t, dx)
            sm = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (r >= lo) & (r < hi)
                if sel.sum() > 20:
                    sm.append(np.mean(ph[sel]) / np.mean(ref[sel]) - 1.0)
            sm = np.abs(np.array(sm))
            out.append(f"  {nm:16s} t {t}: int phi {ph.sum() * dV:.8f} (ref in box {ref.sum() * dV:.8f}) | radial W1 {w1r:.3e} "
                       f"| shell-mean |err| median {np.median(sm):.3e} max {sm.max():.3e} | ray spread rms {np.sqrt(np.mean(q * q)):.3e} "
                       f"max {np.abs(q).max():.3e} (r < 0.9 t)")
    return out


def shock_section():
    """R8b: right-moving shock of the colliding-flow run (x > 0), mapped to the reference frame xi = x_s - x (upstream
    xi < 0); x_s = the embedded hydrodynamic jump (max |d rho/dx|). Temperatures / T0 (code T = p/rho, T0 = 1/gamma)."""
    from make_decks_R8 import RS_C, RS_KAP, RS_M, RS_P0
    from radshock_ref import Shock
    out = ["R8b radiative shock (Lowrie & Edwards 2008 model; reference = radshock_ref.py non-equilibrium diffusion, "
           "'static' = the code's physics without O(u/c) terms, 'full' = with them): T/T0, theta = T_rad/T0"]
    for M0 in RS_M:
        refs = {}
        for full in (False, True):
            try:
                refs[full] = Shock(M0, RS_P0, RS_C, RS_KAP, full=full).solve()
            except RuntimeError as e:
                refs[full] = None
                out.append(f"  M0 {M0} {'full' if full else 'static'} reference: {e}")
        for nm in (f"shock_M{M0:.0f}", f"shock_M{M0:.0f}_step"):
            d = os.path.join(RUNS, nm)
            for f in sorted(glob.glob(os.path.join(d, "parthenon.out0.0000[23].phdf"))) + [os.path.join(d, "parthenon.out0.final.phdf")]:
                if not os.path.exists(f):
                    continue
                with h5py.File(f, "r") as h:
                    t = float(h["Info"].attrs["Time"])
                    P = dict(h["Params"].attrs)
                    x = np.array(h["Locations/x"])[0]
                    xc = 0.5 * (x[1:] + x[:-1])
                    pr = np.array(h["prim"])[0, :, 0, 0]
                    E = np.array(h["rad.Er"])[0, 0, 0]
                a = float(P["radiation/arad"])
                T0 = 0.6
                rho, T, th = pr[0], pr[4] / pr[0] / T0, (np.maximum(E, 0) / a) ** 0.25 / T0
                rgt = xc > 0
                xr, rr, Tr_, thr = xc[rgt], rho[rgt], T[rgt], th[rgt]
                js = int(np.argmax(np.abs(np.diff(rr)))) 
                xs = 0.5 * (xr[js] + xr[js + 1])
                xi = xs - xr
                line = f"  {nm:14s} t {t:.4f}: x_s {xs:.5f} (u_s {xs / t:.4f}) Tmax {Tr_.max():.4f} theta(x_s) {np.interp(0.0, xi[::-1], thr[::-1]):.4f} "
                up = xi < -2e-4
                dn = (xi > 0.004) & (xi < 0.006)
                line += f"T(xi=-2e-4) {np.interp(-2e-4, xi[::-1], Tr_[::-1]):.4f} T1 {np.mean(Tr_[dn]):.5f} rho1 {np.mean(rr[dn]):.5f}"
                for full in (False, True):
                    R = refs[full]
                    if R is None:
                        continue
                    xg = np.linspace(-0.012, 0.004, 801)
                    Tg, thg, rg = R.profile(xg)
                    Tg, thg = Tg / R.T0, thg / R.T0
                    l1 = np.mean(np.abs(np.interp(xg, xi[::-1], thr[::-1]) - thg)) / np.mean(thg - 1.0 + 1e-30)
                    if t == 0.02 or full is False:
                        pass
                    line += (f" | {'full' if full else 'static'} ref: Tmax {Tg.max():.4f} theta(0) {np.interp(0.0, xg, thg):.4f} "
                             f"T(-2e-4) {np.interp(-2e-4, xg, Tg):.4f} T1 {R.state(R.E1, 0, 0)[2] / R.T0:.5f} rho1 "
                             f"{R.state(R.E1, 0, 0)[1]:.5f} L1(theta - 1)/<theta - 1> {l1:.3e}")
                out.append(line)
    return out


def w1(a, b, dx):
    return float(np.sum(np.abs(np.cumsum(a - b))) * dx * dx)


def main(mode):
    L = ["# R8a Ganapol plane source (scripts/analyze_R8.py): phi = c E; SN_ref = exact discrete ordinates on the code set; "
         "Ganapol = closed form (ganapol_ref.py, verified by "
         "Monte Carlo) convolved with the initial Gaussian; L1 = int|phi-ref|/int ref; int = int phi dx (exact 1)"]
    refs = {}
    snref = {}
    for t in (0.5, 1.0):
        L.append(f"t = {t}:")
        for nz, npsi in ORDS:
            xc, _, dx = load(os.path.join(RUNS, f"plane_o{nz}x{npsi}_dt{DTS[0]:.0e}_sou"), t)
            if t not in refs:
                refs[t] = reference(xc, t)
            if (nz, t) not in snref:
                mu, wh = code_slab(nz, npsi)
                snref[(nz, t)] = sn_plane(mu, wh, xc, dx, [t], W)[0]
            sr = snref[(nz, t)]
            var = lambda f: float(np.sum(f * xc * xc) / np.sum(f))
            L.append(f"  {nz:2d}x{npsi:2d} SN_ref (exact discrete ordinates): W1 to Ganapol {w1(sr, refs[t], dx):.3e}  "
                     f"variance {var(sr):.5f} (continuum {(2 / 3) * (t - 1 + np.exp(-t)) + W * W:.5f})")
            for dt in DTS:
                for sp in ("sou", "step"):
                    d = os.path.join(RUNS, f"plane_o{nz}x{npsi}_dt{dt:.0e}_{sp}")
                    xc, ph, dx = load(d, t)
                    if t not in refs:
                        refs[t] = reference(xc, t)
                    ref = refs[t]
                    sr = snref[(nz, t)]
                    l1 = np.sum(np.abs(ph - sr)) / np.sum(sr)
                    inn = np.abs(xc) < 0.9 * t
                    mx = np.max(np.abs(ph[inn] / sr[inn] - 1))
                    L.append(f"    code dt {dt:.0e} {sp:4s}: vs SN_ref L1 {l1:.3e} max rel |x|<0.9t {mx:.3e} | vs Ganapol W1 "
                             f"{w1(ph, ref, dx):.3e} variance {np.sum(ph * xc * xc) / np.sum(ph):.5f} | int "
                             f"{ph.sum() * dx:.10f}  min {ph.min():+.2e}")
    L += shock_section()
    L += point_section()
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
