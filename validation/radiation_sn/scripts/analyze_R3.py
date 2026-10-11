#!/usr/bin/env python
"""R3 free streaming (make_decks_R3.py): searchlight front speed, steady beam intensity and transverse flux,
numerical beam spreading, crossing-beam superposition, shadow transmission. c = 1, Q = 1, beam ordinate from the
run log ("## Sn test source: ordinate a = (nx, ny, nz), w = ...").
Shadow (S4): the "analytic" e^-11.5 holds only for the unspread beam; the numerical beam is wider than the column, so
the predicted transmission is computed from S1's OWN profile just upstream of the column (x = -0.2): every y cell is
traced along (nx, ny) through the column |x|, |y| < 0.125 with the exact chord tau = chi L / sqrt(nx^2 + ny^2)
(chi = 1000 x 0.04 = 40 inside; background attenuation removed separately), T_pred = sum F1 e^-tau / sum F1.
_hr (256 x 128, tlim 3.5): S1 and S4 only; diagnostics at t = 3.5 behind the front (window x < 0.5).
usage: analyze_R3.py run | check   -> results/tables/R3_free_streaming.txt
"""
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rad_common import MOD  # noqa: E402

RUNS = os.path.join(MOD, "runs", "R3")
OUT = os.path.join(MOD, "results", "tables", "R3_free_streaming.txt")
XS, X1 = -1.85, -1.8          # source centre / downstream edge in x
YA = -0.766                   # beam A source centre y
H = 0.1                       # source extent in x and y
CHI_COL, CHI_BG, HW = 1000.0 * 0.04, 0.50025 * 0.04, 0.125   # S4 column / background chi, column half-width


def chord(y0, x0, nx, ny):
    """in-plane path length inside |x|, |y| < HW of the ray (x0, y0) + s (nx, ny)/|n|, for an array of y0."""
    m = ny / nx
    xa, xb = -HW, HW                                     # x-interval inside the column in x
    ya, yb = y0 + m * (xa - x0), y0 + m * (xb - x0)       # ray y at those x
    lo = np.maximum(xa, np.where(m > 0, xa + (-HW - ya) / m, xa + (HW - ya) / m))
    hi = np.minimum(xb, np.where(m > 0, xa + (HW - ya) / m, xa + (-HW - ya) / m))
    return np.maximum(hi - lo, 0.0) * np.sqrt(1.0 + m * m)


def ordinates(d):
    log = open(os.path.join(d, "run.log"), errors="ignore").read()
    return [(int(a), float(nx), float(ny), float(w)) for a, nx, ny, nz, w in
            re.findall(r"ordinate (\d+) = \(([-0-9.e]+), ([-0-9.e]+), ([-0-9.e]+)\), w = ([0-9.e]+)", log)]


def field(f, comp):
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"])
        u = np.array(h["sn.u"])[:, comp, 0]            # [block, ny, nx]
        x = np.array(h["Locations"]["x"]); y = np.array(h["Locations"]["y"])
    xc = 0.5 * (x[:, 1:] + x[:, :-1]); yc = 0.5 * (y[:, 1:] + y[:, :-1])
    X = np.unique(np.round(xc.ravel(), 12)); Y = np.unique(np.round(yc.ravel(), 12))
    F = np.zeros((len(Y), len(X)))
    for b in range(u.shape[0]):
        i0 = np.searchsorted(X, np.round(xc[b, 0], 12)); j0 = np.searchsorted(Y, np.round(yc[b, 0], 12))
        F[j0:j0 + u.shape[1], i0:i0 + u.shape[2]] = u[b]
    return t, X, Y, F


def dumps(d):
    return sorted(glob.glob(os.path.join(d, "parthenon.out0.0*.phdf"))) + sorted(glob.glob(os.path.join(d, "parthenon.out0.final.phdf")))


def centre(X, Y, F, nx, ny, y0):
    """beam centre-line value at each x column (linear in y)."""
    yc = y0 + (ny / nx) * (X - XS)
    return np.array([np.interp(yc[i], Y, F[:, i]) for i in range(len(X))])


def faceflux(X, Y, F, sou):
    """sum_j F_{i+1/2, j} dy at the downstream x-face of each column (upwind face value of the scheme; nx > 0)."""
    Fm = np.concatenate([F[:, :1] * 0.0, F[:, :-1]], axis=1)
    face = F + 0.5 * (F - Fm) if sou else F
    return face.sum(axis=0) * (Y[1] - Y[0]), X + 0.5 * (X[1] - X[0])


def main(mode):
    L = ["# R3 free streaming (scripts/analyze_R3.py): 2D, c = 1, Q = 1, no emission, no coupling. Conserved measure:",
         "# the transverse integral of the upwind FACE flux, sum_j F dy (= I0 H downstream of the source, exact in vacuum",
         "# until the spread beam reaches the y boundary); front = where it reaches half its steady value."]
    for res in ("", "_hr"):
        for sch in ("sou", "step"):
            d1 = os.path.join(RUNS, f"S1_{sch}{res}")
            if not os.path.isdir(d1):
                continue
            sou = (sch == "sou")
            a, nx, ny, w = ordinates(d1)[0]
            I0 = (H / nx) / w
            L.append(f"== {sch}{res}: beam ordinate {a} (nx {nx:.5f}, ny {ny:.5f}, w {w:.5f}); I0 = {I0:.6f}, I0 H = {I0 * H:.6f}")
            for f in dumps(d1):
                t, X, Y, F = field(f, a)
                if t == 0.0:
                    continue
                tf, xf = faceflux(X, Y, F, sou)
                ok = xf > X1
                half = xf[ok][np.argmax(tf[ok] < 0.5 * I0 * H)] if np.any(tf[ok] < 0.5 * I0 * H) else np.nan
                line = f"   t {t:4.1f}: front x {half:+.4f} vs -1.85 + nx t = {-1.85 + nx * t:+.4f}"
                xw = 0.5 if res else 1.0
                if t >= (3.5 if res else 5.0):
                    sel = (xf > X1 + 0.05) & (xf < xw)
                    c = centre(X, Y, F, nx, ny, YA)
                    wid = []
                    for xx in (-1.5, 0.5) if res else (-1.5, 0.5, 1.0):
                        i = int(np.argmin(np.abs(X - xx)))
                        p = np.maximum(F[:, i], 0.0); m = p.sum(); yb = (p * Y).sum() / m
                        wid.append(np.sqrt((p * (Y - yb) ** 2).sum() / m))
                    line += (f" | face-flux integral / (I0 H) on x in (-1.75, {xw:g}): min {tf[sel].min() / (I0 * H):.6f} "
                             f"max {tf[sel].max() / (I0 * H):.6f} | centre-line I/I0 at x = -1, 0, 1: "
                             + " ".join(f"{np.interp(xx, X, c) / I0:.4f}" for xx in (-1.0, 0.0, 1.0))
                             + " | rms width at x = " + ", ".join(f"{xx:g}" for xx in ((-1.5, 0.5) if res else (-1.5, 0.5, 1.0)))
                             + ": " + ", ".join(f"{v:.4f}" for v in wid) + f" (exact {0.1 / np.sqrt(12):.4f})"
                             + f" | min I/I0 {F.min() / I0:+.2e}")
                L.append(line)
            fa = dumps(d1)[-1]
            if not res:
                d2, d3 = os.path.join(RUNS, f"S2_{sch}"), os.path.join(RUNS, f"S3_{sch}")
                b = ordinates(d2)[0][0]
                fb, fc = dumps(d2)[-1], dumps(d3)[-1]
                _, _, _, A1 = field(fa, a); _, _, _, B2 = field(fb, b); _, _, _, A3 = field(fc, a); _, _, _, B3 = field(fc, b)
                L.append(f"   crossing (S3 vs S1, S2 at t = 6): max|I_A(S3) - I_A(S1)|/I0 {np.abs(A3 - A1).max() / I0:.2e}, "
                         f"max|I_B(S3) - I_B(S2)|/I0 {np.abs(B3 - B2).max() / I0:.2e}")
            d4 = os.path.join(RUNS, f"S4_{sch}{res}")
            if not os.path.isdir(d4) or not dumps(d4):
                L.append(f"   shadow: S4_{sch}{res} missing or incomplete")
                continue
            t4f, X, Y, F4 = field(dumps(d4)[-1], a)
            t1f, _, _, F1 = field(fa, a)
            if abs(t4f - t1f) > 1e-9:
                L.append(f"   shadow: S4_{sch}{res} ended at t {t4f:g} != S1 t {t1f:g} (incomplete)")
                continue
            t4, xf = faceflux(X, Y, F4, sou)
            t1, _ = faceflux(X, Y, F1, sou)
            i0 = int(np.argmin(np.abs(X - (-0.2))))                 # upstream profile column (cell centre)
            p1 = np.maximum(F1[:, i0], 0.0) * (Y[1] - Y[0])
            tau = CHI_COL * chord(Y, X[i0], nx, ny) / np.sqrt(nx * nx + ny * ny)
            tpred = (p1 * np.exp(-tau)).sum() / p1.sum()
            miss = p1[tau == 0.0].sum() / p1.sum()
            rows = []
            for xx in ((-0.5, -0.2, 0.2, 0.5) if res else (-0.5, -0.2, 0.2, 0.5, 1.0)):
                i = int(np.argmin(np.abs(xf - xx)))
                att = np.exp(-CHI_BG * (xf[i] - X1) / nx)   # background only (S1 is vacuum); 3D path = dx / nx
                rows.append(f"x {xf[i]:+.3f}: {t4[i] / (t1[i] * att):.4f}")
            gas = h5py.File(dumps(d4)[-1], "r")["prim"]
            vmax = float(np.sqrt((np.array(gas[:, 1:4]) ** 2).sum(1)).max())
            L.append(f"   shadow S4/S1 face-flux integral (background removed): " + " | ".join(rows)
                     + f" | predicted from S1's upstream profile (x {X[i0]:+.3f}): {tpred:.4f} (beam fraction missing"
                     + f" the column {miss:.4f}; unspread beam e^-11.5 = 1.0e-5) | min I/I0 {F4.min() / I0:+.2e}"
                     + f" | gas |v|max {vmax:.1e}")
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 2:   # run|check <runs subdir> <out table>  (e.g. RF10/R3: the R-F10 limiter re-runs)
        RUNS, OUT = os.path.join(MOD, "runs", sys.argv[2]), sys.argv[3]
    main(sys.argv[1])
