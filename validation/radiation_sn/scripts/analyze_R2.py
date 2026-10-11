#!/usr/bin/env python
"""R2 Su-Olson (linearised, +A/-A pairs) vs the semi-analytic reference (su_olson_ref.py).

Per arm: U = (E_r - a T0^4)/(A a T0^4), V = (a T^4 - a T0^4)/(A a T0^4) from the +A and -A runs; the pair mean cancels
the even orders of the linearisation, the half-difference |U(+A) - U(-A)|/2 measures them. Compared at the dumps
nearest tau = 0.1, 1, 3.16, 10 (at the dump's own tau), as cell averages, on |z| <= 6, against:
  ref(code set)  = the code's own slab cosines (isolates the spatial + temporal + solver error),
  ref(S64)       = Gauss-Legendre S64 (the angular-converged transport answer: adds the 32-ordinate angular error).
usage: analyze_R2.py run | check   -> results/tables/R2_su_olson.txt
"""
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import su_olson_ref as SO  # noqa: E402
from make_decks_R2 import ARMS, ARAD, T0, C  # noqa: E402
from rad_common import MOD  # noqa: E402

RUNS = os.path.join(MOD, "runs", "R2")
OUT = os.path.join(MOD, "results", "tables", "R2_su_olson.txt")
TAUW = [0.1, 1.0, 3.16228, 10.0]
ZMAX = 6.0


def load(d, A):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
        with h5py.File(f, "r") as h:
            tau = float(h["Info"].attrs["Time"]) * C
            x = np.array(h["Locations"]["x"])
            xc = 0.5 * (x[:, 1:] + x[:, :-1]).ravel()
            p = np.array(h["prim"])[:, :, 0, 0, :].reshape(p_shape(h))
            Er = np.array(h["rad.Er"]).reshape(-1)
            T = (p[4] / p[0])
            out[round(tau, 6)] = (xc, (Er - ARAD * T0 ** 4) / (A * ARAD * T0 ** 4),
                                  (ARAD * T ** 4 - ARAD * T0 ** 4) / (A * ARAD * T0 ** 4))
    return out


def p_shape(h):
    nb, nv, _, _, nx = h["prim"].shape
    return (nv, nb * nx)


def nearest(d, tau):
    k = min(d, key=lambda t: abs(t - tau))
    return k, d[k]


def main(mode):
    L = ["# R2 Su-Olson (scripts/analyze_R2.py): errors of the +A/-A pair mean vs the semi-analytic reference, |z| <= 6;",
         "# L1 = sum|code-ref| / sum|ref|, max = max|code-ref| / max|ref|; lin = max|U(+A)-U(-A)|/2 / max|U| (linearisation)"]
    refs = {}
    for arm, (nx, dtau, ov) in ARMS.items():
        P = load(os.path.join(RUNS, f"{arm}_p"), 2.0e-3)
        M = load(os.path.join(RUNS, f"{arm}_m"), -2.0e-3)
        if not P or not M:
            L.append(f"== {arm}: MISSING dumps")
            continue
        L.append(f"== {arm} (nx {nx}, dtau {dtau}, {ov or 'base keys'})")
        for tw in TAUW:
            tau, (xc, Up, Vp) = nearest(P, tw)
            _, (_, Um, Vm) = nearest(M, tw)
            m = np.abs(xc) <= ZMAX
            z, h = xc[m], 40.0 / nx
            U, V = 0.5 * (Up + Um)[m], 0.5 * (Vp + Vm)[m]
            lin = np.max(np.abs(Up - Um)[m]) / 2 / np.max(np.abs(U))
            angset = "816" if arm == "ang816" else "48"
            for key, (mu, w) in ((("code" + angset), SO.code_slab(8, 16) if angset == "816" else SO.code_slab()),
                                 ("S64", SO.gauss_slab(64))):
                rk = (key, nx, round(tau, 6))
                if rk not in refs:
                    refs[rk] = SO.solve([tau], z, h, mu, w)
                Ur, Vr = refs[rk][0][0], refs[rk][1][0]
                eU = (np.sum(np.abs(U - Ur)) / np.sum(np.abs(Ur)), np.max(np.abs(U - Ur)) / np.max(np.abs(Ur)))
                eV = (np.sum(np.abs(V - Vr)) / np.sum(np.abs(Vr)), np.max(np.abs(V - Vr)) / np.max(np.abs(Vr)))
                i0 = np.argmin(np.abs(z - 0.0))
                L.append(f"   tau {tau:7.4f} vs ref({key:6s}): U L1 {eU[0]:.2e} max {eU[1]:.2e} | V L1 {eV[0]:.2e} "
                         f"max {eV[1]:.2e} | U(0) code {U[i0]:.5f} ref {Ur[i0]:.5f} | lin {lin:.1e}")
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
