#!/usr/bin/env python
"""I6b: 1D planar gamma-law Noh, N 1024 (unless _N<n>), t 0.6: Mach scan (P0 1e-2 .. 1e-6) with the dual energy on /
off, and at P0 1e-6 the dual-energy keys one at a time. Exact plateau for FINITE Mach: Rankine-Hugoniot of the
reflected shock (upstream rho0 1, u -1, P0; downstream u 0) solved for (rho2, P2, s). Also the fraction of
SWITCHED cells (dual_switched dump) inside the plateau and in the shock (|x| within 3 dx of r_s).
  analyze_I6b.py [--check] -> results/tables/I6b.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, re, sys
import h5py
import numpy as np
from scipy.optimize import brentq

M = _VT + "/ideal_mhd"
G = 5 / 3


def exact(P0, v0=1.0, rho0=1.0):
    # shock moving at +s into gas with u = -v0; behind: u = 0. mass: rho0 (-v0 - s) = -rho2 s; momentum:
    # rho0 (-v0 - s)(-v0) ... use the shock-frame form: m = rho0 (v0 + s), P2 = P0 + m v0, rho2 = m / s
    def f(s):
        m = rho0 * (v0 + s)
        rho2, P2 = m / s, P0 + m * v0
        e0, e2 = P0 / ((G - 1) * rho0), P2 / ((G - 1) * rho2)
        return e2 - e0 - 0.5 * (P0 + P2) * (1 / rho0 - 1 / rho2)
    s = brentq(f, 1e-6, 10.0)
    m = rho0 * (v0 + s)
    return m / s, P0 + m * v0, s


def one(d, P0):
    f = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))[-1]
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"]); p = h["prim"][()]; x = h["Locations"]["x"][()]
    xc = np.concatenate([0.5 * (x[b, 1:] + x[b, :-1]) for b in range(p.shape[0])])
    rho = np.concatenate([p[b, 0, 0, 0] for b in range(p.shape[0])])
    r = np.abs(xc); dx = xc[1] - xc[0]
    rho2, P2, s = exact(P0)
    rs = s * t
    pl = (r > 0.5 * rs) & (r < 0.9 * rs)
    out = {"t": t, "P0": P0, "Mach": 1 / np.sqrt(G * P0), "rho2_exact": rho2, "plateau": float(rho[pl].mean() / rho2 - 1)}
    half = 0.5 * (rho2 + 1.0)
    o = np.argsort(r); rr, qq = r[o], rho[o]
    cr = rr[np.nonzero((qq[:-1] - half) * (qq[1:] - half) <= 0)[0]]
    out["r_shock_rel"] = float(np.median(cr) / rs - 1) if cr.size else None
    fs = sorted(glob.glob(os.path.join(d, "parthenon.out2.*.phdf")))
    if fs and "dual_switched" in h5py.File(fs[-1], "r"):   # absent when the dual energy is off
        with h5py.File(fs[-1], "r") as h:
            sw = np.concatenate([h["dual_switched"][()][b].ravel() for b in range(h["dual_switched"].shape[0])])
        out["sw_plateau"] = float((sw[pl] != 0).mean())
        out["sw_shock"] = float((sw[np.abs(r - rs) < 3 * dx] != 0).mean())
        out["sw_upstream"] = float((sw[(r > rs + 3 * dx) & (r < 0.35)] != 0).mean())
    return out


def main():
    res = {}
    for d in sorted(glob.glob(f"{M}/runs/I6b/noh1d_*")):
        n = os.path.basename(d)
        if not glob.glob(d + "/parthenon.out0.*.phdf"):
            continue
        P0 = float(re.search(r"_P([0-9.e-]+?)_", n + "_").group(1))
        res[n] = one(d, P0)
    js = f"{M}/results/tables/I6b.json"
    if "--check" in sys.argv:
        ok = json.load(open(js)) == json.loads(json.dumps(res)); print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    json.dump(res, open(js, "w"), indent=1)
    f = lambda v: "-" if v is None else f"{v:+.2e}"
    L = ["I6b 1D Noh t 0.6: plateau = mean rho(0.5..0.9 r_s)/rho2_exact - 1 (finite-Mach RH); sw_* = switched fraction",
         f"{'run':<34} {'Mach':>6} {'rho2':>7} {'plateau':>9} {'dr_s/r_s':>9} {'sw_plat':>7} {'sw_shock':>8} {'sw_up':>6}"]
    for k, r in res.items():
        L.append(f"{k:<34} {r['Mach']:6.0f} {r['rho2_exact']:7.4f} {f(r['plateau']):>9} {f(r['r_shock_rel']):>9} "
                 + " ".join(f"{r[q]:7.3f}" if q in r else "      -" for q in ("sw_plateau", "sw_shock", "sw_upstream")))
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
