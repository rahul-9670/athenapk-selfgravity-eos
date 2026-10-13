#!/usr/bin/env python
"""I4 analysis: field-loop advection (exact solution at t = 1 = the initial condition, periodic return).

Per snapshot (cons dumps at t = 0, 0.25, ..., 1):
  ME/ME0      magnetic energy relative to t = 0 (Gardiner & Stone 2005 Fig. "magnetic energy decay")
  divB_mean   volume mean of |div B| dx / B0, B0 = amp = 1e-3 (the loop field, as the pgen's UserRelDivB), central
              differences (the operator dedner_source.cpp uses), dx = cell width
  divB_max    the worst cell of the same quantity, and where it is: r_perp / R of that cell (loop edge = 1)
At t = 1: L1(|B| - |B|_IC) / L1(|B|_IC) and L1(rho - rho_IC) (drat run: the contact).
  analyze_I4.py [--check] -> results/tables/I4.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, re, sys
import h5py
import numpy as np

M = _VT + "/ideal_mhd"
B0, R = 1.0e-3, 0.3


def load(f):
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"])
        c = h["cons"][()]
        loc = [h["Locations"][a][()] for a in "xyz"]
    return t, c, loc


def divb(c, loc, dim, t):
    """per block: |div B| on interior cells (one-cell rim dropped), returns (values*dx/B0, rperp of those cells, dV)."""
    vals, rp, dvs = [], [], []
    for b in range(c.shape[0]):
        dx = [l[b, 1] - l[b, 0] for l in loc]
        sl = (slice(1, -1) if dim > 2 else slice(None), slice(1, -1), slice(1, -1))
        D = (c[b, 5][sl[0], sl[1], 2:] - c[b, 5][sl[0], sl[1], :-2]) / (2 * dx[0]) + \
            (c[b, 6][sl[0], 2:, sl[2]] - c[b, 6][sl[0], :-2, sl[2]]) / (2 * dx[1])
        if dim > 2:
            D = D + (c[b, 7][2:, sl[1], sl[2]] - c[b, 7][:-2, sl[1], sl[2]]) / (2 * dx[2])
        cen = [0.5 * (l[b, 1:] + l[b, :-1]) for l in loc]
        # coordinates relative to the ADVECTED loop centre, periodic (2D v = (2, 1) on 2 x 1; 3D v = (1, 1, 2) on 1 x 1 x 2)
        wrap = lambda q, L: (q + 0.5 * L) % L - 0.5 * L
        if dim == 2:
            X, Y = wrap(cen[0][1:-1] - 2 * t, 2.0), wrap(cen[1][1:-1] - t, 1.0)
        else:
            X, Y = wrap(cen[0][1:-1] - t, 1.0), wrap(cen[1][1:-1] - t, 1.0)
        if dim == 2:
            r = np.sqrt(X[None, :] ** 2 + Y[:, None] ** 2)[None]
        else:
            # iprob 4: loop axis along x2 in the frame rotated in x1-x3; r_perp from the (x', y) plane, x' periodic
            Z = wrap(cen[2][1:-1] - 2 * t, 2.0)
            a2 = np.arctan(1.0 / 2.0)  # field_loop.cpp iprob 4: ang_2 = atan(x1size / x3size), box 1 x 1 x 2
            lam = 1.0 * np.cos(a2) if np.cos(a2) >= np.sin(a2) else 2.0 * np.sin(a2)
            xp = X[None, None, :] * np.cos(a2) + Z[:, None, None] * np.sin(a2)
            xp = (xp + 0.5 * lam) % lam - 0.5 * lam
            r = np.sqrt(xp**2 + Y[None, :, None] ** 2)
        vals.append((np.abs(D) * min(dx[:dim]) / B0).ravel())
        rp.append(np.broadcast_to(r, D.shape).ravel())
        dvs.append(np.full(D.size, np.prod(dx[:dim])))
    return np.concatenate(vals), np.concatenate(rp), np.concatenate(dvs)


def score(d, dim):
    fs = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))
    rows, ic = [], None
    for f in fs:
        t, c, loc = load(f)
        dV = np.array([np.prod([l[b, 1] - l[b, 0] for l in loc[:dim]]) for b in range(c.shape[0])])
        ME = float((0.5 * (c[:, 5] ** 2 + c[:, 6] ** 2 + c[:, 7] ** 2)).sum(axis=(1, 2, 3)) @ dV)
        v, rp, dv = divb(c, loc, dim, t)
        i = int(np.argmax(v))
        absB = np.sqrt(c[:, 5] ** 2 + c[:, 6] ** 2 + c[:, 7] ** 2)
        if ic is None:
            ic = (ME, absB, c[:, 0])
        rows.append({"t": round(t, 6), "ME_ME0": ME / ic[0], "divB_mean": float((v * dv).sum() / dv.sum()),
                     "divB_max": float(v[i]), "rperp_R_of_max": float(rp[i] / R)})
    l1B = float(np.abs(absB - ic[1]).sum() / np.abs(ic[1]).sum())
    l1rho = float(np.abs(c[:, 0] - ic[2]).mean())
    return {"rows": rows, "L1_absB_rel_final": l1B, "L1_rho_final": l1rho}


def collect():
    res = {}
    for d in sorted(glob.glob(f"{M}/runs/I4/I4_*")):
        name = os.path.basename(d)
        if len(glob.glob(os.path.join(d, "*.phdf"))) >= 5:
            res[name] = dict(dim=int(name[3]), **score(d, int(name[3])))
    return res


def text(res):
    L = ["I4 field loop (beta 2e6, |B| = 1e-3): t = 1 is the exact return. divB in units of B0/dx (|div B| dx / B0)",
         f"{'run':<20} {'ME/ME0(t=1)':>11} {'L1|B| rel':>10} {'L1 rho':>9} {'<divB>':>9} {'max divB':>9} {'r/R@max':>8}"
         f"   max divB over t"]
    for k, r in res.items():
        f = r["rows"][-1]
        L.append(f"{k:<20} {f['ME_ME0']:11.4f} {r['L1_absB_rel_final']:10.3e} {r['L1_rho_final']:9.2e} "
                 f"{f['divB_mean']:9.2e} {f['divB_max']:9.2e} {f['rperp_R_of_max']:8.2f}   "
                 + " ".join(f"{x['divB_max']:.2e}" for x in r["rows"]))
    return "\n".join(L)


def main():
    res = collect()
    js = f"{M}/results/tables/I4.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        ok = old.keys() == res.keys() and all(abs(old[k]["L1_absB_rel_final"] - res[k]["L1_absB_rel_final"]) <= 1e-12
                                              for k in old)
        print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    t = text(res)
    open(js.replace(".json", ".txt"), "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
