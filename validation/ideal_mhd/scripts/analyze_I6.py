#!/usr/bin/env python
"""I6 analysis: gamma-law Noh (planar / cylindrical / spherical) vs the exact solution at t = 0.6.

Exact (Noh 1987, gamma 5/3, v0 1, rho0 1): r_s = 0.2; inside rho2 = 4^d, u = 0, P2 = (g-1) rho2 / 2; outside
rho = (1 + t/r)^(d-1), |u| = 1, and (P0 > 0) the cold upstream follows its adiabat P = P0 (rho/rho0)^g.
Metrics per run (cells with r < 0.35, i.e. clear of the boundary disturbance, which starts at r = 0.4):
  L1_rho          volume-weighted L1 of rho vs exact, divided by the volume (r < 0.35)
  plateau         mean rho in 0.5 r_s < r < 0.9 r_s, relative error vs rho2
  wall            mean rho of the cells with r < 0.1 r_s, relative to rho2 (Noh wall heating: known deficit)
  r_shock         radius where rho crosses (rho2 + rho_pre(r_s))/2 (median over cells near the jump)
  pre_min/pre_max min / max of P / P_adiabat(rho) over upstream cells max(1.1 r_s, r_s + 3 dx) < r < 0.35 (X2 analogue: < 1 = the
                  undershoot, > 1 = pre-heating)
  aniso (2D/3D)   plateau mean along the axes minus along the diagonals, relative (grid imprint)
  analyze_I6.py [--check] -> results/tables/I6.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, re, sys
import h5py
import numpy as np

M = _VT + "/ideal_mhd"
G, T, RS, P0 = 5 / 3, 0.6, 0.2, 1.0e-6
PAT = re.compile(r"noh_(\d)d_(\w+?)_N(\d+)$")  # cases: prod, nd, dc, se1 (shock_eta1)


def cells(f, dim):
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"])
        p = h["prim"][()]
        loc = [h["Locations"][a][()] for a in "xyz"]
    nb = p.shape[0]
    cen = [0.5 * (l[:, 1:] + l[:, :-1]) for l in loc]
    X = np.broadcast_to(cen[0][:, None, None, :], (nb,) + p.shape[2:])
    Y = np.broadcast_to(cen[1][:, None, :, None], (nb,) + p.shape[2:])
    Z = np.broadcast_to(cen[2][:, :, None, None], (nb,) + p.shape[2:])
    r = np.sqrt(X**2 + (Y**2 if dim > 1 else 0) + (Z**2 if dim > 2 else 0)).ravel()
    ax = np.abs(np.stack([X.ravel(), Y.ravel(), Z.ravel()][:dim]))
    return t, r, ax, p[:, 0].ravel(), p[:, 4].ravel()


def score(d, dim):
    f = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))[-1]
    t, r, ax, rho, P = cells(f, dim)
    rho2 = ((G + 1) / (G - 1)) ** dim
    rs = RS * t / T
    ex = np.where(r < rs, rho2, (1 + t / r) ** (dim - 1))
    w = r < 0.35
    out = {"t": t, "L1_rho": float(np.abs(rho - ex)[w].mean())}
    pl = (r > 0.5 * rs) & (r < 0.9 * rs)
    out["plateau"] = float(rho[pl].mean() / rho2 - 1)
    out["wall"] = float(rho[r < 0.1 * rs].mean() / rho2 - 1) if np.any(r < 0.1 * rs) else None
    pre_at = (1 + t / rs) ** (dim - 1)
    half = 0.5 * (rho2 + pre_at)
    near = (r > 0.6 * rs) & (r < 1.5 * rs)
    rr, qq = r[near], rho[near]
    o = np.argsort(rr)
    rr, qq = rr[o], qq[o]
    cross = rr[np.nonzero((qq[:-1] - half) * (qq[1:] - half) <= 0)[0]]
    out["r_shock"] = float(np.median(cross)) if cross.size else None
    dx = 2.0 / round(len(r) ** (1.0 / dim))                      # uniform box [-1,1]^d
    up = (r > max(1.1 * rs, rs + 3 * dx)) & (r < 0.35)           # clear of the smeared shock (>= 3 cells)
    q = P[up] / (P0 * rho[up] ** G)
    out["pre_min"], out["pre_max"] = (float(q.min()), float(q.max())) if q.size else (None, None)  # empty at 3D N 32
    if dim > 1:
        axis = pl & (np.sort(ax, axis=0)[-2] < 0.15 * r)            # within ~9 deg of an axis
        diag = pl & (np.abs(ax.max(0) - ax.min(0)) < 0.15 * r) if dim == 2 else \
            pl & ((ax.max(0) - ax.min(0)) < 0.25 * r)               # near the (1,1[,1]) diagonals
        out["aniso"] = float((rho[axis].mean() - rho[diag].mean()) / rho2) if axis.any() and diag.any() else None
    return out


def collect():
    res = {}
    for d in sorted(glob.glob(f"{M}/runs/I6/noh_*")):
        m = PAT.match(os.path.basename(d))
        if m and glob.glob(os.path.join(d, "*.final.phdf")) or (m and len(glob.glob(os.path.join(d, "*.phdf"))) >= 4):
            res[os.path.basename(d)] = dict(dim=int(m.group(1)), case=m.group(2), N=int(m.group(3)),
                                            **score(d, int(m.group(1))))
    return res


def text(res):
    f = lambda v, s="{:+.2e}": "-" if v is None else s.format(v)
    L = ["I6 gamma-law Noh at t = 0.6 vs exact (rho2 = 4 / 16 / 64, r_s = 0.2). pre = P / (P0 (rho/rho0)^g) upstream.",
         f"{'run':<22} {'L1 rho':>9} {'plateau':>9} {'wall':>9} {'r_shock':>8} {'pre_min':>9} {'pre_max':>9} {'aniso':>9}"]
    for k, r in res.items():
        L.append(f"{k:<22} {r['L1_rho']:9.3e} {f(r['plateau']):>9} {f(r['wall']):>9} {f(r['r_shock'], '{:.4f}'):>8} "
                 f"{f(r['pre_min'], '{:.3e}'):>9} {f(r['pre_max'], '{:.3e}'):>9} {f(r.get('aniso')):>9}")
    return "\n".join(L)


def main():
    res = collect()
    js = f"{M}/results/tables/I6.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        ok = old.keys() == res.keys() and all(abs(old[k]["L1_rho"] - res[k]["L1_rho"]) <= 1e-12 * old[k]["L1_rho"] for k in old)
        print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    t = text(res)
    open(js.replace(".json", ".txt"), "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
