#!/usr/bin/env python
"""E7q: double flux in 2D/3D. Periodic oblique slabs of pure contacts (R4 star states): the exact solution is a
translation along k at the uniform initial pressure P*. Reports, key on vs off: max and L1 of |P/P* - 1| at t_final,
the relative change of total energy over the run (periodic box: the standard scheme conserves it to round-off), and
two invariance checks (bitwise): meshblock decomposition, and k = (1,0) on a 2D grid vs the 1D run, row by row.
usage: e7q_eval.py [RUNDIR=E7q] [OUT=E7q_slab.txt]"""
import glob, os, sys
import h5py
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs", sys.argv[1] if len(sys.argv) > 1 else "E7q")
OUT = os.path.join(HERE, "..", "results", "tables", sys.argv[2] if len(sys.argv) > 2 else "E7q_slab.txt")


def load(f):
    """flatten all blocks -> arrays sorted by (z, y, x); returns dict of prim rho,P and cons E, cell volume"""
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float)        # [b, comp, k, j, i]
        c = h["cons"][()].astype(float)
        xs, ys, zs = h["Locations/x"][()], h["Locations/y"][()], h["Locations/z"][()]
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        ncyc = int(np.atleast_1d(h["Info"].attrs["NCycle"])[0])
        names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
    cols = []
    for b in range(p.shape[0]):
        xc = 0.5 * (xs[b, 1:] + xs[b, :-1]); yc = 0.5 * (ys[b, 1:] + ys[b, :-1]); zc = 0.5 * (zs[b, 1:] + zs[b, :-1])
        Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
        vol = np.diff(xs[b])[None, None, :] * np.diff(ys[b])[None, :, None] * np.diff(zs[b])[:, None, None]
        cols.append(np.stack([Z.ravel(), Y.ravel(), X.ravel(), p[b, 0].ravel(), p[b, 4].ravel(),
                              c[b, 4].ravel(), np.broadcast_to(vol, Z.shape).ravel()]))
    a = np.concatenate(cols, axis=1)
    o = np.lexsort((a[2], a[1], a[0]))
    a = a[:, o]
    return dict(z=a[0], y=a[1], x=a[2], rho=a[3], P=a[4], E=a[5], vol=a[6], t=t, ncyc=ncyc)


def final(d):
    f = glob.glob(os.path.join(d, "*.out0.final.phdf"))
    return f[0] if f else None


lines = ["E7q double flux in 2D/3D: periodic oblique slab of pure contacts, R4 star states (ionization regime).",
         "case                          | double flux: max / L1 |P/P*-1|, dE/E(run) | standard: max / L1, dE/E(run) | cycles"]
cases = sorted({os.path.basename(d).rsplit("_", 1)[0] for d in glob.glob(os.path.join(RUNS, "E7q_*"))})
for c in cases:
    row = f"{c:30s}"
    ok = True
    for v in ["dflux", "std"]:
        d = os.path.join(RUNS, f"{c}_{v}")
        f1 = final(d)
        if f1 is None:
            row += f" | {v} missing"; ok = False; continue
        a0, a1 = load(os.path.join(d, "parthenon.out0.00000.phdf")), load(f1)
        Ps = np.median(a0["P"])
        e = np.abs(a1["P"] / Ps - 1.0)
        E0, E1 = (a0["E"] * a0["vol"]).sum(), (a1["E"] * a1["vol"]).sum()
        row += f" | {e.max():.2e} / {e.mean():.2e}, {E1 / E0 - 1:+.2e}"
        if v == "std": row += f" | {a1['ncyc']}"
    lines.append(row)
lines.append("")
lines.append("invariance checks (final states, rho and P and E compared bitwise):")
def same(da, db, rows=False):
    fa, fb = final(da), final(db)
    if fa is None or fb is None: return "missing"
    a, b = load(fa), load(fb)
    if rows:   # 2D grid with k=(1,0): every row j must equal the 1D run
        n1 = b["rho"].size
        res = all(np.array_equal(a[k].reshape(-1, n1), np.broadcast_to(b[k], (a[k].size // n1, n1))) for k in ["rho", "P", "E"])
        mx = max(np.abs(a[k].reshape(-1, n1) / b[k][None, :] - 1).max() for k in ["rho", "P", "E"])
    else:
        res = all(np.array_equal(a[k], b[k]) for k in ["rho", "P", "E"])
        mx = max(np.abs(a[k] / b[k] - 1).max() for k in ["rho", "P", "E"])
    return ("BITWISE" if res else "DIFFER") + f" (max rel {mx:.1e})"
for v in ["dflux", "std"]:
    lines.append(f"  2D n128 32^2 blocks/4 ranks vs 1 block  [{v}]: " + same(f"{RUNS}/E7q_slab11_2d_n128_mb32_{v}", f"{RUNS}/E7q_slab11_2d_n128_{v}"))
    lines.append(f"  3D n64 16^3 blocks/8 ranks vs 1 block   [{v}]: " + same(f"{RUNS}/E7q_slab111_3d_n64_mb16_{v}", f"{RUNS}/E7q_slab111_3d_n64_{v}"))
    lines.append(f"  2D grid k=(1,0) rows vs 1D run          [{v}]: " + same(f"{RUNS}/E7q_slab10_x2d_n128_{v}", f"{RUNS}/E7q_slab10_1d_n128_{v}", rows=True))
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
