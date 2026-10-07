#!/usr/bin/env python
"""E7r: double flux with mesh refinement. Pure moving contacts (R4 star states) crossing static patches (smr) or
with the fine/coarse boundary riding on the contact (amr, maxdensity criterion). For every output snapshot:
max |P/P*-1| (P* = median initial pressure; exact solution is uniform P*), the refinement levels present, and the
total energy (periodic 2D: change over the run; 1D outflow: dflux vs std at equal time).
usage: e7r_eval.py [RUNDIR=E7r] [OUT=E7r_amr.txt]"""
import glob, os, sys
import h5py
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs", sys.argv[1] if len(sys.argv) > 1 else "E7r")
OUT = os.path.join(HERE, "..", "results", "tables", sys.argv[2] if len(sys.argv) > 2 else "E7r_amr.txt")


def snap(f):
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float); c = h["cons"][()].astype(float)
        lv = h["Levels"][()] if "Levels" in h else np.zeros(p.shape[0], int)
        xs, ys, zs = h["Locations/x"][()], h["Locations/y"][()], h["Locations/z"][()]
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
    vol = np.stack([np.diff(xs[b])[None, None, :] * np.diff(ys[b])[None, :, None] * np.diff(zs[b])[:, None, None]
                    for b in range(p.shape[0])])
    return dict(P=p[:, 4], E=(c[:, 4] * vol).sum(), levels=sorted(set(int(x) for x in lv)), t=t, nb=p.shape[0])


def series(d):
    fs = sorted(glob.glob(os.path.join(d, "parthenon.out0.0*.phdf"))) + glob.glob(os.path.join(d, "*.out0.final.phdf"))
    return [snap(f) for f in fs]


lines = ["E7r double flux + mesh refinement: pure moving contacts, R4 star states. Per run: max over snapshots of",
         "max|P/P*-1| (final value), refinement levels seen, blocks at the end, energy.",
         "case                      variant | max|P/P*-1| all snaps (final) | levels seen | nblocks end | dE/E"]
cases = sorted({os.path.basename(d).rsplit("_", 1)[0] for d in glob.glob(os.path.join(RUNS, "E7r_*"))})
for c in cases:
    S = {}
    for v in ["dflux", "std"]:
        d = os.path.join(RUNS, f"{c}_{v}")
        if not glob.glob(os.path.join(d, "*.final.phdf")):
            lines.append(f"{c:25s} {v:5s} | missing"); continue
        S[v] = series(d)
    for v, s in S.items():
        Ps = np.median(s[0]["P"])
        errs = [np.abs(x["P"] / Ps - 1).max() for x in s]
        lev = sorted(set(l for x in s for l in x["levels"]))
        if "2d" in c:
            dE = f"run {s[-1]['E'] / s[0]['E'] - 1:+.2e}"
        else:
            dE = (f"vs std {s[-1]['E'] / S['std'][-1]['E'] - 1:+.2e}" if v == "dflux" and "std" in S else "--")
        lines.append(f"{c:25s} {v:5s} | {max(errs):.2e} ({errs[-1]:.2e}) | {lev} | {s[-1]['nb']} | {dE}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
