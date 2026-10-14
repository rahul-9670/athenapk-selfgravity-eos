#!/usr/bin/env python
"""D6 (AMR, structured dust field) and D7 (production box boundary) analysis.

Cell volumes from each block's own Locations (works on AMR meshes). Per run, t = 0 dump vs every later dump:
  dM_dust(t) = sum(rho f dV)/sum(rho f dV)_0 - 1   (D6: periodic -> only React can change it)
  D7: closure = sum(rho f dV) / (f_ref sum(rho dV)) - 1 (exact 0: f stays f_ref everywhere), max |f/f_ref - 1|;
      frz: max |a/a_ref - 1| (exact 0).
  bounds: max f/f_ref - 1 (must be <= 0: anything above f_ref is clamped by React = permanent loss + a ratchet),
          min a / min a_0 - 1.
  levels: the refinement levels present per dump and the number of blocks (proof that regridding happened).
Writes results/tables/D6_amr.txt, D7_boundary.txt (+ .json); --check.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import h5py
import numpy as np
DU = _VT + "/dust"
F_REF, A_REF = 0.01, 1e-5


def load(fn):
    with h5py.File(fn) as h:
        names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
        pn = [n for n in names if n.startswith("prim_")]
        p = h["prim"][()]
        x, y = h["Locations"]["x"][()], h["Locations"]["y"][()]
        lev = h["Levels"][()]
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
    dV = ((x[:, 1] - x[:, 0]) * (y[:, 1] - y[:, 0]))[:, None, None, None] * np.ones(p[:, 0].shape)
    g = lambda n: p[:, pn.index(n)]
    return dict(t=t, rho=g("prim_density").ravel(), f=g("prim_scalar_0").ravel(), a=g("prim_scalar_1").ravel(),
                dV=dV.ravel(), lev=lev, nb=p.shape[0])


def series(run):
    fs = sorted(glob.glob(os.path.join(DU, "runs", run, "*.out0.0*.phdf"))) + \
        glob.glob(os.path.join(DU, "runs", run, "*.out0.final.phdf"))
    return [load(f) for f in fs]


def main():
    out, rows6, rows7 = {}, ["# D6 structured dust field through AMR regridding (5 crossings)"], \
        ["# D7 production box boundary (diode + clamp + wall), uniform oblique flow, 2 crossings"]
    for name in sorted(json.load(open(os.path.join(DU, "inputs", "D6_meta.json")))):
        S = series(f"D6/{name}")
        if not S:
            rows6.append(f"{name}: MISSING"); continue
        M0 = (S[0]["rho"] * S[0]["f"] * S[0]["dV"]).sum()
        dM = [float((s["rho"] * s["f"] * s["dV"]).sum() / M0 - 1) for s in S]
        r = dict(dM_final=dM[-1], dM_min=min(dM), fmax=float(max(s["f"].max() for s in S) / F_REF - 1),
                 amin=float(min(s["a"].min() for s in S) / S[0]["a"].min() - 1),
                 nblocks=[int(s["nb"]) for s in S], levels=[sorted(set(int(l) for l in s["lev"])) for s in S])
        out[name] = r
        rows6.append(f"{name:12s} dM_dust final {r['dM_final']:+.3e} (min {r['dM_min']:+.3e}) | max f/f_ref - 1 "
                     f"{r['fmax']:+.2e} | min a / min a0 - 1 {r['amin']:+.2e} | blocks {r['nblocks']} | levels "
                     f"{r['levels'][0]} .. {r['levels'][-1]}")
    for name in sorted(json.load(open(os.path.join(DU, "inputs", "D7_meta.json")))):
        S = series(f"D7/{name}")
        if not S:
            rows7.append(f"{name}: MISSING"); continue
        clo = [float((s["rho"] * s["f"] * s["dV"]).sum() / (F_REF * (s["rho"] * s["dV"]).sum()) - 1) for s in S]
        fdev = max(float(np.abs(s["f"] / F_REF - 1).max()) for s in S)
        adev = max(float(np.abs(s["a"] / A_REF - 1).max()) for s in S)
        mass = [float((s["rho"] * s["dV"]).sum() / (S[0]["rho"] * S[0]["dV"]).sum()) for s in S]
        r = dict(closure_max=float(np.abs(clo).max()), fdev_max=fdev, adev_max=adev, gas_mass=mass)
        out[name] = r
        rows7.append(f"{name:12s} max |sum(rho f)/(f_ref sum rho) - 1| {r['closure_max']:.2e} | max |f/f_ref - 1| "
                     f"{fdev:.2e} | max |a/a_ref - 1| {adev:.2e} | gas mass / initial {mass[-1]:.4f}")
    txt6, txt7 = "\n".join(rows6), "\n".join(rows7)
    print(txt6); print(txt7)
    jp = os.path.join(DU, "results", "tables", "D67.json")
    if "--check" in sys.argv:
        ok = json.dumps(json.load(open(jp)), sort_keys=True) == json.dumps(json.loads(json.dumps(out)), sort_keys=True)
        print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
    open(os.path.join(DU, "results", "tables", "D6_amr.txt"), "w").write(txt6 + "\n")
    open(os.path.join(DU, "results", "tables", "D7_boundary.txt"), "w").write(txt7 + "\n")
    json.dump(out, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main()
