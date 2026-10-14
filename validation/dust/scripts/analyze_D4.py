#!/usr/bin/env python
"""D4 analysis: dust advection through periodic moving slabs (exact = the IC after 10 crossings).

Per run, from the t = 0 and final dumps (prim + cons, double precision):
  dM_dust = sum(rho f dV)_final / sum(rho f dV)_0 - 1     periodic box => only React can change it
  dM_ac   = same for rho a_c
  L1(f)   = mean |f - f_IC| / f_ref                        profile error after 10 crossings (translation = identity)
  bounds  = max f / max f_IC - 1, min f / min f_IC - 1, max a/max a_IC - 1, min a/min a_IC - 1
D4b: max |f/f_ref - 1|, max |a/a_ref - 1| (exact: 0).
D4c: dust LOST by sublimation of numerically mixed gas = -dM_dust; also the cold-side loss
     1 - sum(rho f dV over cells with T < 1100 K at the end) / initial cold-slab dust mass (not exact once mixed);
     and the trend with N (pure numerical artefact -> must fall with N if it is a resolution effect).
Writes results/tables/D4_advection.txt (+ .json); --check.
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
        cn = [n for n in names if n.startswith("cons_")]
        p, c = h["prim"][()], h["cons"][()]
        x = h["Locations"]["x"][()]
        dx = float(x[0, 1] - x[0, 0])
        y = h["Locations"]["y"][()]
        dy = float(y[0, 1] - y[0, 0]) if y.shape[1] > 2 else 1.0
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
    g = lambda n: p[:, pn.index(n)].ravel()
    return dict(t=t, rho=g("prim_density"), f=g("prim_scalar_0"), a=g("prim_scalar_1"), P=g("prim_pressure"),
                rf=c[:, cn.index("cons_scalar_0")].ravel() if "cons_scalar_0" in cn else None, dV=dx * dy)


def main():
    meta = json.load(open(os.path.join(DU, "inputs", "D4_meta.json")))
    rows, res = ["# D4 advection through periodic slabs, 10 crossings (exact = IC)"], {}
    rows.append("%-20s %5s %11s %11s %10s %10s %10s %10s %10s" % ("run", "N", "dM_dust", "dM_ac", "L1(f)", "fmax-1",
                                                                  "fmin-1", "amax-1", "amin-1"))
    for name, m in meta.items():
        if m["variant"] == "D4d":
            continue
        fs = sorted(glob.glob(os.path.join(DU, "runs", "D4", name, "*.out0.0*.phdf")))
        ff = glob.glob(os.path.join(DU, "runs", "D4", name, "*.out0.final.phdf"))
        if not fs or not ff:
            rows.append(f"{name:20s} MISSING"); continue
        i0, i1 = load(fs[0]), load(ff[0])
        M0 = (i0["rho"] * i0["f"]).sum(); M1 = (i1["rho"] * i1["f"]).sum()
        A0 = (i0["rho"] * i0["a"]).sum(); A1 = (i1["rho"] * i1["a"]).sum()
        r = dict(N=m["N"], dim=m["dim"], t=i1["t"], dM_dust=float(M1 / M0 - 1), dM_ac=float(A1 / A0 - 1),
                 L1_f=float(np.abs(i1["f"] - i0["f"]).mean() / F_REF),
                 fmax=float(i1["f"].max() / i0["f"].max() - 1), fmin=float(i1["f"].min() / i0["f"].min() - 1),
                 amax=float(i1["a"].max() / i0["a"].max() - 1), amin=float(i1["a"].min() / i0["a"].min() - 1),
                 P_spread=float(np.ptp(i1["P"]) / i1["P"].mean()))
        res[name] = r
        rows.append("%-20s %5d %+11.3e %+11.3e %10.3e %+10.2e %+10.2e %+10.2e %+10.2e" % (
            name, m["N"], r["dM_dust"], r["dM_ac"], r["L1_f"], r["fmax"], r["fmin"], r["amax"], r["amin"]))
    rows.append("# D4c / D4cprod: dust lost = -dM_dust (no exact loss: the gas never changes T in the exact solution)")
    for name, r in res.items():
        if name.startswith("D4c"):
            rows.append(f"  {name:20s} N {r['N']:5d}: lost {-r['dM_dust']:.3e} of the box dust mass, P spread {r['P_spread']:.1e}")
    txt = "\n".join(rows)
    print(txt)
    jp = os.path.join(DU, "results", "tables", "D4_advection.json")
    if "--check" in sys.argv:
        same = json.dumps(json.load(open(jp)), sort_keys=True) == json.dumps(json.loads(json.dumps(res)), sort_keys=True)
        print("CHECK", "PASS" if same else "FAIL"); sys.exit(0 if same else 1)
    open(jp.replace(".json", ".txt"), "w").write(txt + "\n")
    json.dump(res, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main()
