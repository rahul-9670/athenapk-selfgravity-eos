#!/usr/bin/env python
"""I7 analysis: outgoing waves through diode + boundary_flux_clamp (production) and through outflow, against the
[0,1] part of the same problem in the large box [-4,5] (same dx; nothing reaches its boundary by tlim).

Per problem and boundary kind, at every dump time: L1 over [0,1] and max |difference| of rho, P (tubes) or
Bx, psi (div-B pulse), plus the total mass in [0,1] relative to the reference (tubes). Reflections show up as a
max difference far above the truncation error the two runs share (they share it exactly: same dx, same scheme,
the only difference is what happens at x = 0 and x = 1).
  analyze_I7.py [--check] -> results/tables/I7.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import h5py
import numpy as np

M = _VT + "/ideal_mhd"


def load(f, var):
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"])
        q = h[var][()]
        x = h["Locations"]["x"][()]
    xc = np.concatenate([0.5 * (x[b, 1:] + x[b, :-1]) for b in range(q.shape[0])])
    v = np.concatenate([q[b, :, 0, 0, :] for b in range(q.shape[0])], axis=1)
    o = np.argsort(xc)
    return t, xc[o], v[:, o]


def compare(prob, kind):
    var, comps = ("prim", {"rho": 0, "P": 4, "By": 6}) if not prob.startswith("divb") else ("cons", {"Bx": 5, "psi": 8})
    fa = sorted(glob.glob(f"{M}/runs/I7/I7_{prob}_{kind}/parthenon.out0.*.phdf"))
    fr = sorted(glob.glob(f"{M}/runs/I7/I7_{prob}_ref/parthenon.out0.*.phdf"))
    rows = []
    for a, r in zip(fa, fr):
        ta, xa, qa = load(a, var)
        tr, xr, qr = load(r, var)
        if abs(ta - tr) > 1e-12:
            rows.append({"t": ta, "ERROR": f"time mismatch {ta} vs {tr}"})
            continue
        m = (xr > 0) & (xr < 1)
        assert np.allclose(xr[m], xa), "grids differ"
        dx = xa[1] - xa[0]
        row = {"t": round(ta, 6)}
        for nm, i in comps.items():
            d = qa[i] - qr[i][m]
            row[f"L1_{nm}"] = float(np.abs(d).sum() * dx)
            row[f"max_{nm}"] = float(np.abs(d).max())
            row[f"scale_{nm}"] = float(np.abs(qr[i][m]).max())
        if var == "prim":
            row["mass_rel"] = float(qa[0].sum() / qr[0][m].sum() - 1)
        rows.append(row)
    return rows


def main():
    res = {}
    for prob in ("BW", "RJ2a_div", "divb_a0", "divb_prod"):
        for kind in ("diode", "diodewall", "outfl"):
            if glob.glob(f"{M}/runs/I7/I7_{prob}_{kind}/*.phdf"):
                res[f"{prob}_{kind}"] = compare(prob, kind)
    js = f"{M}/results/tables/I7.json"
    if "--check" in sys.argv:
        ok = json.load(open(js)) == json.loads(json.dumps(res))
        print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    L = ["I7 outgoing waves: [0,1] run minus the [0,1] part of the [-4,5] reference (same dx). max = max |diff|; "
         "scale = max |reference| in [0,1]"]
    for k, rows in res.items():
        L.append(k)
        for r in rows:
            L.append("   t %.3f  " % r["t"] + "  ".join(
                f"{q[3:]}: L1 {r[q]:.2e} max {r['max_' + q[3:]]:.2e} (scale {r['scale_' + q[3:]]:.2e})"
                for q in r if q.startswith("L1_")) + (f"  mass {r['mass_rel']:+.2e}" if "mass_rel" in r else ""))
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
