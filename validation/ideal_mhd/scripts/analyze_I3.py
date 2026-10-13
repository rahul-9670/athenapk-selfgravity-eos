#!/usr/bin/env python
"""I3 analysis: MHD shock tubes vs the exact solution (RJ2a, RJ2a_div: exact_mhd_riemann.py) or the converged
N = 16384 run of the same code (BW, D1 (b)).

Error: L1 over x in [0,1] of rho, vx, vy, vz, P, By, Bz (primitive, from the final prim dump), per N; order =
log2 ratio of successive N. For discontinuous solutions the expected L1 order is ~1 (shocks) and lower at contacts
(~1/2 .. 2/3 for a 2nd-order scheme); rarefaction-only problems approach 2nd order away from wave heads/tails.
The N = 16384 run is also scored against the exact solution: it is the cross-check that the exact solver and the code
agree (if the solver had a wrong wave family, the finest run would not converge to it).
  analyze_I3.py [--check]  -> results/tables/I3.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, math, os, re, sys
import h5py
import numpy as np

sys.path.insert(0, _VT + "/ideal_mhd/scripts")
import exact_mhd_riemann as E  # noqa: E402

M = _VT + "/ideal_mhd"
VARS = ["rho", "vx", "vy", "vz", "P", "Bx", "By", "Bz"]
PAT = re.compile(r"I3_(\w+?)_N(\d+)$")


def load(d):
    f = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))[-1]
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"])
        p = h["prim"][()]
        x = h["Locations"]["x"][()]
    xc = np.concatenate([0.5 * (x[b, 1:] + x[b, :-1]) for b in range(p.shape[0])])
    q = np.concatenate([p[b, :8, 0, 0, :] for b in range(p.shape[0])], axis=1)
    o = np.argsort(xc)
    return t, xc[o], q[:, o]


_rs = {}


def exact(name, xc, t):
    if name not in _rs:
        _rs[name] = E.build(name, nfan=2000)[0]
    rs = _rs[name]
    return np.array([rs.sample((x - 0.5) / t) for x in xc]).T, rs


def collect():
    runs = {}
    for d in sorted(glob.glob(f"{M}/runs/I3/I3_*")):
        m = PAT.match(os.path.basename(d))
        if m and glob.glob(os.path.join(d, "*.phdf")):
            runs.setdefault(m.group(1), {})[int(m.group(2))] = d
    res = {}
    for name, ladder in runs.items():
        Nref = max(ladder)
        if name == "BW":
            tr, xr, qr = load(ladder[Nref])
        rows = []
        for N in sorted(ladder):
            t, xc, q = load(ladder[N])
            if name == "BW":
                if N == Nref:
                    continue
                f = Nref // N                       # cell average of the reference onto the coarse cell
                qe = qr.reshape(8, N, f).mean(axis=2)
                ref = "N16384 run"
            else:
                qe, rs = exact(name, xc, t)
                ref = "exact"
            dx = 1.0 / N
            err = {v: float(np.abs(q[i] - qe[i]).sum() * dx) for i, v in enumerate(VARS)}
            rows.append({"N": N, "t": t, "ref": ref, **err})
        for i, r in enumerate(rows):
            for v in ("rho", "P", "By"):
                r[f"ord_{v}"] = (math.log2(rows[i - 1][v] / r[v]) if i and rows[i - 1]["N"] * 2 == r["N"] and r[v] > 0
                                 else None)
        res[name] = rows
        if name != "BW":
            rs = _rs[name]
            res[name + "_exact"] = {"z": [float(z) for z in rs.z], "selftest": {k: float(v) for k, v in rs.selftest().items()},
                                    "waves": [[str(w[0]), str(w[1]), str(w[2])] + [float(a) for a in w[3:]] for w in rs.waves()],
                                    "starL": rs.starL.prim(rs.Bx).tolist(), "starR": rs.starR.prim(rs.Bx).tolist()}
    return res


def text(res):
    L = ["I3 MHD shock tubes: L1 error (x in [0,1]) at the final time; ord = log2 ratio of successive N"]
    for name, rows in res.items():
        if name.endswith("_exact"):
            st = rows["selftest"]
            L.append(f"  {name}: solver residual {st['contact_res']:.1e}, RH {st['RH_rel']:.1e}, fan entropy "
                     f"{st['fan_entropy_rel']:.1e}; L* rho {rows['starL'][0]:.6f} R* rho {rows['starR'][0]:.6f} "
                     f"P* {rows['starL'][4]:.6f}")
            continue
        L.append(f"{name}  (reference: {rows[0]['ref']})")
        L.append(f"  {'N':>6} {'rho':>10} {'ord':>5} {'P':>10} {'ord':>5} {'By':>10} {'ord':>5} {'vx':>10} {'Bz':>10}")
        for r in rows:
            o = lambda v: f"{r['ord_' + v]:5.2f}" if r["ord_" + v] is not None else "    -"
            L.append(f"  {r['N']:6d} {r['rho']:10.3e} {o('rho')} {r['P']:10.3e} {o('P')} {r['By']:10.3e} {o('By')} "
                     f"{r['vx']:10.3e} {r['Bz']:10.3e}")
    return "\n".join(L)


def main():
    res = collect()
    js = f"{M}/results/tables/I3.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        ok = old.keys() == res.keys() and all(
            abs(a["rho"] - b["rho"]) <= 1e-12 * a["rho"] for k in old if not k.endswith("_exact")
            for a, b in zip(old[k], res[k]))
        print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    t = text(res)
    open(js.replace(".json", ".txt"), "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
