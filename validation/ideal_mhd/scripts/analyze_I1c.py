#!/usr/bin/env python
"""I1c: tabulated-EOS linear waves (acoustic, perpendicular fast at beta ~1.2 and ~0.025) in the production key set.
Error = L1 over [0,1] of (rho, vx, By) after one period vs the t = 0 dump, normalised by amp = 1e-6 times the
eigenvector amplitude of that variable (rho0, cf, B0y) -> a relative wave error; order = log2 ratio.
  analyze_I1c.py [--check] -> results/tables/I1c.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, math, os, re, sys
import h5py
import numpy as np

M = _VT + "/ideal_mhd"
PAT = re.compile(r"I1c_(\w+?)_N(\d+)(_pr)?$")


def prim(f):
    with h5py.File(f, "r") as h:
        p = h["prim"][()]
    return np.concatenate([p[b, :8, 0, 0, :] for b in range(p.shape[0])], axis=1)


def main():
    lad = {}
    for d in sorted(glob.glob(f"{M}/runs/I1c/I1c_*")):
        m = PAT.match(os.path.basename(d))
        fs = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))
        if not m or len(fs) < 2:
            continue
        a, b = prim(fs[0]), prim(fs[-1])
        N = a.shape[1]
        sc = {"rho": abs(a[0]).mean() * 1e-6, "vx": max(abs(a[1]).max(), 1e-30), "By": max(abs(a[6] - a[6].mean()).max(), 1e-30)}
        e = {"rho": float(np.abs(b[0] - a[0]).mean() / sc["rho"]),
             "vx": float(np.abs(b[1] - a[1]).mean() / sc["vx"]),
             "By": float(np.abs(b[6] - a[6]).mean() / sc["By"]) if a[6].std() > 0 else None}
        lad.setdefault(m.group(1) + (m.group(3) or ""), {})[int(m.group(2))] = e
    res = {}
    for k, l in lad.items():
        Ns = sorted(l); rows = []
        for i, N in enumerate(Ns):
            r = {"N": N, **l[N]}
            if i and Ns[i - 1] * 2 == N:
                r["ord_rho"] = math.log2(l[Ns[i - 1]]["rho"] / l[N]["rho"])
            rows.append(r)
        res[k] = rows
    js = f"{M}/results/tables/I1c.json"
    if "--check" in sys.argv:
        ok = json.load(open(js)) == json.loads(json.dumps(res)); print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    json.dump(res, open(js, "w"), indent=1)
    L = ["I1c tabulated-EOS waves (production keys), L1 error after one period relative to the wave amplitude"]
    for k, rows in res.items():
        for r in rows:
            L.append(f"  {k:<10} N {r['N']:4d}  rho {r['rho']:.3e} " + (f"({r['ord_rho']:.2f})" if 'ord_rho' in r else "(-)   ")
                     + f"  vx {r['vx']:.3e}" + (f"  By {r['By']:.3e}" if r["By"] is not None else ""))
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
