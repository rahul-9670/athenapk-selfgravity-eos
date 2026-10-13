#!/usr/bin/env python
"""I6c: the dual-energy module's tabulated-EOS planar Noh decks (D3, M 3/10/30, N 1024, full tlim) rerun with the
LIVE predictor (vl2_predictor_recon = same) and dual energy on / off / on + dual_energy_shock_eta1. Scored exactly as
dual_energy/scripts/analyze_D3.py: plateau = cells with 0.15 < |x - 0.5|/(S t) < 0.8, mean of rho/rho2 - 1 and
P/P2 - 1 against the exact general-EOS star state in D3_meta.json; shock = rho = (1 + rho2)/2 crossing vs 0.5 + S t.
  analyze_I6c.py [--check] -> results/tables/I6c.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _VT + "/dual_energy/scripts")
from analyze_D2 import cells  # noqa: E402

M = _VT + "/ideal_mhd"
META = json.load(open(f"{M}/inputs/I6/D3_meta.json"))["runs"]


def main():
    res = {}
    for d in sorted(glob.glob(f"{M}/runs/I6c/D3_*")):
        n = os.path.basename(d); key = "_".join(n.split("_")[:3])
        f = glob.glob(os.path.join(d, "*.out0.final.phdf"))
        if not f:
            continue
        m = META[key]
        t, c, _ = cells(f[0])
        xi = np.abs(c["x"] - 0.5) / (m["S"] * t)
        pl = (xi > 0.15) & (xi < 0.8)
        dx = 1.0 / m["N"]
        half = 0.5 * (1 + m["rho2"])
        right = c["x"] > 0.5
        xr, rr = c["x"][right], c["rho"][right]
        o = np.argsort(xr); xr, rr = xr[o], rr[o]
        cr = xr[np.nonzero((rr[:-1] - half) * (rr[1:] - half) <= 0)[0]]
        res[n] = {"M": m["M"], "rho_plateau": float((c["rho"][pl] / m["rho2"] - 1).mean()),
                  "P_plateau": float((c["P"][pl] / m["P2"] - 1).mean()),
                  "shock_cells": float((cr.max() - (0.5 + m["S"] * t)) / dx) if cr.size else None}
    js = f"{M}/results/tables/I6c.json"
    if "--check" in sys.argv:
        ok = json.load(open(js)) == json.loads(json.dumps(res)); print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    json.dump(res, open(js, "w"), indent=1)
    L = ["I6c tabulated-EOS planar Noh (D3 decks), live PLM predictor: plateau error vs the exact star state",
         f"{'run':<34} {'M':>3} {'rho plateau':>12} {'P plateau':>11} {'shock [cells]':>13}"]
    for k, r in res.items():
        L.append(f"{k:<34} {r['M']:3d} {r['rho_plateau']:+12.3e} {r['P_plateau']:+11.3e} {r['shock_cells']:+13.2f}")
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
