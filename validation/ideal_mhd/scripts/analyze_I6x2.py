#!/usr/bin/env python
"""I6 / X2 follow-up: the pre-shock Tfloor undershoot of the planar tabulated-EOS Noh problem (dual_energy X2).

X2 (dual_energy/results/tables/X2_noh_tfloor.txt): Noh M 10, N 1024, dual energy OFF, vl2_predictor_recon = dc (the
deck value then): the cell just ahead of the shock undershot to P/P0 = 0.273 = Tfloor/T0 in 9 of 25 snapshots.
Here the same decks (inputs/I6/D3_M{10,30}_N1024.in, md5-identical copies) x predictor {dc, same} x dual {off, on},
plus M 10 at N 4096. Per run: Tfloor hits from the .hst (cons-nTfloor) and, per snapshot, the minimum P/P0 among
pre-shock cells (r/r_s > 1, r = |x - 0.5|, r_s = S t from the exact star state in D3_meta.json).
Uses the dual-energy module's readers (analyze_D2.cells, analyze_D1.hst) unchanged.
  analyze_I6x2.py [--check]  -> results/tables/I6x2.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import numpy as np

sys.path.insert(0, _VT + "/dual_energy/scripts")
from analyze_D2 import cells  # noqa: E402
from analyze_D1 import hst  # noqa: E402
from de_common import table_eos  # noqa: E402

M = _VT + "/ideal_mhd"
META = json.load(open(f"{M}/inputs/I6/D3_meta.json"))["runs"]
eos = table_eos()


def one(d):
    name = os.path.basename(d)
    mach = int(name.split("_")[1][1:])
    m = META[f"D3_M{mach}_N1024"]
    P0 = float(eos.P(1.0, m["e0"]))
    h = hst(d)
    nT = h["cons-nTfloor"]
    snaps = sorted(glob.glob(os.path.join(d, "*.out0.0*.phdf")))[1:]
    rows, ev = [], 0
    for f in snaps:
        t, c, _ = cells(f)
        r = np.abs(c["x"] - 0.5) / (m["S"] * t)
        q = c["P"] / P0
        qq = np.where(r > 1.0, q, 9.0)
        i = int(np.argmin(qq))
        ev += bool(q[i] < 0.99)
        rows.append({"t": round(float(t), 6), "minP_P0": float(q[i]), "r_rs": float(r[i])})
    return {"tfloor_hits": float(nT.sum()), "rows_with_hits": int((nT > 0).sum()), "hst_rows": int(nT.size),
            "undershoot_snaps": ev, "snaps": len(rows), "min_over_snaps": min(x["minP_P0"] for x in rows) if rows else None,
            "rows": rows}


def main():
    res = {os.path.basename(d): one(d) for d in sorted(glob.glob(f"{M}/runs/I6/X2_*")) if glob.glob(f"{d}/*.phdf")}
    js = f"{M}/results/tables/I6x2.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        ok = old.keys() == res.keys() and all(
            abs(a["minP_P0"] - b["minP_P0"]) <= 1e-12 for k in old for a, b in zip(old[k]["rows"], res[k]["rows"]))
        print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    L = ["I6/X2: planar tabulated-EOS Noh, t <= 0.012. Tfloor/T0 = 0.2726. undershoot = snapshots with min P/P0 < 0.99 "
         "among pre-shock cells",
         f"{'run':<30} {'Tfloor hits':>11} {'hst rows':>9} {'undershoot':>11} {'min P/P0':>9}"]
    for k, r in res.items():
        L.append(f"{k:<30} {r['tfloor_hits']:11.0f} {r['rows_with_hits']:4d}/{r['hst_rows']:<4d} "
                 f"{r['undershoot_snaps']:4d}/{r['snaps']:<6d} {r['min_over_snaps']:9.3e}")
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
