"""D1x analysis (DE-F3 isolation): per run max|P/P0 - 1| at each dump (one per box crossing) and the L1 density error
vs the exact translated profile (= IC after whole crossings). -> results/tables/D1x_contact.{txt,json}"""
import glob
import json
import os
import numpy as np
from analyze_D1 import load, hst, hcount
from de_common import INP, RUNS, TAB

meta = json.load(open(os.path.join(INP, "D1x_meta.json")))
rows, res = [], {}
for name, m in meta["runs"].items():
    P0 = m.get("P0", meta["P0"])
    for tag in ("on", "off"):
        d = os.path.join(RUNS, "D1x", f"{name}_{tag}")
        fs = sorted(glob.glob(os.path.join(d, "*.out0.0*.phdf"))) + glob.glob(os.path.join(d, "*.out0.final.phdf"))
        if not glob.glob(os.path.join(d, "*.out0.final.phdf")):
            rows.append(f"{name:18s} {tag:3s} MISSING"); continue
        _, X, r0, _, _ = load(fs[0])
        per = []
        for f in fs[1:]:
            t, X1, r1, v1, P1 = load(f)
            per.append(dict(t=t, maxdP=float(np.abs(P1 / P0 - 1).max()), L1rho=float(np.abs(r1 - r0).mean())))
        c = hst(d)
        res[f"{name}_{tag}"] = dict(meta=m, per=per, ndual=hcount(c, "cons-ndual"), nTfloor=hcount(c, "cons-nTfloor"))
        seq = " ".join(f"{p['maxdP']:.2e}" for p in per)
        rows.append(f"{name:18s} {tag:3s} max|P/P0-1| per dump: {seq} | L1(rho) {per[-1]['L1rho']:.3e} | "
                    f"ndual {res[f'{name}_{tag}']['ndual']:.0f} nTfloor {res[f'{name}_{tag}']['nTfloor']:.0f}")
txt = "\n".join(rows)
print(txt)
open(os.path.join(TAB, "D1x_contact.txt"), "w").write(
    "# D1x: isolation of the dual-energy contact pressure error (DE-F3); base D1a (rho 1|2, N 256), one change per run\n" + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D1x_contact.json"), "w"), indent=1)
