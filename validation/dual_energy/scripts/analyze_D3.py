"""D3 analysis (planar Noh / colliding cold streams). Final dump, per run:
  plateau  = cells with 0.15 < |x - 0.5|/(S t) < 0.8 (excludes the Noh wall-heating cells at the centre and the shock
             fronts): mean and max of rho/rho2 - 1, P/P2 - 1 (rho2, P2 from the exact general-EOS solver)
  pre      = cells with |x - 0.5|/(S t) > 1.2: max|P/P0 - 1| (the cold inflow must stay at T0; rho = 1, v = u exactly)
  shock    = position of the rho = (1 + rho2)/2 crossing vs 0.5 + S t (in cells)
  wall     = min over the 4 centre cells of P/(rho^Gamma) relative to the plateau (wall-heating entropy excess)
  switched = number of dual_switched cells inside the shocked region (should be 0: shocked gas is thermally dominant)
Decomposition: the final phdf of N1024 prod vs mb32 / mb32_np8 / mb32_np8_noghost compared BITWISE.
-> results/tables/D3_noh.{txt,json}"""
import glob
import json
import os

import h5py
import numpy as np

from analyze_D2 import cells
from analyze_D1 import hst, hcount
from de_common import INP, RUNS, TAB, table_eos

eos = table_eos()
meta = json.load(open(os.path.join(INP, "D3_meta.json")))["runs"]
P0 = None
rows, res = [], {}


def final(d):
    f = glob.glob(os.path.join(d, "*.out0.final.phdf"))
    return f[0] if f else None


def bitwise(fa, fb):
    with h5py.File(fa) as a, h5py.File(fb) as b:
        for k in ("prim", "cons"):
            x, y = a[k][()], b[k][()]
            # block order may differ with the decomposition: compare as sorted cell arrays by position
            xa = a["Locations"]["x"][()]; xb = b["Locations"]["x"][()]
            ca = 0.5 * (xa[:, 1:] + xa[:, :-1]).ravel(); cb = 0.5 * (xb[:, 1:] + xb[:, :-1]).ravel()
            va = np.moveaxis(x, 1, -1).reshape(-1, x.shape[1])[np.argsort(ca)]
            vb = np.moveaxis(y, 1, -1).reshape(-1, y.shape[1])[np.argsort(cb)]
            if va.shape != vb.shape or not np.array_equal(va.view(np.uint8), vb.view(np.uint8)):
                return f"DIFFER({k}: max rel {np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1e-300)):.1e})"
    return "BITWISE"


for name, m in meta.items():
    P0 = float(eos.P(1.0, m["e0"]))
    variants = ["prod", "off", "eta2off", "noexempt"] + (["mb32", "mb32_np8", "mb32_np8_noghost"] if m["N"] == 1024 else [])
    for v in variants:
        d = os.path.join(RUNS, "D3", f"{name}_{v}")
        f = final(d)
        if not f:
            rows.append(f"{name:14s} {v:18s} MISSING"); continue
        t, c, _ = cells(f)
        o = np.argsort(c["x"]); x = c["x"][o]; rho = c["rho"][o]; P = c["P"][o]; sw = c["sw"][o]
        r = np.abs(x - 0.5) / (m["S"] * t)
        pl = (r > 0.15) & (r < 0.8); pre = r > 1.2; shocked = r < 0.9
        right = x > 0.5
        xs = x[right][np.argmax(rho[right] < 0.5 * (1 + m["rho2"]))]
        G = np.log(m["P2"] / P0) / np.log(m["rho2"])   # effective adiabat exponent for the entropy proxy
        ent = P / rho ** G
        ctr = np.argsort(np.abs(x - 0.5))[:4]
        rec = dict(t=t, rho_mean=float((rho[pl] / m["rho2"] - 1).mean()), rho_max=float(np.abs(rho[pl] / m["rho2"] - 1).max()),
                   P_mean=float((P[pl] / m["P2"] - 1).mean()), P_max=float(np.abs(P[pl] / m["P2"] - 1).max()),
                   pre=float(np.abs(P[pre] / P0 - 1).max()), shock_cells=float((xs - (0.5 + m["S"] * t)) * m["N"]),
                   wall=float(ent[ctr].min() / np.median(ent[pl]) - 1), nsw_shocked=int((sw[shocked] > 0.5).sum()))
        h = hst(os.path.join(RUNS, "D3", f"{name}_{v}"))
        rec["nTfloor"] = hcount(h, "cons-nTfloor"); rec["npfloor"] = hcount(h, "cons-npfloor")
        if v.startswith("mb32"):
            ref = final(os.path.join(RUNS, "D3", f"{name}_prod"))
            rec["vs_1block"] = bitwise(ref, f) if ref else "no ref"
        res[f"{name}_{v}"] = rec
        extra = f" | vs 1 block: {rec['vs_1block']}" if "vs_1block" in rec else ""
        rows.append(f"{name:14s} {v:18s} plateau rho {rec['rho_mean']:+.2e} (max {rec['rho_max']:.1e}) P {rec['P_mean']:+.2e} (max {rec['P_max']:.1e})"
                    f" | pre-shock max|P/P0-1| {rec['pre']:.1e} | shock {rec['shock_cells']:+.1f} cells | wall {rec['wall']:+.2f}"
                    f" | switched in shocked {rec['nsw_shocked']} | nTfloor {rec['nTfloor']:.0f}{extra}")
txt = "\n".join(rows)
print(txt)
open(os.path.join(TAB, "D3_noh.txt"), "w").write(
    "# D3 planar Noh (colliding cold streams, T0 10 K, u = +-M cs0), exact star state from riemann_exact.py on the runtime interpolant\n"
    + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D3_noh.json"), "w"), indent=1)
