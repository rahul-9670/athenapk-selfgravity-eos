"""D2x: mechanism of the strong div B blob errors at beta 1e-4 (DE-F7). D2b beta 1e-4, bamp = B0z, |v| = 10 cs0 at
N 64^2 (4 ranks), variants prod / noexempt (shock_comp 1e30) / eta2off / off. Per dump: max|dP| vs the adiabat, cells
with |dP| > 1e-2 split into switched / E-branch and compressing (-div v > 0) / expanding, the max Mach number of the
velocity relative to the bulk flow (|v - v0|/cs), and the spurious heat over the blob's excess magnetic energy.
Hypothesis: the extended-GLM momentum source -B div B drives supersonic motions; shocked cells are exempted from the
two-sided test and take E - KE - ME, whose error in beta 1e-4 gas is amplified ~1/beta.
-> results/tables/D2x_divb_mechanism.{txt,json}"""
import glob
import json
import os

import h5py
import numpy as np

from analyze_D2 import Pad, e_ad, eos_e, meta
from analyze_D1 import hst
from de_common import RUNS, TAB

m = meta["runs"]["D2b_b0.0001_br1_M10_N128"]
cs0 = json.load(open(os.path.join(os.path.dirname(RUNS), "inputs", "D1_meta.json")))["cs0"]
rows, res = [], {}
for v in ("prod", "noexempt", "eta2off", "off"):
    d = os.path.join(RUNS, "D2x", f"D2b_b1e-4_br1_M10_N64_{v}")
    fs = sorted(glob.glob(d + "/*.out0.0*.phdf")) + glob.glob(d + "/*.out0.final.phdf")
    if len(fs) < 2:
        rows.append(f"{v:9s} MISSING"); continue
    per = []
    EB = None
    for f in fs:
        with h5py.File(f) as h:
            p = h["prim"][()].astype(float); sw = h["dual_switched"][()].astype(float).reshape(p.shape[0], -1) if "dual_switched" in h else np.zeros((p.shape[0], p[:, 0].size // p.shape[0]))
            t = float(np.atleast_1d(h["Info"].attrs["Time"])[0]); x = h["Locations"]["x"][()]; y = h["Locations"]["y"][()]
        dx = x[0, 1] - x[0, 0]; dy = y[0, 1] - y[0, 0]
        rho = p[:, 0, 0]; vx = p[:, 1, 0]; vy = p[:, 2, 0]; P = p[:, 4, 0]
        B2 = (p[:, 5, 0] ** 2 + p[:, 6, 0] ** 2 + p[:, 7, 0] ** 2)
        if EB is None:
            EB = float((0.5 * B2 - 0.5 * m["B0z"] ** 2).sum() * dx * dy); continue
        divv = np.zeros_like(rho)   # interior of each block (edges by one-sided differences: np.gradient)
        for b in range(rho.shape[0]):
            divv[b] = np.gradient(vx[b], dx, axis=1) + np.gradient(vy[b], dy, axis=0)
        r, Pf, s, dv = rho.ravel(), P.ravel(), sw.ravel() > 0.5, divv.ravel()
        dP = Pf / Pad(r) - 1
        bad = np.abs(dP) > 1e-2
        mach = float(np.sqrt((vx - m["u0"]) ** 2 + (vy - m["u0"]) ** 2).max() / cs0)
        heat = float((r * (eos_e(r, Pf) - e_ad(r))).sum() * dx * dy) if v != "off" else float("nan")
        per.append(dict(t=t / m["tlim"], max=float(np.abs(dP).max()), nbad=int(bad.sum()),
                        bad_sw=int((bad & s).sum()), bad_E=int((bad & ~s).sum()),
                        bad_E_comp=int((bad & ~s & (dv < 0)).sum()), bad_sw_comp=int((bad & s & (dv < 0)).sum()),
                        mach=mach, heat=heat / EB))
    h = hst(d)
    res[v] = dict(per=per, nTfloor=float(h.get("cons-nTfloor", np.zeros(1)).sum()), ndual_mean=float(np.mean(h.get("cons-ndual", [np.nan]))))
    for q in per:
        rows.append(f"{v:9s} t/tlim {q['t']:.2f}: max|dP| {q['max']:.2e} | cells>1e-2 {q['nbad']:5d} (switched {q['bad_sw']}, "
                    f"of them compressing {q['bad_sw_comp']}; E-branch {q['bad_E']}, compressing {q['bad_E_comp']}) | "
                    f"max |v - v0|/cs0 {q['mach']:.1f} | heat/E_B,exc {q['heat']:+.1e}")
    rows.append(f"{v:9s} totals: Tfloor hits {res[v]['nTfloor']:.0f}, mean switched cells {res[v]['ndual_mean']:.0f} of 4096")
txt = "\n".join(rows)
print(txt)
open(os.path.join(TAB, "D2x_divb_mechanism.txt"), "w").write(
    "# D2x: strong div B blob (beta 1e-4, bamp = B0z, |v| = 10 cs0), N 64^2; production vs shock exemption off vs eta2 off vs dual off\n" + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D2x_divb_mechanism.json"), "w"), indent=1)
