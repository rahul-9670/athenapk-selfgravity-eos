"""D4 analysis: centre cell (v = 0) of the homologous low-beta compression vs the table adiabat through (rho0, e0)
(de/dln rho = P/rho on the runtime interpolant, RK4, E8's integrator) and flux freezing B_y/rho. Per dump: compression,
e/e_ad - 1 (e from P by exact inversion of the interpolant), (B_y/rho)/(B0/rho0) - 1, beta, switched flag.
-> results/tables/D4_compress.{txt,json}"""
import glob, json, os, sys
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root
import numpy as np
from analyze_D2 import cells
from analyze_D1 import hst, hcount
from de_common import INP, RUNS, TAB, table_eos
sys.path.insert(0, _REPO + "/validation/eos_tabulated/scripts")
from analyze_E8 import adiabat
eos = table_eos()
meta = json.load(open(os.path.join(INP, "D4_meta.json")))["runs"]
rows, res = [], {}
for name, m in meta.items():
    Lr, Ee = adiabat(eos, m["rho0"], m["e0"], m["rho0"] * 101.5)
    for v in ("on", "off"):
        d = os.path.join(RUNS, "D4", f"{name}_{v}")
        fs = sorted(glob.glob(d + "/*.out0.0*.phdf")) + glob.glob(d + "/*.out0.final.phdf")
        if not glob.glob(d + "/*.final.phdf"):
            rows.append(f"{name:10s} {v:3s} MISSING"); continue
        per = []
        for f in fs[1:]:
            t, c, _ = cells(f)
            i = int(np.argmin(np.abs(c["x"] - 0.5)))
            rho, P = c["rho"][i], c["P"][i]
            e = eos.e_of_P(rho, P); ead = float(np.interp(np.log(rho), Lr, Ee))
            per.append(dict(t=t, comp=rho / m["rho0"], de=e / ead - 1, ff=(c["By"][i] / rho) / (m["B0"] / m["rho0"]) - 1,
                            beta=2 * P / c["By"][i] ** 2, sw=float(c["sw"][i]), unif=float(np.ptp(c["rho"]) / rho)))
        h = hst(d)
        res[f"{name}_{v}"] = dict(meta=m, per=per, nTfloor=hcount(h, "cons-nTfloor"), npfloor=hcount(h, "cons-npfloor"))
        mx = max(abs(p["de"]) for p in per); f_ = per[-1]
        rows.append(f"{name:10s} {v:3s} x{f_['comp']:.1f}: e/e_ad-1 {f_['de']:+.3e} (max over leg {mx:.3e}) | B_y/rho {f_['ff']:+.1e}"
                    f" | beta {per[0]['beta']:.1e} -> {f_['beta']:.1e} | centre switched {f_['sw']:.0f} | uniformity {f_['unif']:.1e}"
                    f" | nTfloor {res[f'{name}_{v}']['nTfloor']:.0f} npfloor {res[f'{name}_{v}']['npfloor']:.0f}")
txt = "\n".join(rows); print(txt)
open(os.path.join(TAB, "D4_compress.txt"), "w").write(
    "# D4 homologous compression x101 with transverse B (centre cell vs the table adiabat; eos_interp p)\n" + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D4_compress.json"), "w"), indent=1)
