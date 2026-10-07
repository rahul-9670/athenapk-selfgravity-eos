#!/usr/bin/env python
"""E7g support: kink content of the interpolant along each rarefaction path. Along the exact isentrope from the
left state to rho*_L, sample ln c(ln rho) on a fine grid and split the variation of the slope s = dln c/dln rho
into JUMPS (at table grid-line crossings, |ds| over one fine step) and the smooth remainder. Also counts the
crossings of rho and e grid lines. -> results/tables/E7g_kinks.txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
HERE = os.path.dirname(os.path.abspath(__file__))
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json")))
t = E.load(); out = []
for prob in ("R1_molecular", "R2_dissociation", "R4_ionization"):
    for mode in ("p_over_rho", "p"):
        r = ex[f"{prob}|{mode}"]; eos = RX.TableEOS(t, mode)
        W = RX.WaveCurve(eos, *r["L"], -1); rs = r["Lstar"]["rho"]
        ise = W.isentrope(np.log(rs))
        lnr = np.linspace(np.log(r["L"][0]), np.log(rs), 20001)
        e = np.array([ise(x)[0] for x in lnr]); rho = np.exp(lnr)
        lc = 0.5 * np.log([RX.c2(eos, a, b, h=1e-9) for a, b in zip(rho, e)])
        dl = np.diff(lc)                             # change of ln c over each fine step
        ir = np.floor((np.log10(rho) - t["lr0"]) / t["dlr"]); ie = np.floor((np.log10(e) - t["le0"]) / t["dle"])
        cross = (np.diff(ir) != 0) | (np.diff(ie) != 0)
        typ = np.median(np.abs(dl[~cross]))          # smooth change per fine step
        jumps = np.abs(dl[cross]) - typ              # excess at a grid-line crossing = the jump of c
        jump = jumps.sum(); jmax = jumps.max(); smooth = np.abs(lc[-1] - lc[0])
        s = [np.nan, np.nan]
        out.append(f"{prob:16s} {mode:10s} fan: rho {r['L'][0]*E.RHO_U:.2e}->{rs*E.RHO_U:.2e} g/cc, T {r['L_T']:.0f}->{r['Lstar']['T']:.0f} K | "
                   f"grid lines crossed: rho {int(abs(ir[-1]-ir[0]))}, e {int(abs(ie[-1]-ie[0]))} | slope dlnc/dlnrho {s[0]:+.3f} -> {s[-1]:+.3f}"
                   f" | total |jumps| at grid lines {jump:.3f}, smooth variation {smooth:.3f}")
print("\n".join(out)); open(os.path.join(HERE, "..", "results", "tables", "E7g_kinks.txt"), "w").write("\n".join(out) + "\n")
