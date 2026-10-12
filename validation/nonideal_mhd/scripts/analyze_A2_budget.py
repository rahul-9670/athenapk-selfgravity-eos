#!/usr/bin/env python
"""A2 (energy budget of the RKL2 super-step) on the A7 Ohmic runs: one cycle of force-free Ohmic decay (|B| uniform,
no flow). Exact per-cell budget: d(e_int) = -d(ME) in EVERY cell (Joule heat eta J^2 = magnetic energy lost, J uniform
in magnitude), total E conserved (periodic). Reports per run: total-E change / E, max over cells of
|d(e_int) + d(ME)| / |d(ME)|_mean for e_int = E-KE-ME and for the dual energy, and the cell spread of the heat.
usage: analyze_A2_budget.py [--check]  (writes results/tables/A2_budget.json)"""
import sys, os, glob, re, json
import numpy as np, h5py
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def load(fn):
    c = np.array(h5py.File(fn, 'r')['cons'])[:, :, 0, 0, :]
    c = np.concatenate(list(c), axis=-1)  # [comp, i]
    rho = c[0]; ke = 0.5 * (c[1]**2 + c[2]**2 + c[3]**2) / rho; me = 0.5 * (c[5]**2 + c[6]**2 + c[7]**2)
    return c[4], ke, me, c[4] - ke - me, c[9]
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A7/A7_ohm_*')):
    tag = os.path.basename(d); r, m = map(int, re.match(r'A7_ohm_r(\d+)_m(\d+)', tag).groups())
    E0, ke0, me0, e0, d0 = load(f'{d}/parthenon.out0.00000.phdf'); E1, ke1, me1, e1, d1 = load(f'{d}/parthenon.out0.final.phdf')
    dme = me1 - me0; scale = np.abs(dme).mean()
    errE = np.abs((e1 - e0) + dme).max() / scale; errD = np.abs((d1 - d0) + dme).max() / scale
    dEtot = (E1.sum() - E0.sum()) / E0.sum()
    spread = (np.ptp(e1 - e0)) / scale
    ok = errE < 1e-6 and abs(dEtot) < 1e-13; bad += not ok
    rows.append(dict(tag=tag, ratio_cap=r, mode=m, dME_mean=float(dme.mean()), err_eint=float(errE), err_dual=float(errD),
                     dEtot_rel=float(dEtot), heat_spread=float(spread), ok=bool(ok)))
    print(f'{tag:18s} dME {dme.mean(): .3e}  |de+dME|/dME: E-path {errE:.2e} dual {errD:.2e}  dEtot/E {dEtot: .1e}  {"OK" if ok else "FAIL"}')
json.dump(rows, open(f'{B}/results/tables/A2_budget.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
