#!/usr/bin/env python
"""E6 acoustic waves: after one period the exact solution is the initial state. Reports the L1 density error
(in units of the amplitude) and the measured phase speed (from the phase of the k = 1 Fourier mode)
against the table's adiabatic sound speed. Convergence order from N = 16..256."""
import glob, json, os, sys
import h5py
import numpy as np
R = os.path.join(os.path.dirname(__file__), "..", "runs", "E6_wave")
M = json.load(open(os.path.join(os.path.dirname(__file__), "..", "inputs", "eos_states.json")))


def rho1d(f):
    with h5py.File(f) as h:
        p = h["prim"][()]                       # [block, comp, k, j, i]
        x = h["Locations/x"][()] if "Locations/x" in h else None
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
    return t, p[0, 0, 0, 0, :].astype(float), p[0, 1, 0, 0, :].astype(float)


out = {}
for s, m in M.items():
    rows = []
    for d in sorted(glob.glob(os.path.join(R, f"E6_wave_{s}_n*")), key=lambda q: (len(q), q)):
        fs = sorted(glob.glob(os.path.join(d, "*.out0.00000.phdf"))) + glob.glob(os.path.join(d, "*.out0.final.phdf"))
        if len(fs) < 2:
            continue
        t0, r0, _ = rho1d(fs[0])
        t1, r1, _ = rho1d(fs[-1])
        n = r0.size
        amp = 1e-6 * m["rho_code"]
        x = (np.arange(n) + 0.5) / n
        d0, d1 = r0 - m["rho_code"], r1 - m["rho_code"]
        L1 = float(np.mean(np.abs(r1 - r0)) / amp)
        ph = lambda dd: np.angle(np.sum(dd * np.exp(-2j * np.pi * x)))
        dphi = (ph(d1) - ph(d0) + np.pi) % (2 * np.pi) - np.pi     # residual phase after one period
        c_meas = m["cs_code"] * (1 - dphi / (2 * np.pi))             # +dphi lag = slower wave
        a_ratio = float(np.abs(np.sum(d1 * np.exp(-2j * np.pi * x))) / np.abs(np.sum(d0 * np.exp(-2j * np.pi * x))))
        rows.append(dict(run=os.path.basename(d), n=n, t=t1, L1_over_amp=L1, amp_ratio=a_ratio,
                         c_rel_err=float(c_meas / m["cs_code"] - 1)))
    out[s] = rows
    conv = [r for r in rows if "bis" not in r["run"]]
    print(f"{s} rho {m['rho_cgs']:g} g/cc T {m['T']:g} K  Gamma1 {m['gamma1']:.4f}  cs {m['cs_code']:.5g}")
    for i, r in enumerate(rows):
        order = ""
        if "bis" not in r["run"] and i > 0 and "bis" not in rows[i - 1]["run"]:
            order = f"  order {np.log2(rows[i - 1]['L1_over_amp'] / r['L1_over_amp']):.2f}"
        print(f"   {r['run']:24s} L1/amp {r['L1_over_amp']:.3e}  amplitude kept {r['amp_ratio']:.5f}  "
              f"phase-speed error {r['c_rel_err']:+.2e}{order}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "tables", "E6_waves.json"), "w"), indent=1)
