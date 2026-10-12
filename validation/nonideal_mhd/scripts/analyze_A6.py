#!/usr/bin/env python
"""A6 analysis: Bz(x, t) vs the Barenblatt-Pattle solution (m = 3) of dB/dt = d/dx(Q B^2 dB/dx):
Bz = B0 (T)^(-1/4) sqrt(1 - (x-xc)^2/(xf0^2 T^(1/2)))_+, T = (t0 + t)/t0, t0 = xf0^2/(4 Q B0^2). Reports L1 error
(relative to the exact peak), peak error, front position (outermost cell with Bz > 1e-3 B_peak, linear interpolation)
vs exact, flux conservation sum(Bz) dx. Orders from successive N. Writes results/tables/A6.json.
usage: analyze_A6.py [--check]"""
import sys, os, glob, re, json
import numpy as np, h5py
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
Q, B0, XF0, XC = 1.0e6, 1.0e-3, 0.1, 0.5
T0 = XF0**2 / (4 * Q * B0**2)
def exact(x, t):
    T = (T0 + t) / T0; q = 1 - (x - XC)**2 / (XF0**2 * np.sqrt(T))
    return B0 * T**-0.25 * np.sqrt(np.clip(q, 0, None)), XF0 * T**0.25, B0 * T**-0.25
def front(x, b):
    thr = 1e-3 * b.max(); i = np.where(b > thr)[0][-1]
    return x[i] + (b[i] - thr) / (b[i] - b[i + 1]) * (x[i + 1] - x[i]) - XC
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A6/A6_*')):
    tag = os.path.basename(d); arm, N = re.match(r'A6_(\w+?)_N(\d+)', tag).groups(); N = int(N)
    fn, f0n = f'{d}/parthenon.out0.final.phdf', f'{d}/parthenon.out0.00000.phdf'
    if not os.path.exists(fn): print(tag, 'INCOMPLETE'); bad += 1; continue
    f = h5py.File(fn, 'r'); t = float(f['Info'].attrs['Time'])
    bz = np.concatenate(list(np.array(f['prim'])[:, 7, 0, 0, :])); x = (np.arange(N) + 0.5) / N
    b0 = np.concatenate(list(np.array(h5py.File(f0n, 'r')['prim'])[:, 7, 0, 0, :]))
    ex, xf, bp = exact(x, t)
    log = open(f'{d}/run.log', errors='replace').read(); ncyc = len(re.findall(r'^cycle=', log, re.M))
    rows.append(dict(tag=tag, arm=arm, N=N, t=t, L1=float(np.abs(bz - ex).mean() / bp), peak_err=float(bz.max() / bp - 1),
                     front=float(front(x, bz)), front_exact=float(xf), front_err=float(front(x, bz) / xf - 1),
                     flux_drift=float(bz.sum() / b0.sum() - 1), cycles=ncyc))
for arm in sorted({r['arm'] for r in rows}):
    rs = sorted([r for r in rows if r['arm'] == arm], key=lambda r: r['N'])
    for a, b in zip(rs, rs[1:]): b['order_L1'] = float(np.log2(a['L1'] / b['L1']))
    for r in rs:
        print(f"{arm:18s} N {r['N']:5d}  L1 {r['L1']:.3e} (p {r.get('order_L1', float('nan')):5.2f})  peak {r['peak_err']:+.2e}"
              f"  front {r['front']:.4f} vs {r['front_exact']:.4f} ({r['front_err']:+.2e})  flux {r['flux_drift']:+.1e}  cycles {r['cycles']}")
os.makedirs(f'{B}/results/tables', exist_ok=True); json.dump(rows, open(f'{B}/results/tables/A6.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
