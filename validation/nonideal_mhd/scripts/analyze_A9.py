#!/usr/bin/env python
"""A9 analysis: Ohmic Gaussian across a static coarse-fine boundary (at the peak, x = 0, and at x = +-4). Volume-weighted
L1 error vs the periodic image sum (as A1), overall and on the coarse / fine halves, compared with A1 on uniform N and 2N
(results/tables/A1A3.json). Also the error in the two cells beside x = 0. Writes results/tables/A9.json.
usage: analyze_A9.py [--check]"""
import sys, os, glob, re, json
import numpy as np, h5py
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
a1 = {(r['integ'], r['N']): r['err'] for r in json.load(open(f'{B}/results/tables/A1A3.json')) if r['test'] == 'A1'}
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A9/A9_*')):
    tag = os.path.basename(d); integ, N = re.match(r'A9_(\w+?)_N(\d+)', tag).groups(); N = int(N)
    fs = glob.glob(f'{d}/parthenon.prim.final.phdf')
    if not fs: print(tag, 'INCOMPLETE'); bad += 1; continue
    f = h5py.File(fs[0], 'r'); t = float(f['Info'].attrs['Time']); p = np.array(f['prim']); x = np.array(f['Locations']['x'])
    by = p[:, 6].mean(axis=(1, 2)); xc = 0.5 * (x[:, 1:] + x[:, :-1]); dx = np.diff(x, axis=1)
    T = 0.5 + t; eta = 0.25
    ex = sum(1e-3 / np.sqrt(4 * np.pi * eta * T) * np.exp(-(xc - 8 * n)**2 / (4 * eta * T)) for n in range(-4, 5))
    e = np.abs(by - ex) * dx; w = np.abs(ex) * dx
    fine = xc > 0; L1 = e.sum() / w.sum(); L1f = e[fine].sum() / w[fine].sum(); L1c = e[~fine].sum() / w[~fine].sum()
    xf, bf, ef = xc.ravel(), by.ravel(), ex.ravel(); o = np.argsort(xf); xf, bf, ef = xf[o], bf[o], ef[o]
    i0 = np.searchsorted(xf, 0.0); adj = float(np.abs(bf[i0 - 1:i0 + 1] - ef[i0 - 1:i0 + 1]).max() / ef.max())
    rows.append(dict(tag=tag, integ=integ, N=N, L1=float(L1), L1_fine=float(L1f), L1_coarse=float(L1c), adj=adj,
                     A1_N=a1.get((integ, N)), A1_2N=a1.get((integ, 2 * N))))
for integ in ('unsplit', 'rkl2'):
    rs = sorted([r for r in rows if r['integ'] == integ], key=lambda r: r['N'])
    for a, b in zip(rs, rs[1:]): b['order'] = float(np.log2(a['L1'] / b['L1']))
    for r in rs:
        print(f"{integ:8s} N {r['N']:4d}  L1 {r['L1']:.3e} (p {r.get('order', float('nan')):5.2f})  coarse {r['L1_coarse']:.3e} fine {r['L1_fine']:.3e}"
              f"  [A1 uniform N {r['A1_N'] or float('nan'):.3e}, 2N {r['A1_2N'] or float('nan'):.3e}]  C/F cells {r['adj']:.2e}")
os.makedirs(f'{B}/results/tables', exist_ok=True); json.dump(rows, open(f'{B}/results/tables/A9.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
