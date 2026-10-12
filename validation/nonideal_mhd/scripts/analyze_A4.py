#!/usr/bin/env python
"""A4 analysis (3-D ABC field). Ohmic: amplitude ratio G = <B_final . B_0>/<B_0 . B_0> vs exp(-eta k^2 t) (continuum)
and the residual shape error max|B_final - G B_0| / max|B_0|. AD: null -- max|B_final - B_0| / max|B_0| must -> 0.
Orders from N = 16, 32, 64. Writes results/tables/A4.json. usage: analyze_A4.py [--check]"""
import sys, os, glob, re, json
import numpy as np, h5py
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def Bf(fn):
    f = h5py.File(fn, 'r'); p = np.array(f['prim']); return p[:, 5:8], float(f['Info'].attrs['Time'])
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A4/A4_*')):
    tag = os.path.basename(d); term, integ, N = re.match(r'A4_(\w+?)_(\w+?)_N(\d+)', tag).groups(); N = int(N)
    if not os.path.exists(f'{d}/parthenon.out0.final.phdf'): print(tag, 'INCOMPLETE'); bad += 1; continue
    b0, _ = Bf(f'{d}/parthenon.out0.00000.phdf'); b1, t = Bf(f'{d}/parthenon.out0.final.phdf')
    G = float((b1 * b0).sum() / (b0 * b0).sum()); mx = np.abs(b0).max()
    log = open(f'{d}/run.log', errors='replace').read(); ncyc = len(re.findall(r'^cycle=', log, re.M))
    if term == 'ohm':
        Gex = float(np.exp(-1.0 * (2 * np.pi)**2 * t)); err = abs(G / Gex - 1); shape = float(np.abs(b1 - G * b0).max() / mx)
    else:
        Gex = 1.0; err = float(np.abs(b1 - b0).max() / mx); shape = abs(G - 1)
    rows.append(dict(tag=tag, term=term, integ=integ, N=N, t=t, G=G, G_exact=Gex, err=float(err), shape=shape, cycles=ncyc))
for key in sorted({(r['term'], r['integ']) for r in rows}):
    rs = sorted([r for r in rows if (r['term'], r['integ']) == key], key=lambda r: r['N'])
    for a, b in zip(rs, rs[1:]): b['order'] = float(np.log2(a['err'] / b['err']))
    for r in rs:
        lab = '|G/exact-1|' if r['term'] == 'ohm' else 'max|dB|/B'
        print(f"{r['term']:3s} {r['integ']:8s} N {r['N']:3d}  G {r['G']:.6f} (exact {r['G_exact']:.6f})  {lab} {r['err']:.3e} (p {r.get('order', float('nan')):5.2f})  shape {r['shape']:.2e}  cycles {r['cycles']}")
os.makedirs(f'{B}/results/tables', exist_ok=True); json.dump(rows, open(f'{B}/results/tables/A4.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
