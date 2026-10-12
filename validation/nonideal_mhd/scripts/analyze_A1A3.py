#!/usr/bin/env python
"""A1: Ohmic Gaussian (iprob 40) L1 error vs the PERIODIC-IMAGE-SUM solution (F-A14: a single Gaussian plateaus at
4.2e-4 on the [-4,4] box), By = amp/sqrt(4 pi eta T) sum_n exp(-(x - 8n)^2/(4 eta T)), T = t0 + t, eta = 0.25, t0 = 0.5.
A3: AD damped Alfven eigenmode (iprob 50): the pgen's own analytic-amplitude check (relative error) from run.log.
Writes results/tables/A1A3.json. usage: analyze_A1A3.py [--check]"""
import sys, os, glob, re, json
import numpy as np, h5py
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A1A3/A*_N*')):
    tag = os.path.basename(d); N = int(tag.split('_N')[1]); log = open(f'{d}/run.log', errors='replace').read()
    ncyc = len(re.findall(r'^cycle=', log, re.M))
    if tag.startswith('A1'):
        fs = sorted(glob.glob(f'{d}/parthenon.prim.final.phdf')) or sorted(glob.glob(f'{d}/parthenon.prim.*.phdf'))
        if not fs: print(tag, 'INCOMPLETE'); bad += 1; continue
        f = h5py.File(fs[-1], 'r'); t = float(f['Info'].attrs['Time']); p = np.array(f['prim']); x = np.array(f['Locations']['x'])
        by = p[:, 6].mean(axis=(1, 2)); xc = 0.5 * (x[:, 1:] + x[:, :-1])          # [block, i], y-averaged
        T = 0.5 + t; eta = 0.25
        ex = sum(1e-3 / np.sqrt(4 * np.pi * eta * T) * np.exp(-(xc - 8 * n)**2 / (4 * eta * T)) for n in range(-4, 5))
        err = float(np.abs(by - ex).sum() / np.abs(ex).sum())
    else:
        m = re.findall(r'relative error\s*=\s*(\S+)', log)
        if not m: print(tag, 'INCOMPLETE'); bad += 1; continue
        err = abs(float(m[-1]))
    rows.append(dict(tag=tag, test=tag.rsplit('_', 2)[0], integ=tag.split('_')[-2], N=N, err=err, cycles=ncyc))
for key in sorted({(r['test'], r['integ']) for r in rows}):
    rs = sorted([r for r in rows if (r['test'], r['integ']) == key], key=lambda r: r['N'])
    for a, b in zip(rs, rs[1:]): b['order'] = float(np.log2(a['err'] / b['err']))
    for r in rs: print(f"{r['test']:12s} {r['integ']:8s} N {r['N']:4d}  err {r['err']:.3e} (p {r.get('order', float('nan')):5.2f})  log lines {r['cycles']}")
os.makedirs(f'{B}/results/tables', exist_ok=True); json.dump(rows, open(f'{B}/results/tables/A1A3.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
