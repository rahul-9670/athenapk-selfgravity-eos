#!/usr/bin/env python
"""A5 analysis: By(x, t_end) vs the exact two-medium (composite) solution at the two eta jumps (x = 0.25, 0.75).
Interface value B_i = (B_in sqrt(eta_in) + B_out sqrt(eta_out)) / (sqrt(eta_in) + sqrt(eta_out)); each side
B_i + (B_side - B_i) erf(d / (2 sqrt(eta_side t))) (Carslaw & Jaeger, two semi-infinite media; here "conductivity" =
"diffusivity" = eta, so the effusivity is sqrt(eta)). Errors: L1 (relative to B_in) over the box, and the error of the
two cells beside each interface. Orders from successive N. Writes results/tables/A5.json.
usage: analyze_A5.py [--check]   (--check: exit 1 unless every run completed)"""
import sys, os, glob, re, json
import numpy as np, h5py
from scipy.special import erf
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
C, BIN, BOUT = 100.0, 1.0e-3, 0.0
def exact(x, t, rho_in):
    ei, eo = C * rho_in, C
    Bi = (BIN * np.sqrt(ei) + BOUT * np.sqrt(eo)) / (np.sqrt(ei) + np.sqrt(eo))
    d = np.minimum(np.abs(x - 0.25), np.abs(x - 0.75)); inside = (x > 0.25) & (x < 0.75)
    return np.where(inside, Bi + (BIN - Bi) * erf(d / (2 * np.sqrt(ei * t))), Bi + (BOUT - Bi) * erf(d / (2 * np.sqrt(eo * t)))), Bi
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A5/A5_*')):
    tag = os.path.basename(d); c, sch, N = re.match(r'A5_c(\d+)_(\w+?)_N(\d+)', tag).groups(); c, N = int(c), int(N)
    fn = f'{d}/parthenon.out0.final.phdf'
    if not os.path.exists(fn): print(tag, 'INCOMPLETE'); bad += 1; continue
    f = h5py.File(fn, 'r'); t = float(f['Info'].attrs['Time'])
    by = np.concatenate(list(np.array(f['prim'])[:, 6, 0, 0, :])); x = (np.arange(N) + 0.5) / N
    ex, Bi = exact(x, t, 1.0 / c)
    l1 = np.abs(by - ex).mean() / BIN
    i0 = int(round(0.25 * N)); adj = np.abs(by[[i0 - 1, i0]] - ex[[i0 - 1, i0]]).max() / BIN
    log = open(f'{d}/run.log', errors='replace').read(); ncyc = len(re.findall(r'^cycle=', log, re.M))
    st = re.findall(r'STS ratio: (\S+) Taking (\d+) steps', log)
    rows.append(dict(tag=tag, contrast=c, scheme=sch, N=N, t=t, L1=float(l1), adj=float(adj), cycles=ncyc,
                     s_last=int(st[-1][1]) if st else 0, Bi_exact=float(Bi)))
for c in sorted({r['contrast'] for r in rows}):
    for sch in sorted({r['scheme'] for r in rows}):
        rs = sorted([r for r in rows if r['contrast'] == c and r['scheme'] == sch], key=lambda r: r['N'])
        for a, b in zip(rs, rs[1:]):
            b['order_L1'] = float(np.log2(a['L1'] / b['L1'])); b['order_adj'] = float(np.log2(a['adj'] / b['adj']))
        for r in rs:
            print(f"contrast {c:3d} {sch:9s} N {r['N']:5d}  L1 {r['L1']:.3e} (p {r.get('order_L1', float('nan')):5.2f})"
                  f"  interface cells {r['adj']:.3e} (p {r.get('order_adj', float('nan')):5.2f})  cycles {r['cycles']:6d} s {r['s_last']}")
os.makedirs(f'{B}/results/tables', exist_ok=True); json.dump(rows, open(f'{B}/results/tables/A5.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
