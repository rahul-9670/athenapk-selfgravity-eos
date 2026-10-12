#!/usr/bin/env python
"""A7 analysis (+ the 1-D A4 AD null: term ad). Per run: measured one-cycle amplification G of the mode (projection of By on sin kx and Bz on cos kx,
final / initial) vs the exact RKL2 answer G_th = R_s(z)^2 for the two Strang halves, R_s(z) = a_s + b_s P_s(1 + w1 z) (Legendre P_s)
(Meyer, Balsara & Aslam 2014, closed form -- independent of the code's three-term recurrence), z = -tau lam,
tau = dt/2, lam = eta_eff (4/dx^2) sin^2(k dx/2) (compact face-gradient Laplacian), s and dt read from run.log.
Also the time-exact answer of the same semi-discrete operator, exp(-lam dt). Writes results/tables/A7.json.
usage: analyze_A7.py [--check]   (--check: exit 1 unless |G - G_th| <= 1e-6 + 1e-6 G_th everywhere)"""
import sys, os, re, json, glob
import numpy as np, h5py
from numpy.polynomial import legendre as Lg
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
N, ETA = 64, 100.0
dx = 1.0 / N
def R_s(s, z):
    bs = (s * s + s - 2.0) / (2.0 * s * (s + 1.0)); w1 = 4.0 / (s * s + s - 2.0)
    return (1.0 - bs) + bs * Lg.legval(1.0 + w1 * z, [0] * s + [1])
def modeamp(fn, m):
    f = h5py.File(fn, 'r'); names = [n.decode() if isinstance(n, bytes) else n for n in f['Info'].attrs['ComponentNames']]
    p = np.array(f['prim'])[:, :, 0, 0, :]          # [block, comp, i]
    x = np.array(f['Locations']['x'])[:, :-1] + 0.5 * dx   # cell centres per block
    iy = [n for n in names if n.startswith('prim')].index('prim_magnetic_field_2') if 'prim_magnetic_field_2' in names else 6
    k = 2 * np.pi * m
    by, bz, xc = p[:, iy].ravel(), p[:, iy + 1].ravel(), x.ravel()
    return 0.5 * (2 * np.mean(by * np.sin(k * xc)) + 2 * np.mean(bz * np.cos(k * xc)))
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A7/A7_*')):
    tag = os.path.basename(d); term, r, m = re.match(r'A7_(\w+?)_r(\d+)_m(\d+)', tag).groups(); r, m = int(r), int(m)
    log = open(f'{d}/run.log', errors='replace').read()
    st = re.findall(r'STS ratio: (\S+) Taking (\d+) steps', log)
    cyc = re.findall(r'cycle=0 time=\S+ dt=(\S+)', log)
    f0, f1 = f'{d}/parthenon.out0.00000.phdf', f'{d}/parthenon.out0.final.phdf'
    if len(st) < 2 or not cyc or not os.path.exists(f1):
        print(f'{tag}: INCOMPLETE'); bad += 1; continue
    s = int(st[0][1]); ratio = float(st[0][0]); dt = float(cyc[0])
    lam = ETA * 4.0 / dx**2 * np.sin(np.pi * m * dx)**2
    z = -0.5 * dt * lam
    G = modeamp(f1, m) / modeamp(f0, m); Gth = R_s(s, z)**2; Gex = np.exp(-lam * dt)
    if term == 'ad':
        # The circularly polarized field is force-free (J = k B, J x B = 0): ambipolar diffusion must do NOTHING
        # (A4 null in 1D). Tolerance 1e-5 (truncation of the cell-centred J x B).
        Gth, Gex = 1.0, 1.0
        ok = abs(G - 1.0) <= 1e-5
    else:
        ok = abs(G - Gth) <= 1e-6 + 1e-6 * abs(Gth)
    bad += (not ok)
    rows.append(dict(tag=tag, term=term, ratio_cap=r, mode=m, ratio=ratio, s=s, dt=dt, z=z, z_over_bound=z / (-(s*s+s-2)/2.0),
                     G=G, G_th=Gth, G_exact=Gex, ok=bool(ok)))
    print(f'{tag:20s} s {s:3d} ratio {ratio:8.2f} z/zb {rows[-1]["z_over_bound"]:6.3f}  G {G: .6e}  R^2 {Gth: .6e}  exact {Gex:.3e}  {"OK" if ok else "FAIL"}')
os.makedirs(f'{B}/results/tables', exist_ok=True)
json.dump(rows, open(f'{B}/results/tables/A7.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
