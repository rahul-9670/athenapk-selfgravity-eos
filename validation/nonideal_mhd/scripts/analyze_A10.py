#!/usr/bin/env python
"""A10 analysis: simulated C-shock vs the steady profile of cshock_ref.py. Lab frame -> shock frame: the downstream gas is
at rest, so the profile is compared in By(x) aligned at the By midpoint (B0+B1)/2 (shock position from the run, linear
interpolation). Errors: L1 of By over the window [x_mid - 3W, x_mid + 3W] (W = exact 10-90 width) relative to (B1 - B0);
downstream plateau (rho, By, P) vs the jump conditions; shock position x_mid vs L - V t + const (reported only).
usage: analyze_A10.py [--check]  (writes results/tables/A10.json)"""
import sys, os, glob, re, json, importlib.util
import numpy as np, h5py
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, '..')
spec = importlib.util.spec_from_file_location('cr', os.path.join(H, 'cshock_ref.py')); cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
rho0, P0, B0, u, Q, g = 1.0, 0.1, 2.0, 2.0, 0.01, 5.0 / 3.0
ref = cr.profile(rho0, P0, B0, u, Q, g); B1 = ref['B1']; Bm = 0.5 * (B0 + B1)
xr = ref['x'] - np.interp(Bm, ref['B'], ref['x'])          # exact, aligned at the midpoint (B monotone)
i10, i90 = [np.argmin(np.abs(ref['B'] - (B0 + f * (B1 - B0)))) for f in (0.1, 0.9)]; W = xr[i90] - xr[i10]
rho1 = rho0 * ref['r']; P1 = ref['P'][-1]
rows, bad = [], 0
for d in sorted(glob.glob(f'{B}/runs/A10/A10_*') + glob.glob(f'{B}/runs/A10/A10L_*')):
    tag = os.path.basename(d); lg, integ, N, nd = re.match(r'A10(L?)_(\w+?)_N(\d+)(_nd)?$', tag).groups(); N = int(N); integ += ('_nodual' if nd else '') + ('_long' if lg else '')
    Lx = 2.0 if lg else 1.0; NX = 2 * N if lg else N
    fn = f'{d}/parthenon.out0.final.phdf'
    if not os.path.exists(fn): print(tag, 'INCOMPLETE'); bad += 1; continue
    f = h5py.File(fn, 'r'); t = float(f['Info'].attrs['Time']); p = np.concatenate(list(np.array(f['prim'])[:, :, 0, 0, :]), axis=-1)
    x = (np.arange(NX) + 0.5) * Lx / NX; by = p[6]; rho = p[0]; P = p[4]
    # shock = the crossing of Bm (field rises from B0 upstream (left) to B1 downstream (right))
    k = np.where((by[:-1] < Bm) & (by[1:] >= Bm))[0][-1]; xm = x[k] + (Bm - by[k]) / (by[k + 1] - by[k]) * (x[k + 1] - x[k])
    win = np.abs(x - xm) < 3 * W
    ex = np.interp(x[win] - xm, xr, ref['B'], left=B0, right=B1)
    L1 = float(np.abs(by[win] - ex).mean() / (B1 - B0)); Linf = float(np.abs(by[win] - ex).max() / (B1 - B0))
    dwn = (x > xm + 2 * W) & (x < xm + 3 * W)   # just behind the shock: gas shocked in the steady phase (the region near the wall carries the start-up anomaly)
    rows.append(dict(tag=tag, integ=integ, N=N, t=t, x_mid=float(xm), L1=L1, Linf=Linf,
                     rho_down=float(rho[dwn].mean() / rho1 - 1), B_down=float(by[dwn].mean() / B1 - 1), P_down=float(P[dwn].mean() / P1 - 1)))
for integ in ('unsplit', 'rkl2', 'unsplit_nodual', 'rkl2_nodual', 'rkl2_long', 'rkl2_nodual_long'):
    rs = sorted([r for r in rows if r['integ'] == integ], key=lambda r: r['N'])
    for a, b in zip(rs, rs[1:]): b['order'] = float(np.log2(a['L1'] / b['L1']))
    for r in rs:
        print(f"{integ:15s} N {r['N']:5d}  By profile L1 {r['L1']:.3e} (p {r.get('order', float('nan')):5.2f}) Linf {r['Linf']:.2e}"
              f"  downstream rho {r['rho_down']:+.1e} B {r['B_down']:+.1e} P {r['P_down']:+.1e}  x_mid {r['x_mid']:.4f}")
print(f"exact: V {ref['V']:.4f} r {ref['r']:.4f} W(10-90) {W:.4f} rho1 {rho1:.4f} P1 {P1:.4f} type {ref['ctype']}")
os.makedirs(f'{B}/results/tables', exist_ok=True); json.dump(dict(rows=rows, V=ref['V'], r=ref['r'], W=W), open(f'{B}/results/tables/A10.json', 'w'), indent=1)
if '--check' in sys.argv: sys.exit(1 if bad else 0)
