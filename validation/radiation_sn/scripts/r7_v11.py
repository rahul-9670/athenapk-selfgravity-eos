#!/usr/bin/env python
"""R-F9 fix check: opacity table v11 (gen_opacity_table.py --shape gasfrac; otherwise the v10b recipe).

The shape model: psi = (1 - f_gas) d(x)/<d>_R + f_gas, f_gas = kappa_gas/kappa_BL (BL94 gas branch, pollack evap).
Truth here is INDEPENDENT of the generator: own BL94 walk (as r7_o6_dust.kappa_from), own dust shape, scipy.quad.
(a) table vs truth at nodes (sampled rho x every-4th T): kappa_R = kappa_BL, kappa_P/kappa_R (rho-dependent now)
(b) per-group multipliers mP_g at rho_ref = 1e-12 (the table format stores them T-only) vs truth at every node T
(c) the T-only multiplier approximation: mult(T) x kappa_P,gray(rho,T) vs true kappa_P,g(rho,T) in the collapse-run box
(d) log-kappa lookup (radiation/opacity_interp = log, r1_model.TabOpacity) vs truth at sampled cell centres in the box
(e) kappa_P/kappa_R vs T at rho = 1e-12: v10b (cascade) vs v11 (gasfrac)
(f) 1 - f_gas == the C++ dust weight DustOpacityWeightEvap(pollack) (scripts/host/dust_factor_dump.cpp, column 10)
usage: r7_v11.py run | check   -> results/tables/R7_v11_gasfrac.txt
"""
import os
import struct
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_model as M  # noqa: E402
from rad_common import MOD  # noqa: E402

V11 = os.path.join(MOD, "runs", "R7_v11", "opacity_table_v11_pollack_acc_gasfrac_fmt3.bin")
V10B = os.path.join(MOD, "runs", "R7_v10", "opacity_table_v10b_pollack_acc_fmt3.bin")
DUMP = os.path.join(MOD, "runs", "R7_O6_dump.txt")
OUT = os.path.join(MOD, "results", "tables", "R7_v11_gasfrac.txt")
EDGES = [0.0, 1e12, 1e15, 1e30]
HK = 6.62607015e-27 / 1.380649e-16
BETA, NUB = 1.8, 1.0e14


def kbl(rho, T, istart=0):
    k0 = np.array([2.0e-4, 2.0e16, 0.1, 2.0e81, 1.0e-8, 1.0e-36, 1.5e20, 0.348])
    aa = np.array([0.0, 0.0, 0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0])
    bb = np.array([2.0, -7.0, 0.5, -24.0, 3.0, 10.0, -2.5, 0.0])
    k0[3] = 0.1 * 2000.0 ** 24.5
    aa[3] = 0.0195 * 24.5
    Tc = lambda p, q: ((k0[p] / k0[q]) * rho ** (aa[p] - aa[q])) ** (1.0 / (bb[q] - bb[p]))
    i = istart
    while i < 7:
        j = i + 1
        while j < 7 and Tc(j, j + 1) <= Tc(i, j):
            j += 1
        if T < Tc(i, j):
            return k0[i] * rho ** aa[i] * T ** bb[i]
        i = j
    return k0[7] * rho ** aa[7] * T ** bb[7]


def fgas(rho, T):
    return min(max(kbl(rho, T, 4) / kbl(rho, T), 0.0), 1.0)


def wgt(x, ross, xa=0.0):
    """Planck / Rosseland weight times e^{xa} (exponent shifted so far Wien bands do not underflow)."""
    if ross:
        return x ** 4 * np.exp(xa - x) / (-np.expm1(-x)) ** 2
    return x ** 3 * np.exp(xa - x) / (-np.expm1(-x))


def mean(T, xa, xb, ross, harm, shape):
    xa = max(xa, 1e-8)
    xbk = HK * NUB / T
    pts = [p for p in (xbk,) if xa < p < xb] or None
    f = (lambda x: wgt(x, ross, xa) / shape(x)) if harm else (lambda x: wgt(x, ross, xa) * shape(x))
    num = quad(f, xa, xb, points=pts, limit=500, epsabs=0, epsrel=1e-11)[0]
    den = quad(lambda x: wgt(x, ross, xa), xa, xb, points=pts, limit=500, epsabs=0, epsrel=1e-11)[0]
    v = num / den
    return 1.0 / v if harm else v


def truth(rho, T, groups=False):
    xb = HK * NUB / T
    d = lambda x: min((x / xb) ** BETA, 1.0)
    D = mean(T, 1e-8, 80.0, 1, True, d)
    fg = fgas(rho, T)
    sh = lambda x: (1.0 - fg) * d(x) / D + fg
    R = mean(T, 1e-8, 80.0, 1, True, sh)
    P = mean(T, 1e-8, 80.0, 0, False, sh)
    if not groups:
        return P / R
    mP = []
    for g in range(3):
        xa = HK * EDGES[g] / T
        xbb = (xa + 60.0) if g == 2 else min(HK * EDGES[g + 1] / T, xa + 60.0)
        mP.append(mean(T, xa, xbb, 0, False, sh) / P if xbb > max(xa, 1e-8) else 1.0)
    return P / R, np.array(mP)


def read(path):
    b = open(path, "rb").read()
    _, _, ng, nr, nT = struct.unpack_from("<5q", b, 8)
    lr0, dlr, lT0, dlT = struct.unpack_from("<4d", b, 48)
    off = 8 + 40 + 56 + (ng + 1) * 8
    n2 = nr * nT
    kP = np.frombuffer(b, "<f8", n2, off).reshape(nr, nT)
    kR = np.frombuffer(b, "<f8", n2, off + 8 * n2).reshape(nr, nT)
    mP = np.frombuffer(b, "<f8", ng * nT, off + 24 * n2).reshape(ng, nT)
    return lr0 + dlr * np.arange(nr), lT0 + dlT * np.arange(nT), kP, kR, mP


def st(a):
    a = np.abs(np.asarray(a))
    return f"max {a.max():.3e}  p99 {np.percentile(a, 99):.3e}  median {np.median(a):.3e}  (n {a.size})"


def main(mode):
    lr, lT, kP, kR, mP = read(V11)
    L = [f"# R-F9 table v11 check (scripts/r7_v11.py): {os.path.basename(V11)} grid {len(lr)}x{len(lT)}; truth independent "
         "of the generator (own BL94 pollack walk, own shape, scipy.quad epsrel 1e-11)"]
    # (a) nodes
    ir = np.arange(0, len(lr), 10)
    jT = np.arange(0, len(lT), 4)
    eR, eP, eP_box = [], [], []
    for i in ir:
        for j in jT:
            rho, T = 10 ** lr[i], 10 ** lT[j]
            eR.append(kR[i, j] / kbl(rho, T) - 1)
            e = (kP[i, j] / kR[i, j]) / truth(rho, T) - 1
            eP.append(e)
            if -20 <= lr[i] <= -8 and 5 <= T <= 2000:
                eP_box.append(e)
    L.append(f"(a) kappa_R(node) vs BL94 pollack: rel {st(eR)}")
    L.append(f"(a) kappa_P/kappa_R(node) vs truth, rho every 10th x T every 4th node: rel {st(eP)}")
    L.append(f"(a) same in the collapse-run box (rho 1e-20..1e-8, T 5..2000 K): rel {st(eP_box)}")
    # (b) multipliers at rho_ref, every 4th node T (all nodes with a live group)
    em = []
    for j in jT:
        T = 10 ** lT[j]
        _, mt = truth(1.0e-12, T, groups=True)
        fr = M.fractions(EDGES, T)
        live = fr > 1e-12
        em.append(np.max(np.abs(mP[:, j] - mt)[live] / np.maximum(mt[live], 1e-300)))
    L.append(f"(b) mP_g(node T) vs truth at rho_ref 1e-12, groups with Planck fraction > 1e-12: rel {st(em)}")
    # (c) T-only multiplier approximation in the box: per-group kappa_P,g(rho,T) = mult(T) kP_gray(rho,T) vs truth
    ec, worst = [], (0, 0, 0)
    for lrho in (-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0):
        i = int(np.argmin(np.abs(lr - lrho)))
        for j in jT:
            T = 10 ** lT[j]
            if not (5 <= T <= 2000):
                continue
            ratio, mt = truth(10 ** lr[i], T, groups=True)
            fr = M.fractions(EDGES, T)
            live = fr > 1e-3
            e = np.max(np.abs(mP[:, j] / mt - 1)[live])
            ec.append(e)
            if e > worst[0]:
                worst = (e, lr[i], T)
    L.append(f"(c) T-only multipliers: mult(T) vs the multiplier at the cell's own rho, box, groups with Planck "
             f"fraction > 1e-3: rel {st(ec)}; worst at log rho {worst[1]:.1f}, T {worst[2]:.4g} K")
    # (d) log lookup at cell centres in the box (sampled)
    tab = M.TabOpacity(V11, 1.0, 1.0, EDGES, log_interp=True)
    ed = []
    for lrho in np.arange(-19.95, -8.0, 0.5):
        for lt in np.linspace(np.log10(5.0), np.log10(2000.0), 97)[:-1] + 0.5 * (lT[1] - lT[0]):
            rho, T = 10 ** lrho, 10 ** lt
            ed.append(tab.kappaP(rho, T) / (kbl(rho, T) * truth(rho, T)) - 1)
    L.append(f"(d) kappa_P log-lookup vs truth at off-node points in the box (24 rho x 96 T): rel {st(ed)}")
    # (e) kappa_P/kappa_R vs T at rho 1e-12, v10b vs v11
    lr10, lT10, kP10, kR10, _ = read(V10B)
    i11 = int(np.argmin(np.abs(lr - (-12.0))))
    i10 = int(np.argmin(np.abs(lr10 - (-12.0))))
    row = []
    for T0 in (10, 100, 139.9, 140.1, 141, 170, 200, 500, 800, 1000, 1200, 1500, 2000, 3000):
        j11 = int(np.argmin(np.abs(lT - np.log10(T0))))
        j10 = int(np.argmin(np.abs(lT10 - np.log10(T0))))
        row.append(f"{10 ** lT[j11]:.5g}K {kP10[i10, j10] / kR10[i10, j10]:.4f}->{kP[i11, j11] / kR[i11, j11]:.4f}")
    L.append("(e) kappa_P/kappa_R at rho 1e-12, v10b -> v11: " + " | ".join(row))
    jj = (10 ** lT[:-1] >= 100.0) & (10 ** lT[1:] <= 300.0)
    d10 = np.abs(np.diff(kP10[i10] / kR10[i10]))[jj]
    d11 = np.abs(np.diff(kP[i11] / kR[i11]))[jj]
    L.append(f"(e) largest node-to-node change of kappa_P/kappa_R over 100-300 K (rho 1e-12, dlogT 0.0031): v10b "
             f"{d10.max():.4f} at {10 ** lT[:-1][jj][np.argmax(d10)]:.4g} K, v11 {d11.max():.2e} at {10 ** lT[:-1][jj][np.argmax(d11)]:.4g} K")
    # (f) C++ dust weight vs 1 - f_gas
    d = np.loadtxt(DUMP)
    dev = max(abs(d[k, 9] - (1.0 - fgas(10.0 ** d[k, 0], d[k, 1]))) for k in range(len(d)))
    L.append(f"(f) C++ DustOpacityWeightEvap(pollack) vs 1 - f_gas of the table shape: max |dw| {dev:.2e} over {len(d)} points")
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
