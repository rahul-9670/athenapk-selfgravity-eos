#!/usr/bin/env python
"""R7 O2/O3: what opacity the transport actually sees, for the production table v9 (md5 a7f55ddd).

O2  table vs its definition, at the table NODES (no interpolation):
    (a) gray kappa_R == kappa_BL (pollack-modified Bell & Lin 1994, the generator's own bell_lin_kappa)
    (b) gray kappa_P / kappa_R and the per-group Planck multipliers mP_g recomputed with an INDEPENDENT quadrature
        (scipy.quad in x, generator's monochromatic shape psi) -- tests the generator's 400-point Simpson + band caps
    (c) the decomposition identity sum_g frac_g(T) mP_g(T) = 1 (Planck mean over bands), at nodes
O3  the runtime lookup (radiation_opacity.hpp bilin: LINEAR in kappa on a (log10 rho, log10 T) grid; group multipliers
    linear in log10 T) vs the generator evaluated directly at cell CENTRES (the worst case of linear interpolation),
    over the whole table and over the box the collapse run visits (rho 1e-20..1e-8 g/cm^3, T 5..2000 K), + the identity
    (c) at cell centres.
usage: r7_opacity.py run | check    -> results/tables/R7_O2_O3_opacity.txt
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_model as M  # noqa: E402  (TabOpacity = the code's lookup, fractions = independent Planck fractions)
from rad_common import MOD  # noqa: E402
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root

GEN = _REPO + "/src/radiation/gen_opacity_table.py"
TABLE = _REPO + "/src/radiation/opacity_table_v9_pollack_fmt3.bin"
LOGI = False
OUT = os.path.join(MOD, "results", "tables", "R7_O2_O3_opacity.txt")
EDGES = [0.0, 1e12, 1e15, 1e30]


def load_gen():
    spec = importlib.util.spec_from_file_location("genop", GEN)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    g.EVAP = "pollack"   # the v9 recipe (R7-O1: regenerated bit-for-bit with --evap pollack)
    g.QUAD = "accurate"  # the TRUTH uses accurate band means (R-F4: the historical Simpson rule is 0.75 % low < 140 K)
    return g


def qmean(G, T, xa, xb, ross, harmonic=False):
    """Independent band mean of psi (scipy.quad, breakpoints at the grain break), same weights as the generator."""
    xa = max(xa, 1e-8)
    if xb <= xa:
        return 1.0
    xbk = G.h_over_k * G.NU_BREAK / T
    pts = [p for p in (xbk,) if xa < p < xb] or None
    w = lambda x: G.spectral_weight(np.array([x]), xa, ross)[0]
    f = lambda x: (1.0 / G.psi(np.array([x]), T)[0] if harmonic else G.psi(np.array([x]), T)[0]) * w(x)
    num = quad(f, xa, xb, points=pts, limit=400, epsabs=0, epsrel=1e-11)[0]
    den = quad(w, xa, xb, points=pts, limit=400, epsabs=0, epsrel=1e-11)[0]
    v = num / den
    return 1.0 / v if harmonic else v


def indep_group(G, T):
    psiR = qmean(G, T, 1e-8, 80.0, 1, True)
    psiP = qmean(G, T, 1e-8, 80.0, 0)
    mP = []
    for g in range(3):
        xa = G.h_over_k * EDGES[g] / T
        xb = (xa + 60.0) if g == 2 else min(G.h_over_k * EDGES[g + 1] / T, xa + 60.0)
        mP.append(qmean(G, T, xa, xb, 0) / psiP)
    return psiP / psiR, np.array(mP)


def stats(a, mask=None):
    a = np.abs(a if mask is None else a[mask])
    return f"max {a.max():.3e}  p99 {np.percentile(a, 99):.3e}  median {np.median(a):.3e}"


def main(mode):
    G = load_gen()
    tab = M.TabOpacity(TABLE, 1.0, 1.0, EDGES, log_interp=LOGI)   # cgs kappa, rho in cgs; LOGI = opacity_interp log
    import struct
    b = open(TABLE, "rb").read()
    _, _, ng, nr, nT = struct.unpack_from("<5q", b, 8)
    n2 = nr * nT
    off = 8 + 40 + 56 + (ng + 1) * 8
    kR = np.frombuffer(b, "<f8", n2, off + 8 * n2).reshape(nr, nT)
    lr = tab.lr0 + tab.dlr * np.arange(nr)
    lT = tab.lT0 + tab.dlT * np.arange(nT)
    L = [f"# R7 O2/O3 (scripts/r7_opacity.py) lookup {'LOG' if LOGI else 'linear'} table {os.path.basename(TABLE)} grid {nr}x{nT}: log10 rho "
         f"[{lr[0]:.3f},{lr[-1]:.3f}] d={tab.dlr:.4f}, log10 T [{lT[0]:.4f},{lT[-1]:.4f}] d={tab.dlT:.5f}",
         f"# generator {GEN} (EVAP = pollack); independent quadrature = scipy.quad (epsrel 1e-11)"]
    # ---- O2a: kappa_R == kappa_BL at every node
    kBL = np.array([[G.bell_lin_kappa(10 ** r, 10 ** t) for t in lT] for r in lr])
    L.append(f"O2a gray kappa_R(node) vs bell_lin_kappa(node): rel {stats(kR / kBL - 1)}")
    # ---- O2b/c at nodes: T-only quantities, every node T
    rows = []
    for j in range(nT):
        T = 10 ** lT[j]
        ratio, mP = indep_group(G, T)
        kP_tab = tab.kP[0, j] / kR[0, j]
        fr = M.fractions(EDGES, T)
        rows.append((T, kP_tab / ratio - 1, np.max(np.abs(tab.mPt[:, j] - mP) / np.maximum(mP, 1e-300) * (fr > 1e-12)),
                     np.sum(fr * tab.mPt[:, j]) - 1, np.sum(fr * mP) - 1))
    r = np.array(rows)
    L.append(f"O2b gray kappa_P/kappa_R (table) vs independent quadrature, all node T: rel {stats(r[:, 1])}")
    L.append(f"O2b per-group mP_g (table) vs independent quadrature (groups with frac > 1e-12): rel {stats(r[:, 2])}")
    L.append(f"O2c sum_g frac_g mP_g - 1 at nodes: table {stats(r[:, 3])} ; independent quadrature {stats(r[:, 4])}")
    for T0 in (5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 2000.0):
        j = int(np.argmin(np.abs(lT - np.log10(T0))))
        L.append(f"    node T={10 ** lT[j]:9.3f} K: kP/kR tab {tab.kP[0, j] / kR[0, j]:.6f} rel {r[j, 1]:+.2e}; "
                 f"mP tab {np.array2string(tab.mPt[:, j], precision=5)} max rel {r[j, 2]:.2e}; sum-1 {r[j, 3]:+.2e}")
    # ---- O3: cell centres
    lrc = lr[:-1] + 0.5 * tab.dlr
    lTc = lT[:-1] + 0.5 * tab.dlT
    kPc_tab = np.array([[tab.kappaP(10 ** rr, 10 ** tt) for tt in lTc] for rr in lrc])
    psiPR = np.array([G.band_mean(10 ** tt, 1e-8, 80.0, 0) / G.band_mean(10 ** tt, 1e-8, 80.0, 1, harmonic=True)
                      for tt in lTc])
    kPc_true = np.array([[G.bell_lin_kappa(10 ** rr, 10 ** tt) for tt in lTc] for rr in lrc]) * psiPR[None, :]
    e = kPc_tab / kPc_true - 1
    box = (np.abs(lrc[:, None] - (-14.0)) <= 6.0) & (lTc[None, :] >= np.log10(5.0)) & (lTc[None, :] <= np.log10(2000.0))
    L.append(f"O3 kappa_P bilinear lookup vs generator at cell centres, whole table: rel {stats(e)}")
    L.append(f"O3 same, collapse run box (rho 1e-20..1e-8, T 5..2000 K): rel {stats(e, box)}")
    i, j = np.unravel_index(np.argmax(np.abs(np.where(box, e, 0))), e.shape)
    L.append(f"    worst in box: rho {10 ** lrc[i]:.3e} g/cm^3, T {10 ** lTc[j]:.4g} K, lookup/true - 1 = {e[i, j]:+.3e}")
    for T1, T2 in ((5, 140), (140, 200), (200, 900), (900, 1300), (1300, 2000)):
        m = box & (lTc[None, :] >= np.log10(T1)) & (lTc[None, :] < np.log10(T2))
        L.append(f"    box, T in [{T1},{T2}) K: rel {stats(e, m)}")
    # where the large errors are: T cells (inside the box) whose worst |error| exceeds 5 %
    worst_T = np.max(np.abs(np.where(box, e, 0.0)), axis=0)
    for j in np.where(worst_T > 0.05)[0]:
        L.append(f"    T cell {10 ** lT[j]:.4g}-{10 ** lT[j + 1]:.4g} K (centre {10 ** lTc[j]:.4g}): worst |lookup/true - 1| "
                 f"{worst_T[j]:.3e} over rho in box")
    # multipliers at cell-centre T + identity
    mc = []
    for tt in lTc:
        T = 10 ** tt
        _, mP = indep_group(G, T) if (T >= 5.0 and T <= 2000.0) else (None, None)
        if mP is None:
            continue
        fr = M.fractions(EDGES, T)
        mt = tab.mP(T)
        mc.append((T, np.max(np.abs(mt - mP) / np.maximum(mP, 1e-300) * (fr > 1e-12)), np.sum(fr * mt) - 1))
    mc = np.array(mc)
    for T, err, _ in mc[mc[:, 1] > 0.05]:
        L.append(f"    mP lookup worst > 5 %: cell-centre T {T:.4g} K, max rel {err:.3e}")
    L.append(f"O3 mP_g lookup (linear in log T) vs independent at cell-centre T in 5..2000 K: rel {stats(mc[:, 1])}")
    L.append(f"O3 sum_g frac_g mP_g - 1 at cell-centre T in 5..2000 K (lookup): {stats(mc[:, 2])}")
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # optional: r7_opacity.py run|check [table] [out]  (default: the production table v9 -> R7_O2_O3_opacity.txt)
    if len(sys.argv) > 2:
        TABLE = sys.argv[2]
        OUT = sys.argv[3]
        LOGI = len(sys.argv) > 4 and sys.argv[4] == "log"
    main(sys.argv[1])
