#!/usr/bin/env python
"""R-F11 fix: opacity table v12 = Semenov et al. 2003 dust (monochromatic, nrm_h_s) + Helling 2001 gas, format v4
(gen_opacity_table.py --model semenov, src/radiation/opacity_semenov.py).

(a) column identification: Planck / Rosseland means of each monochromatic column of nrm/Hom_spheres/<region>.dat at
    region-interior T vs the authors' own code (opacity.f, runs/R7_O5 outputs, rho = 1e-13): col 2 (abs + (1-g) sca)
    is the column the code's (extinction) means were computed from.
(b) gas port: opacity_semenov.Helling vs opacity.f gop through cop, every R7_O5 point above T5 + dT5 (4 rho, P and R).
(c) table vs an INDEPENDENT evaluation at sampled nodes: own reading of the raw 113-point files, own log-log
    interpolation, scipy.integrate.quad per band, own region blend -> kappa_P, kappa_R, kappa_P,g, kappa_R,g.
(d) the C++ loaders (scripts/host/opacity_v4_dump.cpp: OpacityTable::Load + BuildGroupOpacityTableFromFile, linear and
    log lookup, dust weights) vs a numpy bilinear reading of the same file at random off-node (rho, T).
(e) v12 / v11 in the collapse run's range (rho 1e-16 .. 1e-10, T 5 .. 1500 K): the size of the R-F11 change.
(f) the 10-30 K gap between Semenov's polynomial fit and the monochromatic integral (fit residual, not v11).
usage: r7_v12.py run | check   -> results/tables/R7_v12_semenov.txt
"""
import os
import struct
import subprocess
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO + "/src/radiation")
import opacity_semenov as SM  # noqa: E402
from r7_v11 import V11, read  # noqa: E402
from rad_common import MOD  # noqa: E402
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root

SD = os.environ.get("SEMENOV_DIR", os.path.join(MOD, "literature", "semenov2003_download"))  # authors' download, not shipped
V12 = os.path.join(MOD, "runs", "R7_v12", "opacity_table_v12_semenov_hs_fmt4.bin")
DUMP = os.path.join(MOD, "runs", "R7_v12", "opacity_v4_dump")
OUT = os.path.join(MOD, "results", "tables", "R7_v12_semenov.txt")
C, HK = 2.99792458e10, 6.62607015e-27 / 1.380649e-16


def read4(path):
    b = open(path, "rb").read()
    _, ver, fl, ng, nr, nT = struct.unpack_from("<6q", b, 0)
    lr0, dlr, lT0, dlT = struct.unpack_from("<4d", b, 48)
    off = 48 + 56 + (ng + 1) * 8
    n2 = nr * nT
    arr = lambda k, n, shp: np.frombuffer(b, "<f8", n, off + 8 * k).reshape(shp)  # noqa: E731
    kP, kR = arr(0, n2, (nr, nT)), arr(n2, n2, (nr, nT))
    mP, mR = arr(3 * n2, ng * n2, (ng, nr, nT)), arr((3 + ng) * n2, ng * n2, (ng, nr, nT))
    wdP, wdR = arr((3 + 2 * ng) * n2, n2, (nr, nT)), arr((4 + 2 * ng) * n2, n2, (nr, nT))
    edges = np.frombuffer(b, "<f8", ng + 1, 104)
    return dict(ver=ver, fl=fl, ng=ng, lr=lr0 + dlr * np.arange(nr), lT=lT0 + dlT * np.arange(nT), kP=kP, kR=kR,
                mP=mP, mR=mR, wdP=wdP, wdR=wdR, edges=edges)


def raw(region, model="Hom_spheres"):
    rows = [[float(t.replace("D", "E")) for t in ln.split()]
            for ln in open(os.path.join(SD, "nrm", model, f"{region}.dat")) if ln.strip()]
    a = np.array(rows)
    nu = C / (a[:, 0] * 1e-4)
    o = np.argsort(nu)
    return nu[o], a[o, 1:]


def kfun(nu, k):
    ln, lk = np.log(nu), np.log(k)
    s0 = (lk[1] - lk[0]) / (ln[1] - ln[0])

    def f(x):   # x = ln nu
        if x < ln[0]:
            return np.exp(lk[0] + s0 * (x - ln[0]))
        return np.exp(np.interp(x, ln, lk))
    return f


def qmean(f, T, a, b, ross):
    """Band mean by scipy quad in ln nu; weights normalised in log space (no underflow). None if the band carries
    less than 1e-12 of the full-spectrum weight (its mean then has no effect on any coupling)."""
    la = max(np.log(a) if a > 0 else -1e9, np.log(1e8))
    lb = min(np.log(b) if b < 1e299 else 1e9, np.log(1e17))
    p = 5 if ross else 4
    xs = np.linspace(la, lb, 4001)
    ys = HK * np.exp(xs) / T
    lw_raw = p * xs - ys - (2 if ross else 1) * np.log(-np.expm1(-ys))
    c = float(lw_raw.max())

    def w(x):
        y = HK * np.exp(x) / T
        return np.exp(p * x - y - (2 if ross else 1) * np.log(-np.expm1(-y)) - c)
    pk = np.log(3.9 * T / HK) if ross else np.log(2.8 * T / HK)
    pts = [pk] if la < pk < lb else None
    den = quad(w, la, lb, limit=400, points=pts, epsrel=1e-11)[0]
    xf = np.linspace(np.log(1e8), np.log(1e17), 8001)
    yf = HK * np.exp(xf) / T
    full = np.trapezoid(np.exp(p * xf - yf - (2 if ross else 1) * np.log(-np.expm1(-yf)) - c), xf)
    if not den > 1e-12 * full:
        return None
    if ross:
        return den / quad(lambda x: w(x) / f(x), la, lb, limit=400, points=pts, epsrel=1e-11)[0]
    return quad(lambda x: w(x) * f(x), la, lb, limit=400, points=pts, epsrel=1e-11)[0] / den


def fort(kind, T, rho="1e-13"):
    a = np.loadtxt(os.path.join(MOD, "runs", "R7_O5", f"nrmhs_{kind}_{rho}", f"k{kind}.out"), skiprows=1)
    return float(np.interp(T, a[:, 0], a[:, 1]))


def bil(A, t, lr, lT):
    fi = (lr - t["lr"][0]) / (t["lr"][1] - t["lr"][0]); i = min(max(int(fi), 0), len(t["lr"]) - 2)
    fj = (lT - t["lT"][0]) / (t["lT"][1] - t["lT"][0]); j = min(max(int(fj), 0), len(t["lT"]) - 2)
    u, v = min(max(fi - i, 0), 1), min(max(fj - j, 0), 1)
    return (1 - u) * (1 - v) * A[..., i, j] + u * (1 - v) * A[..., i + 1, j] + (1 - u) * v * A[..., i, j + 1] + \
        u * v * A[..., i + 1, j + 1]


def lines():
    L = ["# R7-v12: Semenov 2003 (nrm_h_s) + Helling 2001 opacity table, format v4 (scripts/r7_v12.py)", ""]
    t = read4(V12)
    L.append(f"table {os.path.basename(V12)}: format v{t['ver']} flags {t['fl']} ng {t['ng']} grid {len(t['lr'])}x"
             f"{len(t['lT'])} edges {[float(e) for e in t['edges']]}")
    # (a)
    L += ["", "(a) monochromatic column means vs opacity.f (rho 1e-13): ratio col_k / code, Planck | Rosseland"]
    L.append("region T[K]   col2 P   col3 P   col4 P  |  col2 R   col3 R   col4 R")
    for reg, T in ((1, 50), (1, 100), (2, 200), (3, 350), (4, 550), (5, 1000)):
        nu, K = raw(reg)
        rp = [qmean(kfun(nu, K[:, c]), T, 0, 1e300, False) / fort("P", T) for c in (0, 1, 2)]
        rr = [qmean(kfun(nu, K[:, c]), T, 0, 1e300, True) / fort("R", T) for c in (0, 1, 2)]
        L.append(f"  {reg}  {T:6.0f}  " + " ".join(f"{x:8.3f}" for x in rp) + "  | " +
                 " ".join(f"{x:8.3f}" for x in rr))
    # (b)
    H = SM.Helling(SD)
    worst = 0.0
    for rho in ("1e-16", "1e-13", "1e-11", "1e-09"):
        for kind in "PR":
            a = np.loadtxt(os.path.join(MOD, "runs", "R7_O5", f"nrmhs_{kind}_{rho}", f"k{kind}.out"), skiprows=1)
            T5 = float(SM.evap_temperatures(float(rho))[4])
            for T, k in a[a[:, 0] > T5 + 100.0]:
                worst = max(worst, abs(np.log(H.kappa(kind, float(rho), float(T)) / k)))
    L += ["", f"(b) Helling gas port vs opacity.f (496 points, 4 rho, P and R): max |ln ratio| = {worst:.2e} "
              "(code prints 6 digits)"]
    # (c)
    L += ["", "(c) table vs independent quad evaluation at sampled nodes: max |rel| over kP, kR, kP_g, kR_g"]
    rg = [raw(r) for r in range(1, 6)]
    fa = [kfun(nu, K[:, 2]) for nu, K in rg]
    ft = [kfun(nu, K[:, 0]) for nu, K in rg]
    rng = np.random.default_rng(12)
    worst_c, nn = 0.0, 0
    jlo, jhi = int(np.searchsorted(t["lT"], np.log10(3.0))), int(np.searchsorted(t["lT"], np.log10(2000.0)))
    for _ in range(120):
        i = int(rng.integers(20, 200)); j = int(rng.integers(jlo, jhi))
        rho, T = 10.0**t["lr"][i], 10.0**t["lT"][j]
        w = SM.region_weights(rho, T)
        gP = H.kappa("P", rho, T) if w[5] > 0 else 0.0
        gR = H.kappa("R", rho, T) if w[5] > 0 else 0.0
        bands = [(0.0, 1e300)] + [(t["edges"][g], t["edges"][g + 1]) for g in range(t["ng"])]
        vals = []
        for (a, b) in bands:
            qp = [qmean(fa[r], T, a, b, False) if w[r] > 0 else 0.0 for r in range(5)]
            qr = [qmean(ft[r], T, a, b, True) if w[r] > 0 else 0.0 for r in range(5)]
            P = None if any(q is None for q in qp) else sum(w[r] * qp[r] for r in range(5)) + w[5] * gP
            R = None if any(q is None for q in qr) else sum(w[r] * qr[r] for r in range(5)) + w[5] * gR
            if w[5] == 1.0:
                P, R = gP, gR
            vals.append((P, R))
        tab = [(t["kP"][i, j], t["kR"][i, j])] + [(t["kP"][i, j] * t["mP"][g, i, j], t["kR"][i, j] * t["mR"][g, i, j])
                                                  for g in range(t["ng"])]
        for (P, R), (tp, tr) in zip(vals, tab):
            for x, y in ((P, tp), (R, tr)):
                if x is not None and x > 0:
                    worst_c = max(worst_c, abs(y / x - 1))
        nn += 1
    L.append(f"  {nn} nodes (T 3..2000 K): max |table/quad - 1| = {worst_c:.2e}")
    # (d)
    pts = np.column_stack([10 ** rng.uniform(-19.5, -1, 2000), 10 ** rng.uniform(0.0, 4.5, 2000)])
    inp = "\n".join(f"{r:.17e} {T:.17e}" for r, T in pts)
    out = subprocess.run([DUMP, V12, str(t["ng"])], input=inp, capture_output=True, text=True, check=True).stdout
    d = np.array([[float(x) for x in ln.split()] for ln in out.splitlines() if not ln.startswith("#")])
    worst_d = {"lin": 0.0, "log": 0.0, "w": 0.0}
    for row in d:
        lr, lT = np.log10(row[0]), np.log10(row[1])
        ref_lin = [bil(t["kP"], t, lr, lT), bil(t["kR"], t, lr, lT)]
        ref_log = [np.exp(bil(np.log(t["kP"]), t, lr, lT)), np.exp(bil(np.log(t["kR"]), t, lr, lT))]
        for g in range(t["ng"]):
            ref_lin += [bil(t["mP"][g], t, lr, lT), bil(t["mR"][g], t, lr, lT)]
            ref_log += [np.exp(bil(np.log(t["mP"][g]), t, lr, lT)), np.exp(bil(np.log(t["mR"][g]), t, lr, lT))]
        c_lin = [row[2], row[3]] + [row[8 + 4 * g + q] for g in range(t["ng"]) for q in (0, 1)]
        c_log = [row[4], row[5]] + [row[8 + 4 * g + q] for g in range(t["ng"]) for q in (2, 3)]
        worst_d["lin"] = max(worst_d["lin"], max(abs(a / b - 1) for a, b in zip(c_lin, ref_lin)))
        worst_d["log"] = max(worst_d["log"], max(abs(a / b - 1) for a, b in zip(c_log, ref_log)))
        worst_d["w"] = max(worst_d["w"], abs(row[6] - bil(t["wdP"], t, lr, lT)), abs(row[7] - bil(t["wdR"], t, lr, lT)))
    L += ["", f"(d) C++ loaders vs numpy, 2000 random (rho, T): linear max rel {worst_d['lin']:.2e}, log max rel "
              f"{worst_d['log']:.2e}, dust weights max abs {worst_d['w']:.2e}"]
    # (e)
    lr1, lT1, kP1, kR1, _ = read(V11)
    L += ["", "(e) v12 / v11 (gray kappa_P | kappa_R) in the collapse-run range"]
    L.append("T[K]    " + "".join(f"   rho={r:.0e}   " for r in (1e-16, 1e-13, 1e-10)))
    for T in (5, 10, 20, 50, 100, 130, 150, 170, 250, 350, 500, 800, 1000, 1500):
        j = int(np.argmin(abs(t["lT"] - np.log10(T))))
        s = f"{10**t['lT'][j]:7.1f} "
        for r in (1e-16, 1e-13, 1e-10):
            i = int(round((np.log10(r) - t["lr"][0]) / (t["lr"][1] - t["lr"][0])))
            s += f"  {t['kP'][i, j] / kP1[i, j]:6.2f} | {t['kR'][i, j] / kR1[i, j]:6.2f}  "
        L.append(s)
    # (f)
    L += ["", "(f) Semenov polynomial fit / monochromatic integral (rho 1e-13): the fit residual at the bottom of region 1"]
    nu, K = raw(1)
    for T in (10, 15, 20, 30, 50):
        L.append(f"  T {T:3d} K: Planck fit/mono {fort('P', T) / qmean(kfun(nu, K[:, 0]), T, 0, 1e300, False):.3f}  "
                 f"Rosseland fit/mono {fort('R', T) / qmean(kfun(nu, K[:, 0]), T, 0, 1e300, True):.3f}")
    return L


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    L = lines()
    if mode == "run":
        open(OUT, "w").write("\n".join(L) + "\n")
        print("\n".join(L))
    else:
        assert open(OUT).read().splitlines() == L, "R7_v12_semenov.txt differs from a fresh computation"
        print("CHECK OK", OUT)
