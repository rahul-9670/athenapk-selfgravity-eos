#!/usr/bin/env python3
"""Semenov et al. (2003, A&A 410, 611) dust + Helling (2001) gas opacities for gen_opacity_table.py --model semenov.

R-F11 (validation/radiation_sn, 2026-10-09). The production table v11 is a Bell & Lin (1994) magnitude with an
analytic modified-blackbody shape; against Semenov it is kappa_P 2.5x low at 10-100 K, x0.32 kappa_R at 275-425 K and
1e1-1e4x low in the gas regime. This module replaces BOTH the magnitude and the shape with the authors' own data:

* DUST: the MONOCHROMATIC opacities distributed with the paper (folder `nrm/<topology_shape>/<region>.dat`, "normal"
  silicates Fe/(Fe+Mg) = 0.3), one file per temperature region 1..5 (ice / volatile organics / refractory organics /
  troilite / silicates+iron). Columns, VERIFIED here by reproducing the code's own mean opacities (R7 O5 binary
  runs/R7_O5): lambda [um], kappa_abs + (1-g) kappa_sca (transport extinction), kappa_ext, kappa_abs, kappa_sca, all
  [cm^2 per g of GAS] (dust-to-gas 0.01 built in). Col 2 reproduces Semenov's "Planck" and "Rosseland" means (both are
  EXTINCTION means in opacity.f) to 1-3 % at 100-350 K and 1-17 % in regions 2-5; at 10 K the code's 5th-order
  polynomial fit is 2.1x above the monochromatic integral (a fit residual at the bottom of region 1).
  This table uses kappa_abs (col 4) for the Planck (emission/absorption) means -- emission is absorption, never
  extinction -- and the transport extinction (col 2) for the Rosseland (flux) means.
* REGIONS AND BLENDS: exactly opacity.f `cop`: evaporation temperatures T(1) = water ice (rho-dependent, `bint`,
  linear in rho on Semenov's 8-node grid, clamped to it here), T(2) = 275 K, T(3) = 425 K, T(4) = 680 K,
  T(5) = min(max(T_Fe, T_opx), max(T_opx, T_ol)) (rho-dependent); half-widths dT = 5, 5, 15, 5, 100 K; inside
  |T - T(k)| <= dT(k) the mean is BB - AA sin(pi/2 (T - T(k))/dT(k)) between region k and region k+1 (k = 5: gas).
  One deviation: opacity.f evaluates its two end-member FITS at the band edges T(k) -+ dT(k); here the end-member
  group means are evaluated at the actual T (their Planck weights follow T). Both blends are linear in the MEANS
  (not in kappa_nu), as in opacity.f.
* GAS: Helling's Planck / Rosseland mean tables kP_h2001.dat / kR_h2001.dat (shipped with opacity.f) via `gop`/`EINT`
  ported exactly (log10 kappa, linear in T then linear in rho on the 71 x 71 grid). Gray within groups (no
  frequency-resolved gas data): every group gets the gas Planck mean for emission and the gas Rosseland mean for
  flux. Outside Helling's grid: rho clamped to [2.364e-18, 2.364e-7] (opacity.f returns ZERO there); above 1e4 K
  the caller's Bell & Lin is used, blended log-linearly over [9e3, 1e4] K.
* DUST WEIGHT: the grains' share of the gray kappa_P and kappa_R, for dust/opacity_evap = table (the dust package
  scales only the grain part; it must split the opacity exactly as this table does).

Every number reproduced from a file is checked in validation/radiation_sn/scripts/r7_v12.py.
"""
import os

import numpy as np

C_LIGHT = 2.99792458e10
H_K = 6.62607015e-27 / 1.380649e-16

# opacity.f cop: evaporation-temperature data (rho grid, then [ice, Fe, opx, ol] x 8)
_RO = np.array([1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
_TT = np.array([[109.0, 118.0, 129.0, 143.0, 159.0, 180.0, 207.0, 244.0],
                [835.0, 908.0, 994.0, 1100.0, 1230.0, 1395.0, 1612.0, 1908.0],
                [902.0, 980.0, 1049.0, 1129.0, 1222.0, 1331.0, 1462.0, 1621.0],
                [929.0, 997.0, 1076.0, 1168.0, 1277.0, 1408.0, 1570.0, 1774.0]])
_DT = np.array([5.0, 5.0, 15.0, 5.0, 100.0])
# opacity.f gop: Helling's grid
_GT = np.array([500.00, 521.86, 544.68, 568.50, 593.35, 619.30, 646.38, 674.64, 704.14, 734.93, 767.06, 800.60,
                835.61, 872.15, 910.28, 950.08, 991.63, 1034.99, 1080.24, 1127.47, 1176.77, 1228.23, 1281.93,
                1337.99, 1396.49, 1457.55, 1521.28, 1587.80, 1657.23, 1729.69, 1805.32, 1884.26, 1966.65, 2052.64,
                2142.39, 2236.07, 2333.84, 2435.89, 2542.40, 2653.56, 2769.59, 2890.69, 3017.09, 3149.01, 3286.70,
                3430.41, 3580.41, 3736.96, 3900.36, 4070.91, 4248.91, 4434.69, 4628.60, 4830.98, 5042.22, 5262.69,
                5492.80, 5732.98, 5983.65, 6245.29, 6518.36, 6803.38, 7100.86, 7411.34, 7735.41, 8073.64, 8426.66,
                8795.12, 9179.68, 9581.07, 10000.00])
_GR = np.array([0.2364E-06, 0.1646E-06, 0.1147E-06, 0.7985E-07, 0.5560E-07, 0.3872E-07, 0.2697E-07, 0.1878E-07,
                0.1308E-07, 0.9107E-08, 0.6342E-08, 0.4417E-08, 0.3076E-08, 0.2142E-08, 0.1492E-08, 0.1039E-08,
                0.7234E-09, 0.5038E-09, 0.3508E-09, 0.2443E-09, 0.1701E-09, 0.1185E-09, 0.8252E-10, 0.5746E-10,
                0.4002E-10, 0.2787E-10, 0.1941E-10, 0.1352E-10, 0.9412E-11, 0.6554E-11, 0.4565E-11, 0.3179E-11,
                0.2214E-11, 0.1542E-11, 0.1074E-11, 0.7476E-12, 0.5206E-12, 0.3626E-12, 0.2525E-12, 0.1758E-12,
                0.1225E-12, 0.8528E-13, 0.5939E-13, 0.4136E-13, 0.2880E-13, 0.2006E-13, 0.1397E-13, 0.9727E-14,
                0.6774E-14, 0.4717E-14, 0.3285E-14, 0.2288E-14, 0.1593E-14, 0.1109E-14, 0.7726E-15, 0.5381E-15,
                0.3747E-15, 0.2609E-15, 0.1817E-15, 0.1265E-15, 0.8813E-16, 0.6137E-16, 0.4274E-16, 0.2976E-16,
                0.2073E-16, 0.1443E-16, 0.1005E-16, 0.7000E-17, 0.4875E-17, 0.3395E-17, 0.2364E-17])
FOLDERS = {"h_s": "Hom_spheres", "h_a": "Hom_aggregates", "c_s": "Comp_spheres", "c_a": "Comp_aggregates",
           "m_s": "Multishell_spheres", "p_s": "Porous_comp_spheres", "p_m": "Porous_multishell_spheres"}


def _bint(x, xs, ys):
    """opacity.f bint: linear in x on the bracketing pair, end pairs extrapolated (here: clamped to the grid)."""
    x = np.clip(x, xs[0], xs[-1])
    return np.interp(x, xs, ys)


def evap_temperatures(rho):
    """T(1..5) of opacity.f cop at gas density rho [g/cm^3] (array ok)."""
    rho = np.asarray(rho, dtype=float)
    tev = [_bint(rho, _RO, _TT[i]) for i in range(4)]
    t5 = np.minimum(np.maximum(tev[1], tev[2]), np.maximum(tev[2], tev[3]))
    return np.stack([tev[0], np.full_like(rho, 275.0), np.full_like(rho, 425.0), np.full_like(rho, 680.0), t5])


def region_weights(rho, T):
    """Weights w[0..5] (regions 1..5, gas) of the opacity.f cop blend at (rho, T); sum = 1."""
    Tk = evap_temperatures(rho)
    T = float(T)
    kk = 6
    if T <= Tk[0] + _DT[0]:
        kk = 1
    for it in range(2, 6):
        if Tk[it - 2] + _DT[it - 2] < T <= Tk[it - 1] + _DT[it - 1]:
            kk = it
    w = np.zeros(6)
    if kk == 6:
        w[5] = 1.0
        return w
    smooth = any(abs(T - Tk[i]) <= _DT[i] for i in range(5))
    if smooth:
        s = np.sin(np.pi / 2.0 / _DT[kk - 1] * (T - Tk[kk - 1]))
        w[kk - 1] = 0.5 * (1.0 - s)    # BB - AA sin = L (1 - s)/2 + R (1 + s)/2
        w[kk] = 0.5 * (1.0 + s)        # kk = 5 -> w[5] = gas
    else:
        w[kk - 1] = 1.0
    return w


class Helling:
    """opacity.f init_g + gop + EINT, ported exactly (log10 kappa grids, linear in T then linear in rho)."""

    def __init__(self, sdir):
        self.g = {}
        for kind in ("P", "R"):
            txt = open(os.path.join(sdir, "opacity", f"k{kind}_h2001.dat")).read().split()
            v = np.array([float(t) for t in txt[1:1 + 71 * 71]])
            self.g[kind] = v.reshape(71, 71)    # eG(i = rho, j = T)

    def kappa(self, kind, rho, T):
        D = self.g[kind]
        rho = float(np.clip(rho, _GR[-1], _GR[0]))
        T = float(np.clip(T, _GT[0], _GT[-1]))
        ip = int(np.searchsorted(_GT, T, side="left"))        # first Y(i) >= YP, 1-based i -> ip = i - 1
        ip = min(max(ip, 1), 70)
        xt = D[:, ip - 1] + (D[:, ip] - D[:, ip - 1]) / (_GT[ip] - _GT[ip - 1]) * (T - _GT[ip - 1])
        i = int(np.argmax(_GR <= rho))                          # first X(i) <= XP (X decreasing)
        i = min(max(i, 1), 70)
        dp = xt[i - 1] + (xt[i] - xt[i - 1]) / (_GR[i] - _GR[i - 1]) * (rho - _GR[i - 1])
        return 10.0 ** dp


class SemenovDust:
    """Monochromatic Semenov dust: per-region kappa_abs, kappa_tr on a fine log-nu grid, and group means."""

    def __init__(self, sdir, model="h_s", nnu=12000, lnu_lo=np.log(1e8), lnu_hi=np.log(1e17)):
        self.lnu = np.linspace(lnu_lo, lnu_hi, nnu)
        self.nu = np.exp(self.lnu)
        self.kabs, self.ktr = [], []
        folder = os.path.join(sdir, "nrm", FOLDERS[model])
        for r in range(1, 6):
            rows = [[float(t.replace("D", "E")) for t in ln.split()]
                    for ln in open(os.path.join(folder, f"{r}.dat")) if ln.strip()]
            a = np.array(rows)
            nu = C_LIGHT / (a[:, 0] * 1e-4)
            o = np.argsort(nu)
            self.kabs.append(self._extend(nu[o], a[o, 3]))
            self.ktr.append(self._extend(nu[o], a[o, 1]))

    def _extend(self, nu, k):
        """log-log interpolation; below the data a power law from the last two long-wave points, above it constant."""
        ln, lk = np.log(nu), np.log(k)
        out = np.interp(self.lnu, ln, lk)
        lo = self.lnu < ln[0]
        slope = (lk[1] - lk[0]) / (ln[1] - ln[0])
        out[lo] = lk[0] + slope * (self.lnu[lo] - ln[0])
        return np.exp(out)

    def band(self, T, a, b, n=1500):
        """Planck (absorption) and Rosseland (transport) means of every region over [a, b) Hz, on a grid with the band
        edges as end nodes (trapezoid in ln nu)."""
        la = max(np.log(a) if a > 0 else -1e300, self.lnu[0])
        lb = min(np.log(b) if b < 1e299 else 1e300, self.lnu[-1])
        if not lb > la:
            return [(0.0, 0.0)] * 5, 0.0
        l = np.linspace(la, lb, n)
        nu = np.exp(l)
        x = np.minimum(H_K * nu / T, 700.0)
        em = np.exp(-x)
        wB = nu**4 * em / (1.0 - em)
        wD = nu**5 * em / (1.0 - em) ** 2
        sB, sD = np.trapezoid(wB, l), np.trapezoid(wD, l)
        res = []
        for r in range(5):
            ka = np.exp(np.interp(l, self.lnu, np.log(self.kabs[r])))
            kt = np.exp(np.interp(l, self.lnu, np.log(self.ktr[r])))
            kP = np.trapezoid(wB * ka, l) / sB if sB > 0 else 0.0
            kR = sD / np.trapezoid(wD / kt, l) if sD > 0 else 0.0
            res.append((float(kP), float(kR)))
        return res, float(sB)

    def means(self, T, edges):
        """[r][0] = full-spectrum (kP, kR), [r][1 + g] = group g, for regions r = 0..4 at temperature T."""
        ng = len(edges) - 1
        full, _ = self.band(T, 0.0, 1e300, n=6000)
        grp = [self.band(T, edges[g], edges[g + 1])[0] for g in range(ng)]
        return [[full[r]] + [grp[g][r] for g in range(ng)] for r in range(5)]
