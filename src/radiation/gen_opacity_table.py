#!/usr/bin/env python3
"""Generate a tabulated multigroup opacity table for the AthenaPK M1 radiation package.

STATE-OF-THE-ART opacity architecture: the frequency-resolved dust+gas physics lives HERE
(offline, documented, swappable), and the C++ side only interpolates 2D (log rho, log T) tables.
Drop in a real Semenov (2003) / DSHARP / Draine monochromatic dataset by replacing `kappa_nu`.

What this produces, per (log10 rho_cgs, log10 T_K) grid point, in CODE units:
  - gray  kappa_P(rho,T), kappa_R(rho,T)          (Planck- and Rosseland-mean absorption)
  - per-group kappa_P,g(rho,T), kappa_R,g(rho,T)  (g = 0 .. n_group-1)
  - scattering kappa_s(rho,T)
all obtained by integrating a monochromatic kappa_nu = kappa_BL(rho,T) * psihat(nu,T) over the
Planck weight B_nu (emission/absorption) and the Rosseland weight dB_nu/dT (flux), per group and
over the full band.

Physics / honesty:
  * MAGNITUDE is anchored to Bell & Lin (1994) so the GRAY ROSSELAND reproduces the trusted law
    exactly (psihat is normalized so <psihat>_Rosseland,full = 1 => kappa_R,gray == kappa_BL).
    Bell & Lin's cold kappa ~ 2e-4 T^2 IS the Rosseland mean of a beta=2 dust, so this is
    physically self-consistent, not a fudge.
  * The frequency SHAPE psi(nu,T) is a multi-component modified-blackbody dust emissivity
    (nu^beta far-IR rise saturating to the geometric/gray limit above a grain-size break) with a
    MULTI-SPECIES SUBLIMATION CASCADE (ice mantle ~150 K, refractory silicate/carbon ~1500 K):
    above each front the corresponding opacity is removed and the shape blends toward gray gas.
  * NEW vs the analytic model it replaces: kappa_P is the TRUE Planck mean (not planck_ross_ratio
    * kappa_R), and the per-group means come from real frequency integration -- both tabulated.

This is a physically-motivated multi-component model, NOT literal Mie theory on measured optical
constants; the ARCHITECTURE accepts such a dataset unchanged (swap `kappa_nu`).
"""
import numpy as np
import struct
import argparse

# --- CGS constants ---
h_P = 6.62607015e-27      # erg s
k_B = 1.380649e-16        # erg/K
h_over_k = h_P / k_B      # s K  (= 4.7992e-11)

# --- table file format (see opacity_table_format.hpp; both must agree) ---
MAGIC = 0x4F50414354424C31      # ASCII "OPACTBL1"; cannot collide with a legacy file, whose
                                # first int64 is ng (a small positive group count)
VERSION = 3   # AUDIT F-24: v3 appends the ng+1 group frequency edges [Hz]
FLAG_KAPPA_CGS = 1              # bit 0: gray kappa arrays are cgs, scale by opacity_unit at load
#
# AUDIT N13 (2026-08-05). This script used to hardcode
#     RHO_UNIT = 5.467e-19 ; LEN_UNIT = 2.81e16 ; OPACITY_UNIT = RHO_UNIT*LEN_UNIT
# and write kappa ALREADY MULTIPLIED by OPACITY_UNIT. Those are ROUNDED copies of scales that
# the run derives exactly (units/be_normalization.hpp), so every tabulated kappa came out
# +0.135 % too large -- a bias no C++ guard could see, because the constants lived here.
#
# Version 2 writes kappa in **cgs** and lets the loader multiply by the RUNNING simulation's
# opacity_unit. The table is then correct for every normalization, and this script no longer
# needs to know the code units at all. Do not reintroduce them.


# ---------------------------------------------------------------------------------------
# GRAIN EVAPORATION MODEL (2026-09-22). "bl94" = Bell & Lin (1994) exactly, the historical table.
# BL94 regime 4 (metal-grain evaporation, kappa = 2e81 rho T^-24) meets the metal-grain branch
# 0.1 T^0.5 where T^24.5 = 2e82 rho, i.e. T_ev = 810 K at rho = 1e-11 and 420 K at 1e-18. BL94 is
# calibrated at accretion-disk densities and its T_ev scales only as rho^(1/24.5), so at FIRST-CORE
# densities it evaporates the dust ~400-800 K below laboratory silicate evaporation -- and below this
# script's own sublimation cascade (refractory survives to 1400-1600 K), which only reshapes the
# frequency dependence because the magnitude is anchored to BL94. The table therefore lost its dust
# opacity between ~700 and ~1400 K while claiming the dust was still there.
# "pollack" keeps BL94 everywhere else but moves the evaporation onset to the lab-calibrated
#     T_ev(rho) = 2000 K * (rho / g cm^-3)^0.0195      (Isella & Natta 2005, from Pollack+1994),
# by choosing regime-4 coefficients so that it meets 0.1 T^0.5 exactly at T_ev(rho):
#     kappa_4 = 0.1 * 2000^24.5 * rho^(0.0195*24.5) * T^-24.
# T_ev = 1221 K at rho = 1e-11, 1167 K at 1e-12, 891 K at 1e-18. Below T_ev nothing changes.
EVAP = "bl94"
EVAP_T0, EVAP_PWR = 2000.0, 0.0195
# R-F4 (validation/radiation_sn, 2026-10-08): "simpson" = the historical 400-point Simpson rule on [xa, xb]
# (bit-identical tables), "accurate" = adaptive quadrature with the grain break and the integrable x -> 0 behaviour
# handled (scipy.integrate.quad, breakpoint at x_break, epsrel 1e-11). The Simpson rule cannot integrate the x^beta
# shape near x = 0 (Rosseland harmonic mean of x^-1.8 x^4 ...): kappa_P was 0.75 % low at every node below ~140 K.
QUAD = "simpson"
# R-F9 (validation/radiation_sn, 2026-10-08): the SHAPE model. "cascade" = historical (bit-identical tables):
# psi = S dust + (1 - S), S = surviving-dust fraction of a two-species sublimation cascade (ice 140-170 K, refractory
# 1400-1600 K). Its (1 - S) term is a GRAY continuum at the dust's own (geometric-limit) amplitude, so losing the ice
# mantles REPLACES 35 % of the dust opacity by gray opacity: kappa_P/kappa_R drops 1.871 -> 1.741 within 0.01 K above
# 140 K and stays ~0.96-0.98 to 1400 K. Physically the ice loss lowers the MAGNITUDE (BL94 regimes 1-3 already do
# that, and the magnitude is anchored to BL94), and the gas continuum is the GAS opacity, orders of magnitude weaker.
# "gasfrac": psi = (1 - f_gas) d(x)/<d>_R + f_gas, f_gas = kappa_gas/kappa_BL with kappa_gas the BL94 gas branch
# (regimes >= 4: molecules, H-, bound-free/free-free, electron scattering) -- the SAME split the C++ dust factor uses
# (the research code's dust package, DustOpacityWeight[Evap], w = 1 - f_gas), so table and dust factor agree on how much opacity is dust.
SHAPE = "cascade"


def evap_T(rho):
    """Lab-calibrated refractory evaporation temperature [K] (Isella & Natta 2005)."""
    return EVAP_T0 * rho**EVAP_PWR


def bell_lin_kappa(rho, T, istart=0):
    """Bell & Lin (1994) Rosseland-mean gray opacity [cm^2/g], rho,T in cgs. Regime-skip-robust
    walk (matches C++ BellLinKappaFixed). 8 piecewise power laws kappa = k0 rho^a T^b.
    istart = 4 walks the GAS branch only (as C++ BellLinKappaFrom(4)): the dust/gas split of F-07 / R-F8."""
    k0 = np.array([2.0e-4, 2.0e16, 0.1, 2.0e81, 1.0e-8, 1.0e-36, 1.5e20, 0.348])
    aa = np.array([0.0, 0.0, 0.0, 1.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0])
    if EVAP == "pollack":
        k0[3] = 0.1 * EVAP_T0**24.5
        aa[3] = EVAP_PWR * 24.5
    bb = np.array([2.0, -7.0, 0.5, -24.0, 3.0, 10.0, -2.5, 0.0])

    def Tcross(p, q):
        return ((k0[p] / k0[q]) * rho**(aa[p] - aa[q]))**(1.0 / (bb[q] - bb[p]))
    i = istart
    while i < 7:
        j = i + 1
        while j < 7 and Tcross(j, j + 1) <= Tcross(i, j):
            j += 1
        if T < Tcross(i, j):
            return k0[i] * rho**aa[i] * T**bb[i]
        i = j
    return k0[7] * rho**aa[7] * T**bb[7]


# ---------------------------------------------------------------------------------------
# Dust frequency-shape parameters (documented, swappable).
BETA = 1.8               # far-IR dust emissivity index (silicate/carbon aggregates; 1.5-2)
NU_BREAK = 1.0e14        # grain-size turnover [Hz] (lambda ~ few micron); flat (gray) above
# multi-species sublimation cascade: (fraction of dust opacity, T_sub_lo, T_sub_hi) [K]
DUST_SPECIES = [
    (0.35, 140.0, 170.0),   # volatile ice mantles (H2O/CO2) -> far-IR emissivity carriers
    (0.65, 1400.0, 1600.0), # refractory silicate + carbon grains
]


def dust_survival(T):
    """Surviving fraction of the dust far-IR opacity vs T (multi-species sublimation cascade)."""
    S = 0.0
    species = DUST_SPECIES
    if EVAP == "pollack":
        # The per-group multipliers are rho-INDEPENDENT (tabulated in T at rho_ref = 1e-12), so the
        # SHAPE cannot follow T_ev(rho). Centre the refractory ramp on T_ev(rho_ref) = 1167 K; the
        # MAGNITUDE (bell_lin_kappa) carries the full rho dependence. They agree to ~+-100 K over
        # the collapse run's density range instead of disagreeing by ~600 K.
        te = evap_T(1.0e-12)
        species = [DUST_SPECIES[0], (DUST_SPECIES[1][0], te - 100.0, te + 100.0)]
    for frac, Tlo, Thi in species:
        if T <= Tlo:
            S += frac
        elif T < Thi:
            S += frac * (Thi - T) / (Thi - Tlo)
        # else 0
    return S


def gas_fraction(rho, T):
    """R-F9: f_gas = kappa_gas / kappa_BL in [0, 1] (BL94 gas branch over the full law, EVAP-aware)."""
    return min(max(bell_lin_kappa(rho, T, 4) / bell_lin_kappa(rho, T), 0.0), 1.0)


_DN_CACHE = {}


def dust_norm(T):
    """R-F9: full-band harmonic Rosseland mean of the raw dust shape (so d/<d>_R has unit Rosseland mean)."""
    if T not in _DN_CACHE:
        _DN_CACHE[T] = band_mean(T, 1e-8, 80.0, ross=1, harmonic=True, fg=-1.0)
    return _DN_CACHE[T]


_GL = {}


def gray_means_gasfrac(T, fgs, n=400):
    """R-F9, vectorised over an ARRAY of gas fractions at one T: (<psi>_P,full, <psi>_R,full,harmonic) of
    psi = (1 - fg) d/<d>_R + fg. Planck: exactly linear in fg, from the cached quad means. Rosseland: harmonic, by
    Gauss-Legendre in u = ln x on [ln 1e-8, ln x_break] + [ln x_break, ln 80] (smooth there), taken as the RATIO to the
    same quadrature at fg = 0 -- whose harmonic mean is 1 by construction of <d>_R -- so fg = 0 is exact and the
    quadrature error cancels to first order (checked against scipy quad by check_gasfrac_quadrature())."""
    fgs = np.asarray(fgs, dtype=float)
    D = dust_norm(T)
    dP = band_mean(T, 1e-8, 80.0, ross=0, fg=-1.0)
    psiP = (1.0 - fgs) * dP / D + fgs
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    t, wt = _GL[n]
    xb = h_over_k * NU_BREAK / T
    cuts = [np.log(1e-8)] + ([np.log(xb)] if 1e-8 < xb < 80.0 else []) + [np.log(80.0)]
    us, ws = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        us.append(0.5 * (b - a) * t + 0.5 * (b + a)); ws.append(0.5 * (b - a) * wt)
    u = np.concatenate(us); w = np.concatenate(ws); x = np.exp(u)
    wr = spectral_weight(x, 1e-8, 1) * x * w                 # Rosseland weight dx = x du
    d = np.minimum((x / xb)**BETA, 1.0) / D
    ps = (1.0 - fgs[:, None]) * d[None, :] + fgs[:, None]
    h = (wr[None, :] / ps).sum(axis=1)
    h0 = (wr / d).sum()
    return psiP, h0 / h


def check_gasfrac_quadrature():
    """R-F9: vectorised gray means vs scipy quad (band_mean with QUAD accurate) -- max relative difference."""
    global QUAD
    q0, QUAD = QUAD, "accurate"
    worst = 0.0
    for T in (3.0, 30.0, 140.0, 300.0, 1000.0, 1500.0, 3000.0, 1.0e4, 1.0e5):
        fgs = np.array([0.0, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-6, 1.0])
        P, R = gray_means_gasfrac(T, fgs)
        for f, pp, rr in zip(fgs, P, R):
            rq = band_mean(T, 1e-8, 80.0, ross=1, harmonic=True, fg=float(f))
            pq = band_mean(T, 1e-8, 80.0, ross=0, fg=float(f))
            worst = max(worst, abs(rr / rq - 1.0), abs(pp / pq - 1.0))
    QUAD = q0
    return worst


def psi(x, T, fg=None):
    """Monochromatic opacity SHAPE at x = h nu / (k T): dust nu^beta rise saturating to the
    geometric (gray) limit above the grain break, blended toward gray gas as dust sublimates.
    Returns an UN-normalized shape; the caller divides by its Rosseland-weighted mean.
    fg None: SHAPE "cascade" (historical); fg = -1: the raw dust shape; fg in [0, 1]: SHAPE "gasfrac"."""
    xb = h_over_k * NU_BREAK / T
    dust_shape = np.minimum((x / xb)**BETA, 1.0)
    if fg is None:
        S = dust_survival(T)
        return S * dust_shape + (1.0 - S)   # (1-S) -> gray gas continuum
    if fg < 0.0:
        return dust_shape
    return (1.0 - fg) * dust_shape / dust_norm(T) + fg   # fg -> the BL94 gas continuum (gray)


# --- Planck / Rosseland spectral weights, with the common factor e^{-xa} pulled OUT of the
# ratio: the returned integrand carries e^{xa-x} (<= 1 on [xa,xb]) instead of e^{-x}, so it never
# overflows/underflows even for deep-Wien bands (xa >> 1). Every use divides two of these with the
# same xa, so the e^{xa} cancels exactly. ross=1 => Rosseland weight (x^4 dB/dT), else Planck (x^3 B).
def spectral_weight(x, xa, ross):
    em = np.exp(xa - x)               # e^{xa-x}, always in (0,1]
    d = 1.0 - np.exp(-x)
    return (x**4 * em / (d * d)) if ross else (x**3 * em / d)


_BM_CACHE = {}


def band_mean(T, xa, xb, ross, harmonic=False, nq=400, fg=None):
    """<psi(^-1)>_weight over [xa,xb]. Simpson in x; e^{-xa} factored out of the num/den ratio."""
    xa = max(xa, 1e-8)
    if xb <= xa:
        return 1.0
    if QUAD == "accurate":
        key = (T, xa, xb, ross, harmonic, fg)
        if key not in _BM_CACHE:
            from scipy.integrate import quad
            xbk = h_over_k * NU_BREAK / T
            pts = [p for p in (xbk,) if xa < p < xb] or None
            w = lambda x: float(spectral_weight(np.array([x]), xa, ross)[0])
            sh = lambda x: float(psi(np.array([x]), T, fg)[0])
            f = (lambda x: w(x) / sh(x)) if harmonic else (lambda x: w(x) * sh(x))
            num = quad(f, xa, xb, points=pts, limit=500, epsabs=0, epsrel=1e-11)[0]
            den = quad(w, xa, xb, points=pts, limit=500, epsabs=0, epsrel=1e-11)[0]
            val = num / (den + 1e-300)
            _BM_CACHE[key] = 1.0 / val if harmonic else val
        return _BM_CACHE[key]
    x = np.linspace(xa, xb, nq + 1)
    w = np.ones(nq + 1); w[1:-1:2] = 4; w[2:-1:2] = 2
    base = spectral_weight(x, xa, ross)
    s = psi(x, T, fg)
    if harmonic:
        s = 1.0 / s
    num = np.sum(w * s * base); den = np.sum(w * base)
    val = num / (den + 1e-300)
    return 1.0 / val if harmonic else val


def group_opacities(rho, T, edges):
    """Return (kP_gray, kR_gray, [kP_g...], [kR_g...]) [cm^2/g] for kappa_nu = kBL * psihat."""
    ng = len(edges) - 1
    kBL = bell_lin_kappa(rho, T)
    XFULL = 80.0
    # normalize psi so <psihat>_Rosseland,full = 1  => kappa_R,gray == kBL (anchor to Bell&Lin).
    #
    # AUDIT F-23 (2026-08-28) -- THE NORMALISER MUST BE THE HARMONIC (TRUE ROSSELAND) MEAN.
    #
    # This line read `band_mean(T, 1e-8, XFULL, ross=1)` -- the ARITHMETIC Rosseland-weighted
    # mean -- while the per-group Rosseland six lines below correctly used harmonic=True. The
    # Rosseland mean is by definition harmonic. With kappa_nu = kBL * psi / N:
    #     kappa_R = (kBL/N) <psi>_R,harm   =>  kappa_R == kBL  REQUIRES  N = <psi>_R,harm
    #     kappa_P = (kBL/N) <psi>_P,arith
    # Using N = <psi>_R,arith (> harmonic) therefore made kappa_P TOO SMALL, by a factor that
    # is a constant 2.6923 for every T below ~140 K -- i.e. essentially the whole domain this
    # run occupies (median gas T = 8.44 K, max 192.9 K).
    #
    # Shipped v4 gave kappa_P/kappa_R = 0.6897 at ALL rho and all T < 140 K. That is backwards
    # on physical grounds too: for dust the Rosseland mean is depressed by transparency
    # windows, so kappa_P/kappa_R > 1 is expected. With the fix it is 1.8568.
    #
    # THREE consumers were biased by this ONE line, all in the same direction:
    #   * kappa_P (gray)  -> the emission/absorption matter coupling was 2.69x too weak;
    #   * mR = pR_harm/N  -> the per-group extinction chi feeding the asymptotic-preserving
    #                        flux limiter (F-36) and the flux drag was 2.69x too small;
    #   * rslaMax         -> chi again, so the reported RSLA violation was 2.69x too LOW
    #                        (F-27: the true value is ~50, not 18.8).
    # mP = pP_band/psiP_full is UNAFFECTED -- the error cancels in that ratio.
    #
    # kappa_R (gray) was and remains correct: it is set to kBL by construction, verified
    # against Bell & Lin to 0.05 % at seven (rho,T) points spanning the run's domain.
    # ross=1 Rosseland weight, ross=0 Planck weight.
    fg = gas_fraction(rho, T) if SHAPE == "gasfrac" else None   # R-F9
    psiR_full = band_mean(T, 1e-8, XFULL, ross=1, harmonic=True, fg=fg)  # <psi>_Ross,full (HARMONIC)
    psiP_full = band_mean(T, 1e-8, XFULL, ross=0, fg=fg)             # <psi>_Planck,full
    kR_gray = kBL                                             # by construction
    kP_gray = kBL * (psiP_full / psiR_full)                  # TRUE Planck mean
    kP_g, kR_g = [], []
    for g in range(ng):
        xa = h_over_k * edges[g] / T
        xb = (xa + 60.0) if g == ng - 1 else h_over_k * edges[g + 1] / T
        xb = min(xb, xa + 60.0)
        # per-group Planck mean = kBL * <psi>_{P,band}/psiR_full ; Rosseland = kBL * harmonic mean
        pP = band_mean(T, xa, xb, ross=0, fg=fg)
        pR = band_mean(T, xa, xb, ross=1, harmonic=True, fg=fg)
        kP_g.append(kBL * pP / psiR_full)
        kR_g.append(kBL * pR / psiR_full)
    return kP_gray, kR_gray, kP_g, kR_g


def scattering_kappa(rho, T):
    """Gray electron-scattering opacity [cm^2/g] (Bell&Lin regime 7 floor, ionized gas)."""
    return 0.348 if T > 1.0e4 else 0.0


# ---------------------------------------------------------------------------------------
def build(path, nr, nT, rho_min, rho_max, T_min, T_max, edges, fmt=3):
    ng = len(edges) - 1
    lr = np.linspace(np.log10(rho_min), np.log10(rho_max), nr)   # log10 rho_cgs
    lT = np.linspace(np.log10(T_min), np.log10(T_max), nT)       # log10 T_K
    # GRAY magnitude tables over (rho,T) [code units]: true Planck mean kP, Rosseland kR, scatter ks.
    kP = np.zeros((nr, nT)); kR = np.zeros((nr, nT)); ks = np.zeros((nr, nT))
    # PER-GROUP MULTIPLIERS over (group,T) [dimensionless]: kappa_{P,R},g / kappa_{P,R},gray.
    # rho-INDEPENDENT (the kBL magnitude cancels in the ratio -> shape <psi>_band/<psi>_full),
    # so tabulated in T only. Evaluated at a reference rho (the ratio is rho-free by construction).
    mP = np.zeros((ng, nT)); mR = np.zeros((ng, nT))
    rho_ref = 1.0e-12
    for j in range(nT):
        T = 10.0**lT[j]
        gP, gR, pg, rg = group_opacities(rho_ref, T, edges)
        for g in range(ng):
            mP[g, j] = pg[g] / (gP + 1e-300)   # kappa_P,g / kappa_P,gray
            mR[g, j] = rg[g] / (gR + 1e-300)   # kappa_R,g / kappa_R,gray
    if SHAPE == "gasfrac":
        # R-F9: the gray means depend on rho through f_gas -> vectorised over rho per T (gray_means_gasfrac).
        for j in range(nT):
            T = 10.0**lT[j]
            kBL = np.array([bell_lin_kappa(10.0**lr[i], T) for i in range(nr)])
            fgs = np.array([gas_fraction(10.0**lr[i], T) for i in range(nr)])
            psiP, psiR = gray_means_gasfrac(T, fgs)
            kR[:, j] = kBL
            kP[:, j] = kBL * psiP / psiR
            ks[:, j] = [scattering_kappa(10.0**lr[i], T) for i in range(nr)]
    else:
        for i in range(nr):
            rho = 10.0**lr[i]
            for j in range(nT):
                T = 10.0**lT[j]
                gP, gR, _, _ = group_opacities(rho, T, edges)
                # CGS [cm^2/g] -- NOT code units. The loader applies the run's opacity_unit.
                kP[i, j] = gP
                kR[i, j] = gR
                ks[i, j] = scattering_kappa(rho, T)
    # binary v2: int64 [MAGIC, VERSION, FLAGS, ng, nr, nT];
    #   double [lr0, dlr, lT0, dlT, gen_rho_unit, gen_len_unit, gen_opacity_unit]
    #     (the three gen_* are provenance and are 0 for a cgs table -- it assumes no units)
    #   double edges[ng+1]                    (AUDIT F-24: group frequency edges [Hz])
    #   gray kP[nr,nT], kR[nr,nT], ks[nr,nT]  (row-major float64, CGS cm^2/g)
    #   mult mP[ng,nT], mR[ng,nT]             (row-major float64, dimensionless)
    # 2026-09-01: --format lets a table be written in the OLDER v2 layout (no edges block).
    # WHY THIS IS NEEDED, and it is not a preference: the live production binary
    # build_gpu_halloff (610185ea) supports opacity format version 2 ONLY, and aborts on a v3
    # file with "is format version 3, newer than this build supports (2)" -- measured, it
    # killed job 2694581. The AUDIT F-23 fix (harmonic Rosseland normaliser, ~2.69x on kappa_P
    # below 140 K) is a table-only change and is otherwise adoptable in production TODAY; the
    # only thing blocking it is the v3 container. Writing the corrected numbers in v2 lets the
    # fix ship WITHOUT a binary swap, which would otherwise drag in 30 modified source files.
    # SAFE ONLY WHEN THE EDGES ARE UNCHANGED: v2 carries no edges, so the F-24 validation is
    # absent and the runtime partition matches only by the same coincidence of defaults that
    # F-24 documents. Do NOT write a RE-BINNED table as v2 -- that is exactly the silent
    # mismatch F-24 exists to catch.
    ver = int(fmt)
    if ver not in (2, 3):
        raise SystemExit("--format must be 2 or 3")
    if ver == 2 and list(edges) != [0.0, 1.0e12, 1.0e15, 1.0e30]:
        raise SystemExit(
            "REFUSING to write format 2 with non-default edges %s: v2 stores no edges, so the "
            "loader cannot detect a runtime/table band mismatch (AUDIT F-24). Use --format 3."
            % (list(edges),))
    with open(path, "wb") as f:
        np.array([MAGIC, ver, FLAG_KAPPA_CGS, ng, nr, nT], dtype=np.int64).tofile(f)
        np.array([lr[0], lr[1]-lr[0], lT[0], lT[1]-lT[0], 0.0, 0.0, 0.0],
                 dtype=np.float64).tofile(f)
        # AUDIT F-24: the group frequency edges the multipliers below were BUILT with, so the
        # loader can reject a runtime band partition that disagrees. v2 stored only ng.
        if ver >= 3:
            np.array(edges, dtype=np.float64).tofile(f)
        kP.astype(np.float64).tofile(f)
        kR.astype(np.float64).tofile(f)
        ks.astype(np.float64).tofile(f)
        mP.astype(np.float64).tofile(f)
        mR.astype(np.float64).tofile(f)
    print(f"wrote {path}: format v{ver} (kappa in CGS) ng={ng} grid {nr}x{nT}  "
          f"logrho[{lr[0]:.1f},{lr[-1]:.1f}] logT[{lT[0]:.2f},{lT[-1]:.2f}]")
    # sanity anchors (cgs gray Rosseland vs Bell&Lin; per-group multipliers)
    for T in [10.0, 100.0, 1000.0, 1.0e4]:
        kBL = bell_lin_kappa(1e-12, T)
        gP, gR, pg, rg = group_opacities(1e-12, T, edges)
        mp = [pg[g]/gP for g in range(ng)]
        print(f"  T={T:>7.0f}K: kR(BL)={kBL:.3e} kR_tab={gR:.3e} kP_tab={gP:.3e} "
              f"kP/kR={gP/max(gR,1e-30):.3f}  multP={[f'{m:.3f}' for m in mp]}")


# ---------------------------------------------------------------------------------------
# FORMAT 4 (R-F11, 2026-10-09): --model semenov. Per-group multipliers are rho-DEPENDENT (Semenov's evaporation
# temperatures and the dust -> gas blend depend on rho, so a T-only multiplier is wrong in every transition band),
# and the grains' share of the gray kappa_P / kappa_R is stored (FLAG_DUST_WEIGHT) so dust/opacity_evap = table can
# split the opacity exactly as the table does. Layout after the v3 header + edges:
#   kP, kR, ks [nr, nT];  mP, mR [ng, nr, nT];  wdP, wdR [nr, nT]   (float64, row-major)
FLAG_DUST_WEIGHT = 2
SEMENOV_T_GAS_TOP = 1.0e4     # top of Helling's grid; Bell & Lin above, log-blended over [0.9, 1] x this


def build_semenov(path, nr, nT, rho_min, rho_max, T_min, T_max, edges, sdir, dust):
    import opacity_semenov as SM
    ng = len(edges) - 1
    lr = np.linspace(np.log10(rho_min), np.log10(rho_max), nr)
    lT = np.linspace(np.log10(T_min), np.log10(T_max), nT)
    D = SM.SemenovDust(sdir, dust)
    H = SM.Helling(sdir)
    Tdust_max = float(SM.evap_temperatures(10.0**lr[-1])[4]) + 100.0   # no dust above this anywhere
    # dust region means per T node: dm[j, r, b, 0/1] (b = 0 full, 1 + g group), P = absorption, R = transport
    dm = np.zeros((nT, 5, ng + 1, 2))
    for j in range(nT):
        T = 10.0**lT[j]
        if T <= Tdust_max:
            dm[j] = np.array(D.means(T, edges))
    kP = np.zeros((nr, nT)); kR = np.zeros((nr, nT)); ks = np.zeros((nr, nT))
    mP = np.ones((ng, nr, nT)); mR = np.ones((ng, nr, nT))
    wdP = np.zeros((nr, nT)); wdR = np.zeros((nr, nT))
    for i in range(nr):
        rho = 10.0**lr[i]
        for j in range(nT):
            T = 10.0**lT[j]
            w = SM.region_weights(rho, T) if T <= Tdust_max else np.array([0, 0, 0, 0, 0, 1.0])
            wd = w[:5]
            dP = float(wd @ dm[j, :, 0, 0]); dR = float(wd @ dm[j, :, 0, 1])
            gP = gR = 0.0
            if w[5] > 0.0:
                if T <= SEMENOV_T_GAS_TOP:
                    gP, gR = H.kappa("P", rho, T), H.kappa("R", rho, T)
                    if T > 0.9 * SEMENOV_T_GAS_TOP:       # log-blend into Bell & Lin over [9e3, 1e4] K
                        q = (np.log(T) - np.log(0.9 * SEMENOV_T_GAS_TOP)) / (-np.log(0.9))
                        kb = bell_lin_kappa(rho, T)
                        gP = float(np.exp((1 - q) * np.log(gP) + q * np.log(kb)))
                        gR = float(np.exp((1 - q) * np.log(gR) + q * np.log(kb)))
                else:
                    gP = gR = bell_lin_kappa(rho, T)
            kP[i, j] = dP + w[5] * gP
            kR[i, j] = dR + w[5] * gR
            ks[i, j] = scattering_kappa(rho, T)
            wdP[i, j] = dP / kP[i, j] if kP[i, j] > 0 else 0.0
            wdR[i, j] = dR / kR[i, j] if kR[i, j] > 0 else 0.0
            for g in range(ng):
                mP[g, i, j] = (float(wd @ dm[j, :, 1 + g, 0]) + w[5] * gP) / kP[i, j]
                mR[g, i, j] = (float(wd @ dm[j, :, 1 + g, 1]) + w[5] * gR) / kR[i, j]
    with open(path, "wb") as f:
        np.array([MAGIC, 4, FLAG_KAPPA_CGS | FLAG_DUST_WEIGHT, ng, nr, nT], dtype=np.int64).tofile(f)
        np.array([lr[0], lr[1] - lr[0], lT[0], lT[1] - lT[0], 0.0, 0.0, 0.0], dtype=np.float64).tofile(f)
        np.array(edges, dtype=np.float64).tofile(f)
        for a in (kP, kR, ks, mP, mR, wdP, wdR):
            a.astype(np.float64).tofile(f)
    print(f"wrote {path}: format v4 (Semenov 2003 nrm_{dust} + Helling 2001 gas, rho-dependent multipliers, dust "
          f"weights) ng={ng} grid {nr}x{nT} logrho[{lr[0]:.1f},{lr[-1]:.1f}] logT[{lT[0]:.3f},{lT[-1]:.3f}]")
    for rho, T in ((1e-13, 10.0), (1e-13, 100.0), (1e-11, 300.0), (1e-11, 1000.0), (1e-11, 2000.0)):
        ii = int(round((np.log10(rho) - lr[0]) / (lr[1] - lr[0]))); jj = int(round((np.log10(T) - lT[0]) / (lT[1] - lT[0])))
        print(f"  rho={10**lr[ii]:.1e} T={10**lT[jj]:8.2f} K: kP={kP[ii, jj]:.3e} kR={kR[ii, jj]:.3e} "
              f"wdP={wdP[ii, jj]:.3f} mP={[f'{mP[g, ii, jj]:.3f}' for g in range(ng)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="opacity_table.bin")
    ap.add_argument("--nr", type=int, default=60)
    ap.add_argument("--nT", type=int, default=120)
    ap.add_argument("--rho_min", type=float, default=1e-20)
    ap.add_argument("--rho_max", type=float, default=1e-2)
    ap.add_argument("--T_min", type=float, default=3.0)
    ap.add_argument("--T_max", type=float, default=1e5)
    # default 3-group FHC structure (edges in Hz; 0 and +inf on the ends)
    ap.add_argument("--format", default=3, type=int,
                    help="binary layout version: 3 (default, stores group edges) or 2 "
                         "(legacy, no edges -- only for the DEFAULT edges, see build())")
    ap.add_argument("--edges", default="0,1e12,1e15,1e30")
    ap.add_argument("--quad", default="simpson", choices=["simpson", "accurate"],
                    help="band-mean quadrature: simpson (historical, bit-identical) or accurate (R-F4)")
    ap.add_argument("--evap", default="bl94", choices=["bl94", "pollack"],
                    help="grain-evaporation model: bl94 (historical, reproduces v7_f23 bit-for-bit) "
                         "or pollack (lab-calibrated T_ev(rho) = 2000 rho^0.0195 K)")
    ap.add_argument("--model", default="bl94", choices=["bl94", "semenov"],
                    help="bl94 (historical, formats 2/3) or semenov (R-F11: Semenov 2003 dust + Helling gas, format 4)")
    ap.add_argument("--semenov_dir", default="semenov2003_download", help="Semenov's download: opacity/ (code + Helling gas) and nrm/ (kappa_lambda)")
    ap.add_argument("--semenov_dust", default="h_s", help="topology_shape of the nrm model (h_s, h_a, c_s, c_a, ...)")
    ap.add_argument("--shape", default="cascade", choices=["cascade", "gasfrac"],
                    help="spectral shape: cascade (historical, bit-identical) or gasfrac (R-F9: gray part = the BL94 "
                         "gas fraction of the opacity; no greying at ice sublimation)")
    args = ap.parse_args()
    EVAP = args.evap          # module-level: read by bell_lin_kappa and dust_survival
    QUAD = args.quad          # module-level: read by band_mean
    SHAPE = args.shape        # module-level: read by group_opacities (R-F9)
    edges = [float(x) for x in args.edges.split(",")]
    if args.model == "semenov":
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        build_semenov(args.out, args.nr, args.nT, args.rho_min, args.rho_max, args.T_min, args.T_max, edges,
                      args.semenov_dir, args.semenov_dust)
        raise SystemExit(0)
    build(args.out, args.nr, args.nT, args.rho_min, args.rho_max,
          args.T_min, args.T_max, edges, fmt=args.format)
