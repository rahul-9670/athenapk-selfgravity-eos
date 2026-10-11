#!/usr/bin/env python
"""Figures for RADIATION_VALIDATION.md (results/figures/*.png), each drawn from run outputs / saved references:
F1 R5a envelope temperature by group partition (300 K core); F2 R8b M0 = 2 shock vs the diffusion reference;
F3 R8a plane source at t = 1; F4 R7-O5 table v11 vs Semenov et al. 2003 at 1e-13 g/cm^3."""
import glob
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rad_common import MOD, TABLE_V11  # noqa: E402

FIG = os.path.join(MOD, "results", "figures")
R = os.path.join(MOD, "runs")
os.makedirs(FIG, exist_ok=True)


def f1():
    from analyze_R5 import dump
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a, lab in (("G1", "gray"), ("P3", "P3 production {1e12,1e15}"), ("E4", "4 groups {1e12,1e13,1e14}"),
                   ("E8", "8 groups (0.5 dex)"), ("F10", "10 groups (0.25 dex)"), ("F18", "18 groups (0.125 dex, ref)")):
        D = dump(os.path.join(R, "R5", f"c_{a}_T300"), -1)
        s = D["X"] > 0
        o = np.argsort(D["X"][s])
        ax.plot(D["X"][s][o], D["T"][s][o], label=lab, lw=1.6 if a in ("P3", "F18") else 1.0)
    ax.set_xlim(2, 20)
    ax.set_yscale("log")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("T_gas [K]")
    ax.set_title("R5a: 300 K core heating a 10 K envelope (t = 1.2e-5 code)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F1_R5_groups.png"), dpi=130)


def f2():
    from radshock_ref import Shock
    from make_decks_R8 import RS_C, RS_KAP, RS_P0
    S = Shock(2.0, RS_P0, RS_C, RS_KAP, full=False).solve()
    xg = np.linspace(-0.012, 0.006, 1500)
    Tg, thg, _ = S.profile(xg)
    with h5py.File(os.path.join(R, "R8", "shock_M2", "parthenon.out0.final.phdf"), "r") as h:
        a = float(dict(h["Params"].attrs)["radiation/arad"])
        x = np.array(h["Locations/x"])[0]
        xc = 0.5 * (x[1:] + x[:-1])
        p = np.array(h["prim"])[0, :, 0, 0]
        E = np.array(h["rad.Er"])[0, 0, 0]
    rg = xc > 0
    js = int(np.argmax(np.abs(np.diff(p[0][rg]))))
    xs = 0.5 * (xc[rg][js] + xc[rg][js + 1])
    xi = xs - xc[rg]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xg, Tg / S.T0, "k-", lw=1, label="reference T (non-equilibrium diffusion)")
    ax.plot(xg, thg / S.T0, "k--", lw=1, label="reference T_rad")
    ax.plot(xi, (p[4] / p[0])[rg] / 0.6, "C0.", ms=2, label="code T_gas")
    ax.plot(xi, (np.maximum(E, 0)[rg] / a) ** 0.25 / 0.6, "C1.", ms=2, label="code T_rad")
    ax.set_xlim(-0.012, 0.006)
    ax.set_xlabel("x - x_shock (code)")
    ax.set_ylabel("T / T0")
    ax.set_title("R8b: Mach 2 radiative shock, t = 0.02")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F2_R8_shock.png"), dpi=130)


def f3():
    from analyze_R8 import load, reference
    from ganapol_ref import sn_plane
    from make_decks_R8 import W
    from su_olson_ref import code_slab
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xc, ph, dx = load(os.path.join(R, "R8", "plane_o4x8_dt2e-03_sou"), 1.0)
    ax.plot(xc, reference(xc, 1.0), "k-", lw=1.4, label="Ganapol (continuum, smoothed)")
    mu, wh = code_slab(4, 8)
    ax.plot(xc, sn_plane(mu, wh, xc, dx, [1.0], W)[0], color="0.6", lw=0.8, label="exact S_N, code 4x8 set")
    ax.plot(xc, ph, "C0", lw=1, label="code 4x8 SOU")
    xc2, ph2, _ = load(os.path.join(R, "R8", "plane_o16x32_dt2e-03_sou"), 1.0)
    ax.plot(xc2, ph2, "C3", lw=1, label="code 16x32 SOU")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xlabel("x (mean free paths)")
    ax.set_ylabel("scalar flux")
    ax.set_title("R8a: plane source in a pure scatterer, t = 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F3_R8_plane.png"), dpi=130)


def f4():
    from r7_o5_semenov import semenov, table_at
    from r7_v11 import read
    tab = read(TABLE_V11)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, ls in ((("nrm", "h", "s"), "-"), (("nrm", "p", "s"), ":")):
        T, sR = semenov(m, True, 1e-13)
        _, sP = semenov(m, False, 1e-13)
        ax.loglog(T, sR, "C0" + ls, lw=1, label=f"Semenov {'/'.join(m)} kappa_R")
        ax.loglog(T, sP, "C1" + ls, lw=1, label=f"Semenov {'/'.join(m)} kappa_P")
    tP, tR = table_at(tab, 1e-13, T)
    ax.loglog(T, tR, "C0", lw=2.2, alpha=0.5, label="table v11 kappa_R (= BL94)")
    ax.loglog(T, tP, "C1", lw=2.2, alpha=0.5, label="table v11 kappa_P")
    ax.set_ylim(1e-6, 30)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("kappa [cm^2/g of gas]")
    ax.set_title("R7-O5: mean opacities at 1e-13 g/cm^3")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F4_R7_semenov.png"), dpi=130)


if __name__ == "__main__":
    for f in (f1, f2, f3, f4):
        f()
        print(f.__name__, "done")
