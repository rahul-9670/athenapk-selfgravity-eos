#!/usr/bin/env python
"""R7-O5: the production opacity table (v11) against an INDEPENDENT one, Semenov et al. 2003 (A&A 410, 611).

Source: the authors' own Fortran code `opacity.f` v1.0 (16/10/2002) with its Helling (2001) gas tables, downloaded
2026-10-08 from https://www2.mpia-hd.mpg.de/~semenov/Opacities/Code/opacity.zip (md5 6380ff3d..., kept untouched in
literature/semenov2003_download/). It is compiled here (gfortran -std=legacy) into runs/R7_O5/ and run per (dust model,
mean, rho) on 201 log-spaced T in [10, 9999] K (its stated domain: 5 K < T < ~1e4 K, 2e-18 < rho < 2e-7 g/cm^3).
Models: nrm = "normal" silicates Fe/(Fe+Mg) = 0.3; topology h/c/p = homogeneous / composite / porous composite; shape
s/a = spheres / aggregates. Semenov's kappa is per gram of gas (dust-to-gas 0.01 built in), like BL94 and the table.
Comparison: table v11 gray kappa_P, kappa_R at the same (rho, T) with the runtime's log-kappa interpolation in log T
(rho is put on a table node), and kappa_P/kappa_R. Ratios table/Semenov per Semenov temperature region (the code's own
five dust regions + gas). No dust package (kdust = 1).
usage: r7_o5_semenov.py run | check   -> results/tables/R7_O5_semenov.txt
"""
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r7_v11 import read  # noqa: E402
from rad_common import MOD, TABLE_V11  # noqa: E402

SRC = os.path.join(os.environ.get("SEMENOV_DIR", os.path.join(MOD, "literature", "semenov2003_download")), "opacity")  # not shipped
RUN = os.path.join(MOD, "runs", "R7_O5")
OUT = os.path.join(MOD, "results", "tables", "R7_O5_semenov.txt")
MODELS = (("nrm", "h", "s"), ("nrm", "c", "s"), ("nrm", "p", "s"), ("nrm", "h", "a"))
RHOS = (1e-16, 1e-13, 1e-11, 1e-9)
NT, T0, T1 = 201, 10.0, 9999.0
BANDS = ((10, 100), (100, 275), (275, 425), (425, 680), (680, 1500), (1500, 3000), (3000, 9999))


def semenov(model, ross, rho):
    d = os.path.join(RUN, f"{''.join(model)}_{'R' if ross else 'P'}_{rho:.0e}")
    os.makedirs(d, exist_ok=True)
    for f in ("kP_h2001.dat", "kR_h2001.dat"):
        shutil.copy(os.path.join(SRC, f), d)
    out = os.path.join(d, "kR.out" if ross else "kP.out")
    if os.path.exists(out):
        os.remove(out)
    open(os.path.join(d, "opacity.inp"), "w").write(
        f"{model[0]}\n{model[1]}\n{model[2]}\n{'.true.' if ross else '.false.'}\n   {rho:.3E}\n{NT:5d}\n"
        f"   {T0:.3E}\n   {T1:.3E}\n")
    r = subprocess.run([os.path.join(RUN, "opacity")], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    a = np.loadtxt(out, skiprows=1)
    return a[:, 0], a[:, 1]


def table_at(tab, rho, T):
    lr, lT, kP, kR, _ = tab
    ir = int(round((np.log10(rho) - lr[0]) / (lr[1] - lr[0])))
    assert abs(lr[ir] - np.log10(rho)) < 1e-9, "rho must be a table node"
    x = np.log10(T)
    return (10 ** np.interp(x, lT, np.log10(kP[ir])), 10 ** np.interp(x, lT, np.log10(kR[ir])))


def main(mode):
    os.makedirs(RUN, exist_ok=True)
    exe = os.path.join(RUN, "opacity")
    if not os.path.exists(exe):
        subprocess.run([os.environ.get("FC", "gfortran"), "-O1", "-std=legacy", "-w", os.path.join(SRC, "opacity.f"), "-o", exe], check=True)
    tab = read(TABLE_V11)
    L = ["# R7-O5: table v11 (gray, kdust = 1) vs Semenov et al. 2003 opacity.f v1.0 (scripts/r7_o5_semenov.py)",
         "# ratio = table / Semenov; per T band: median [min, max]; Semenov regions: <~100 ice, 100-275 no ice, 275-425 "
         "+ organics gone, 425-680 troilite gone, 680-~1500 silicates+iron, >~1500 gas (Helling 2001)"]
    for rho in RHOS:
        for m in MODELS:
            T, sR = semenov(m, True, rho)
            T2, sP = semenov(m, False, rho)
            assert np.allclose(T, T2)
            tP, tR = table_at(tab, rho, T)
            L.append(f"rho {rho:.0e} g/cm^3, model {'/'.join(m)}:")
            for nm, a in (("kappa_R", tR / sR), ("kappa_P", tP / sP), ("kP/kR", (tP / tR) / (sP / sR))):
                cells = []
                for lo, hi in BANDS:
                    s = (T >= lo) & (T < hi)
                    cells.append(f"{lo}-{hi} K {np.median(a[s]):6.3f} [{a[s].min():6.3f}, {a[s].max():6.3f}]")
                L.append(f"  {nm:7s} " + " | ".join(cells))
            if m == MODELS[0]:
                for Tq in (10.0, 30.0, 100.0, 300.0, 1000.0, 2000.0, 5000.0):
                    i = int(np.argmin(np.abs(T - Tq)))
                    L.append(f"    T {T[i]:8.2f} K: Semenov kR {sR[i]:.3e} kP {sP[i]:.3e} | table kR {tR[i]:.3e} "
                             f"kP {tP[i]:.3e} cm^2/g")
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
