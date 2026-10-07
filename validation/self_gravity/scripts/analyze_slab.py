#!/usr/bin/env python
"""SG-D4 advected self-gravitating slab: artemis-style L1 errors after one period.

L1_q = sum_cells vol |U_q - U_q*| / V_box for the conserved (rho, m1, m2, m3, E); RMS-L1 =
sqrt(sum L1_q^2) (artemis src/pgen/grav_slab.hpp UserWorkAfterLoop). After one period the exact
state equals the initial condition. Golden artemis numbers (tst/scripts/self_gravity/grav_slab.py):
  Nx=16: RMS 1.850706e-01, d 7.195540e-02, E 1.516752e-01
  Nx=32: RMS 4.414474e-02, d 1.904644e-02, E 3.310753e-02
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import sgv  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")
ARTEMIS = {16: (1.850706e-01, 7.195540e-02, 1.516752e-01), 32: (4.414474e-02, 1.904644e-02, 3.310753e-02)}
GAM = 1.66666666667


def exact(x, y, z):
    eps, kx, ky, kz = 0.3, 1 / 3, 2 / 3, 2 / 3
    k2 = kx * kx + ky * ky + kz * kz
    kr = kx * x + ky * y + kz * z
    c1, c2, c3, c4 = np.cos(kr), np.cos(2 * kr), np.cos(3 * kr), np.cos(4 * kr)
    rho = 1.0 + eps * c1 + eps * eps / 3 * c2
    p = 6.0 + eps / k2 * ((1 - eps * eps / 12) * c1 + eps / 3 * c2 + eps * eps / 12 * c3 + eps ** 3 / 144 * c4)
    vx, vy = 0.8, 0.6
    return [rho, rho * vx, rho * vy, 0 * rho, p / (GAM - 1) + 0.5 * rho * (vx * vx + vy * vy)]


def analyse(rd):
    f = sgv.final_dump(rd)
    d = sgv.load(f, ["prim"])
    r, v1, v2, v3, p = (d["prim"][i] for i in range(5))
    U = [r, r * v1, r * v2, r * v3, p / (GAM - 1) + 0.5 * r * (v1 * v1 + v2 * v2 + v3 * v3)]
    Ux = exact(d["x"], d["y"], d["z"])
    V = d["vol"].sum()
    l1 = [float(np.sum(d["vol"] * np.abs(a - b)) / V) for a, b in zip(U, Ux)]
    nx = int(round((6 * np.pi) / d["dx"][0]))
    out = {"run": os.path.relpath(rd, RUNS), "nx": nx, "t": d["time"], "rms_l1": float(np.sqrt(sum(e * e for e in l1))),
           "L1_d": l1[0], "L1_M1": l1[1], "L1_M2": l1[2], "L1_M3": l1[3], "L1_E": l1[4]}
    if nx in ARTEMIS:
        a = ARTEMIS[nx]
        out["ratio_to_artemis_rms_d_E"] = [out["rms_l1"] / a[0], l1[0] / a[1], l1[4] / a[2]]
    H = sgv.hst(glob.glob(os.path.join(rd, "*.hst"))[0])
    ok = H["grav-iters"] >= 0
    out.update(mom_x_drift=float(H["1-mom"][-1] - H["1-mom"][0]), mass_rel_drift=float(H["mass"][-1] / H["mass"][0] - 1),
               grav_iters_max=int(H["grav-iters"][ok].max()), grav_nonconv=int(H["grav-nonconv"][ok].sum()))
    return out


def main():
    rows = []
    for rd in sorted(glob.glob(os.path.join(RUNS, "D4_slab", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        f = sgv.final_dump(rd)
        if f and "final" in f:
            rows.append(analyse(rd))
            print(json.dumps({k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in rows[-1].items()}))
    json.dump(rows, open(os.path.join(OUT, "tables", "slab.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
