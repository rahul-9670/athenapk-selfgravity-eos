#!/usr/bin/env python
"""SG-D5 static n=1 polytrope, SG-D6 polytrope collision, SG-D7 pressure-free collapse.

D5: rho_c(t)/rho_c(0), max |v| inside the star, L1 density error inside R vs the Lane-Emden IC,
    mass/energy from the history file. Lane-Emden: rho = sin(xi)/xi, xi = r/sqrt(2), R = 0.75 pi sqrt(2).
D6: total momentum, centre of mass and mass vs time (exact: P = 0, COM fixed, M constant).
D7: rho_max(t) vs the exact cycloid  rho/rho0 = sec^6(beta), t/t_ff = (2/pi)(beta + sin b cos b),
    t_ff = pi sqrt(3/8) = 1.923825 for four_pi_G = 1, rho0 = 1 (validation WP-15).
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
import sgv  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")
R_POLY = 0.75 * np.pi * np.sqrt(2.0)
T_FF = np.pi * np.sqrt(3.0 / 8.0)


def lane_emden(r):
    xi = r / np.sqrt(2.0)
    return np.where(xi > 1e-12, np.sin(xi) / np.maximum(xi, 1e-300), 1.0)


def polytrope(rd):
    rows = []
    for f in sgv.dumps(rd):
        d = sgv.load(f, ["prim"])
        r = np.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        ins = r < R_POLY
        rho = d["prim"][0]
        v = np.sqrt(d["prim"][1] ** 2 + d["prim"][2] ** 2 + d["prim"][3] ** 2)
        ex = lane_emden(r)
        rows.append({"t": d["time"], "rho_c": float(rho[np.argmin(r)]), "rho_max": float(rho.max()),
                     "vmax_in": float(v[ins].max()), "vrms_in": sgv.rms(v[ins], d["vol"][ins]),
                     "L1_rho_in": float(np.sum(np.abs(rho[ins] - ex[ins]) * d["vol"][ins]) / np.sum(ex[ins] * d["vol"][ins]))})
    out = {"run": os.path.relpath(rd, RUNS), "dx_min": float(d["dx"].min()), "series": rows}
    hs = glob.glob(os.path.join(rd, "*.hst"))
    if hs:
        H = sgv.hst(hs[0])
        out["mass_rel_drift"] = float(H["mass"][-1] / H["mass"][0] - 1)
        out["Etot_rel_drift"] = float(H["tot-E"][-1] / H["tot-E"][0] - 1)  # excludes W
        if "cons-W" in H:  # total energy incl. gravitational PE (open box: boundary fluxes not subtracted)
            i0 = int(np.argmax(H["cons-W"] != 0))  # cons-W is 0 on the t = 0 row (no solve yet)
            Et = H["tot-E"] + H["cons-W"]
            out["E_plus_W_rel_drift"] = float((Et[-1] - Et[i0]) / abs(H["cons-W"][i0]))
        out["mom_max_abs"] = float(max(np.abs(H["1-mom"]).max(), np.abs(H["2-mom"]).max(), np.abs(H["3-mom"]).max()))
        ok = H["grav-iters"] >= 0
        out.update(grav_iters_max=int(H["grav-iters"][ok].max()), grav_nonconv=int(H["grav-nonconv"][ok].sum()))
        if "cons-Mout" in H:
            out["mass_out_bnd"] = float(H["cons-Mout"][-1])
    return out


def collision(rd):
    H = sgv.hst(glob.glob(os.path.join(rd, "*.hst"))[0])
    rows = []
    for f in sgv.dumps(rd):
        d = sgv.load(f, ["prim"])
        m = d["prim"][0] * d["vol"]
        rows.append({"t": d["time"], "M": float(m.sum()), "com": [float(np.sum(m * d[c]) / m.sum()) for c in "xyz"],
                     "P": [float(np.sum(m * d["prim"][i])) for i in (1, 2, 3)]})
    M0 = H["mass"][0]
    pscale = 0.5 * M0 * 1.0  # |p| of each star ~ M_star * 1 ; total M ~ 2 M_star (+ ambient)
    i0 = int(np.argmax(H["cons-W"] != 0))
    Et = H["tot-E"] + H["cons-W"]
    return {"run": os.path.relpath(rd, RUNS), "series": rows,
            "E_plus_W_rel_drift": float((Et[-1] - Et[i0]) / abs(H["cons-W"][i0])),
            "mass_rel_drift": float(H["mass"][-1] / H["mass"][0] - 1),
            "P_boundary_out": [float(H["cons-Poutx"][-1]), float(H["cons-Pouty"][-1]), float(H["cons-Poutz"][-1])],
            "hst_t": H["time"].tolist(), "hst_px": H["1-mom"].tolist(), "hst_py": H["2-mom"].tolist(),
            "hst_pz": H["3-mom"].tolist(), "hst_mass": H["mass"].tolist(),
            "max_abs_P_over_pstar": float(max(np.abs(H["1-mom"]).max(), np.abs(H["2-mom"]).max(), np.abs(H["3-mom"]).max()) / pscale)}


def rho_exact(t, rho0=1.0):
    """Exact cycloid for a uniform sphere of density rho0 (t_ff scales as rho0^-1/2)."""
    if t <= 0:
        return rho0
    tff = T_FF / np.sqrt(rho0)
    f = lambda b: (2 / np.pi) * tff * (b + np.sin(b) * np.cos(b)) - t
    b = brentq(f, 0.0, np.pi / 2 - 1e-12)
    return rho0 * np.cos(b) ** -6


def freefall(rd):
    """grav_analytic adds rho_out = 0.01 INSIDE the sphere too, so rho0 = 1.01 there; the exact
    solution is evaluated for the measured initial interior density (WP-15 had rho_in = 1.00 with the
    same 0.01 ambient outside -- the problem is otherwise identical, and scale-free in rho0)."""
    rows = []
    rho0 = None
    for f in sgv.dumps(rd):
        d = sgv.load(f, ["prim"])
        rho = d["prim"][0]
        r = np.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        if rho0 is None:
            rho0 = float(rho[np.argmin(r)])
        ex = rho_exact(d["time"], rho0)
        rows.append({"t": d["time"], "rho_c": float(rho[np.argmin(r)]), "rho_max": float(rho.max()), "rho_exact": ex,
                     "err_pct": float(100 * (rho.max() / ex - 1)),           # rho_max (WP-15's metric; sits at the EDGE)
                     "err_c_pct": float(100 * (rho[np.argmin(r)] / ex - 1))})  # central density (primary, SG-F3)
    return {"run": os.path.relpath(rd, RUNS), "rho0": rho0, "series": rows}


def main():
    res = {"polytrope": [], "collision": [], "freefall": []}
    for rd in sorted(glob.glob(os.path.join(RUNS, "D5_polytrope", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        if sgv.dumps(rd):
            p = polytrope(rd)
            res["polytrope"].append(p)
            s = p["series"]
            print(f"{p['run']:36s} t={s[-1]['t']:.2f} rho_c/rho_c0={s[-1]['rho_c']/s[0]['rho_c']:.5f} "
                  f"vmax_in={max(x['vmax_in'] for x in s):.3e} L1_in(final)={s[-1]['L1_rho_in']:.3e} "
                  f"dM={p.get('mass_rel_drift', np.nan):+.2e} d(E+W)/|W0|={p.get('E_plus_W_rel_drift', np.nan):+.2e} it_max={p.get('grav_iters_max')}")
    for rd in sorted(glob.glob(os.path.join(RUNS, "D6_collision", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        if sgv.dumps(rd):
            c = collision(rd)
            res["collision"].append(c)
            print(f"{c['run']:36s} max|P|/p_star={c['max_abs_P_over_pstar']:.3e} COM(final)={c['series'][-1]['com']}")
    for rd in sorted(glob.glob(os.path.join(RUNS, "D7_freefall", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        if sgv.dumps(rd):
            ff = freefall(rd)
            res["freefall"].append(ff)
            print(ff["run"], "rho_c:", " ".join(f"t={x['t']:.2f}:{x['err_c_pct']:+.3f}%" for x in ff["series"][::4]))
    json.dump(res, open(os.path.join(OUT, "tables", "dynamic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
