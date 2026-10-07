#!/usr/bin/env python
"""SG-S1..S7: compare grav.phi of the one-cycle runs with the exact potentials.

Metrics (volume-weighted over all active cells):
  eps_abs  = rms(phi - phi*)                         (Tomida and Stone 2023 eq. 6)
  eps_rel  = rms(phi - phi*) / rms(phi*)
  eps_norm = rms((phi - phi*) / phi*)                (Tomida and Stone 2023 eq. 31, isolated only)
  max_rel  = max |phi - phi*| / |phi*|
  wall_rel = max |phi - phi*| / |phi*| over cells within one cell of the domain boundary
  core_rel = rms((phi - phi*)/phi*) over r < half the inscribed radius
Periodic problems compare zero-mean potentials (the swindle fixes phi only up to a constant).
Writes results/tables/static.json and static.md, and figures/static_*.png.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exact_potentials as ep  # noqa: E402
import sgv  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")

# collapse_be constants (src/pgen/collapse_be.cpp:48-50) and the f = 5 IC
BE_RC, BE_RCSQ, BE_F = 6.45, 26.0 / 3.0, 5.0
BE_RHO = lambda r: BE_F * (1.0 + r * r / BE_RCSQ) ** -1.5
BE_AMB = BE_RHO(BE_RC)


def exact_for(test, d, L):
    x, y, z = d["x"], d["y"], d["z"]
    box = ((-L, -L, -L), (L, L, L))
    if test.startswith("S1"):
        return ep.phi_sinusoid(x, y, z, 1.0, (1.0, 1.0, 1.0), four_pi_G=4 * np.pi), None
    if test.startswith("S2"):
        return ep.phi_gaussian(x, y, z, 1.0, 1.0) + ep.phi_prism(x, y, z, 1e-10, *box), 1.0
    if test.startswith("S3"):
        M = 4.0 / 3.0 * np.pi * 4.0 ** 3
        return ep.phi_uniform_sphere(x, y, z, M, 4.0) + ep.phi_prism(x, y, z, 1e-10, *box), M
    if test.startswith("S4"):
        R = 6.0 / 1024.0
        phi = (ep.phi_uniform_sphere(x, y, z, 2.0, R, c=(6.0 / 1024, 0, 0))
               + ep.phi_uniform_sphere(x, y, z, 1.0, R, c=(-12.0 / 1024, 0, 0))
               + ep.phi_prism(x, y, z, 1e-10, *box))
        return phi, 3.0
    if test.startswith("S5"):
        a, c = 4.0, 1.7435595774162693
        return ep.phi_maclaurin(x, y, z, 1.0, a, c) + ep.phi_prism(x, y, z, 1e-10, *box), 4.0 / 3 * np.pi * a * a * c
    if test.startswith("S6"):
        M, a, rt = 1.0, 1.0, 6.0
        return (ep.phi_plummer_truncated(x, y, z, M, a, rt) + ep.phi_prism(x, y, z, 1e-10, *box),
                M * rt ** 3 / (rt ** 2 + a ** 2) ** 1.5)
    if test.startswith("S7"):
        sph, Mx = ep.phi_spherical_quadrature(x, y, z, lambda r: np.where(r <= BE_RC, BE_RHO(r) - BE_AMB, 0.0), BE_RC)
        return sph + ep.phi_prism(x, y, z, BE_AMB, *box), Mx + BE_AMB * (2 * L) ** 3
    raise ValueError(test)


def analyse(run_dir):
    test = os.path.basename(os.path.dirname(run_dir))
    f = sgv.final_dump(run_dir)
    if f is None:
        return None
    d = sgv.load(f, ["grav.phi", "prim"])
    xmax = float(np.max(np.abs(d["x"]) + 0.5 * d["dx"]))
    L = 0.5 if test.startswith("S1") else xmax
    if test.startswith("S1"):
        L = None
    phi_ex, M_ex = exact_for(test, d, xmax if not test.startswith("S1") else 1.0)
    phi = d["grav.phi"]
    vol = d["vol"]
    rho = d["prim"][0]
    res = {"run": os.path.relpath(run_dir, RUNS), "test": test, "ncells": int(phi.size),
           "dx_min": float(d["dx"].min()), "dx_max": float(d["dx"].max()), "ncycle": d["ncycle"]}
    if test.startswith("S1"):
        e = (phi - np.sum(phi * vol) / vol.sum()) - (phi_ex - np.sum(phi_ex * vol) / vol.sum())
        res.update(eps_abs=sgv.rms(e, vol), eps_rel=sgv.rms(e, vol) / sgv.rms(phi_ex, vol),
                   max_abs=float(np.abs(e).max()))
    else:
        e = phi - phi_ex
        r = np.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        hw = np.max(np.abs(np.stack([d["x"], d["y"], d["z"]])) + 0.5 * d["dx"], axis=0)
        wall = hw >= xmax - 1e-9 * xmax
        core = r < 0.5 * xmax
        rel = e / phi_ex
        res.update(eps_abs=sgv.rms(e, vol), eps_rel=sgv.rms(e, vol) / sgv.rms(phi_ex, vol),
                   eps_norm=sgv.rms(rel, vol), max_rel=float(np.abs(rel).max()),
                   wall_rel=float(np.abs(rel[wall]).max()), wall_rms=sgv.rms(rel[wall], vol[wall]),
                   core_rel=sgv.rms(rel[core], vol[core]),
                   phi_c=float(phi[np.argmin(r)]), phi_c_exact=float(phi_ex[np.argmin(r)]),
                   mass_grid=float(np.sum(rho * vol)), mass_exact=float(M_ex))
    # Force error: central-difference grad phi vs central-difference grad phi* on the same stencil
    # (isolates solver + BC error from gradient truncation). Uniform meshes only.
    G = sgv.to_global(d, phi)
    if G is not None and not test.startswith("S1"):
        P, (xc, yc, zc) = G
        PX = sgv.to_global(d, phi_ex)[0]
        RH = sgv.to_global(d, rho)[0]
        h = float(d["dx"][0])
        def grad(a):
            return [(a[1:-1, 1:-1, 2:] - a[1:-1, 1:-1, :-2]) / (2 * h), (a[1:-1, 2:, 1:-1] - a[1:-1, :-2, 1:-1]) / (2 * h),
                    (a[2:, 1:-1, 1:-1] - a[:-2, 1:-1, 1:-1]) / (2 * h)]
        g, gx = grad(P), grad(PX)
        dg = np.sqrt(sum((a - b) ** 2 for a, b in zip(g, gx)))
        gm = np.sqrt(sum(b ** 2 for b in gx))
        Z, Y, X = np.meshgrid(zc[1:-1], yc[1:-1], xc[1:-1], indexing="ij")
        rr = np.sqrt(X * X + Y * Y + Z * Z)
        w = RH[1:-1, 1:-1, 1:-1]
        ok = gm > 0
        res.update(force_rel_massw=float(np.sum(w[ok] * dg[ok] / gm[ok]) / np.sum(w[ok])),
                   force_rel_core=float(np.median((dg / np.where(ok, gm, np.inf))[(rr < 0.25 * xmax) & ok])),
                   force_rel_max_inner=float(np.max((dg / np.where(ok, gm, np.inf))[(rr < 0.5 * xmax) & (rr > 2 * h) & ok])))
    h = glob.glob(os.path.join(run_dir, "*.hst"))
    if h:
        H = sgv.hst(h[0])
        if isinstance(H, dict) and "grav-iters" in H:
            ok = H["grav-iters"] >= 0
            res["grav_iters"] = [int(v) for v in H["grav-iters"][ok]]
            res["grav_res_rel"] = [float(v) for v in H["grav-res-rel"][ok]]
            res["grav_nonconv"] = int(H["grav-nonconv"][ok].sum())
    return res


def main():
    rows = []
    for rd in sorted(glob.glob(os.path.join(RUNS, "S*", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        if not os.path.isdir(rd):
            continue
        try:
            r = analyse(rd)
        except Exception as exc:  # report, do not hide
            r = {"run": os.path.relpath(rd, RUNS), "error": repr(exc)}
        if r:
            rows.append(r)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) and abs(v) > 1e-3 else v)
                              for k, v in r.items() if not isinstance(v, list)}))
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "tables", "static.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
