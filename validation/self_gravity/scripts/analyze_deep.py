#!/usr/bin/env python
"""DA1: production collapse_be IC on 6 nested static levels -- phi and force error PER LEVEL against
the exact potential (homogeneous cube + spherical excess). Force from block-local central differences
of phi and of phi* on the same stencil (block-edge cells excluded: no ghosts in the dump).
DA2: deep free fall -- central density vs the exact cycloid through 4 static levels."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyze_static as ast  # noqa: E402
import exact_potentials as ep  # noqa: E402
import sgv  # noqa: E402

RUNS, OUT = ast.RUNS, ast.OUT


def da1(rd, test="S7", L=8.0):
    """Per-level potential and force error on a (static) AMR mesh against the exact potential of `test`."""
    f = sgv.final_dump(rd)
    d = sgv.load(f, ["grav.phi", "prim"])
    nb, sh = d["nblocks"], d["block_shape"]
    phi_ex, _ = ast.exact_for(test, d, L)
    P = d["grav.phi"].reshape(nb, *sh)
    PX = phi_ex.reshape(nb, *sh)
    X, Y, Z = (d[c].reshape(nb, *sh) for c in "xyz")
    H = d["dx"].reshape(nb, *sh)[:, 0, 0, 0]
    lev = d["level"].reshape(nb, *sh)[:, 0, 0, 0]
    rows = []
    for l in np.unique(lev):
        sel = lev == l
        e_phi = ((P[sel] - PX[sel]) / PX[sel]).ravel()
        g = lambda a, h: [(a[:, 1:-1, 1:-1, 2:] - a[:, 1:-1, 1:-1, :-2]) / (2 * h),
                          (a[:, 1:-1, 2:, 1:-1] - a[:, 1:-1, :-2, 1:-1]) / (2 * h),
                          (a[:, 2:, 1:-1, 1:-1] - a[:, :-2, 1:-1, 1:-1]) / (2 * h)]
        h = H[sel][:, None, None, None]
        gn, gx = g(P[sel], h), g(PX[sel], h)
        dg = np.sqrt(sum((a - b) ** 2 for a, b in zip(gn, gx)))
        gm = np.sqrt(sum(b ** 2 for b in gx))
        rel = (dg / np.where(gm > 0, gm, np.inf)).ravel()
        r = np.sqrt(X[sel][:, 1:-1, 1:-1, 1:-1] ** 2 + Y[sel][:, 1:-1, 1:-1, 1:-1] ** 2
                    + Z[sel][:, 1:-1, 1:-1, 1:-1] ** 2).ravel()
        rows.append({"level": int(l), "dx": float(H[sel][0]), "nblocks": int(sel.sum()),
                     "phi_rel_rms": float(np.sqrt(np.mean(e_phi ** 2))), "phi_rel_max": float(np.abs(e_phi).max()),
                     "force_rel_median": float(np.median(rel)), "force_rel_p99": float(np.percentile(rel, 99)),
                     "force_rel_max": float(rel.max()), "r_of_max": float(r[np.argmax(rel)])})
        print(json.dumps(rows[-1]))
    return rows


def da2(rd):
    import analyze_dynamic as adyn
    ff = adyn.freefall(rd)
    for q in ff["series"][::max(1, len(ff["series"]) // 12)] + ff["series"][-1:]:
        print(f"DA2 t={q['t']:.4f} rho_c={q['rho_c']:.5g} exact={q['rho_exact']:.5g} err_c={q['err_c_pct']:+.3f}% "
              f"(rho_max err {q['err_pct']:+.2f}%)")
    return ff


if __name__ == "__main__":
    res = {"DA1": {}, "DA2": {}}
    for rd in sorted(glob.glob(os.path.join(RUNS, "DA2_freefall_deep", "*"))):
        if sgv.dumps(rd):
            res["DA2"][os.path.basename(rd)] = da2(rd)
    for rd in sorted(glob.glob(os.path.join(RUNS, "DA1_deep_static", "*"))):
        if sgv.run_rc(rd) == 0:
            res["DA1"][os.path.basename(rd)] = da1(rd)
    # S3a / S4: static AMR meshes of round 1, graded there in phi only -> per-level FORCE error here
    for test, pat, L in (("S3", "S3_sphere_amr", 8.0), ("S4", "S4_binary", 0.5)):
        res[pat] = {}
        for rd in sorted(glob.glob(os.path.join(RUNS, pat, "*"))):
            if sgv.run_rc(rd) == 0 and sgv.final_dump(rd):
                print(f"== {pat}/{os.path.basename(rd)}")
                res[pat][os.path.basename(rd)] = da1(rd, test, L)
    json.dump(res, open(os.path.join(OUT, "tables", "deep_amr.json"), "w"), indent=1)
