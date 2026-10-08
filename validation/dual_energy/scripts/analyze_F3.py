"""F3 / F3c analysis: hydro/dual_energy_pdv = face vs centred on the D2a perpendicular-field cold sinusoid
(dP = P/P_ad(rho) - 1 at the last dump; max and volume-weighted L1), on uniform meshes (reproduces the D2a perp
rows of F3b_face_pdv.txt) and on refined meshes (F3c/F3d: SMR fine region 0.375-0.625 around the density maximum,
and AMR maxdensity, 3 levels (refine above 1.15, beta 0.001: 1.05 -- rho_max is only 1.087 there; F3e); finest spacing 1024, 2048, 4096 for the order at the level jumps). Also checks that the F3c uniform rerun with the
flux-corrected face velocities (dual_vflux, binary 0445a0fb) is bitwise the F3b run. -> results/tables/F3c_face_pdv_refined.txt"""
import glob, os
import h5py
import numpy as np
from analyze_D2 import cells, Pad
from de_common import RUNS, TAB


def final(d):
    f = glob.glob(os.path.join(RUNS, d, "*.out0.final.phdf")); return f[0] if f else None


def metric(d):
    f = final(d)
    if not f:
        return None
    _, c, _ = cells(f)
    with h5py.File(f) as h:   # 1D: per-cell widths (cells() returns one uniform-mesh volume)
        x = h["Locations"]["x"][()]
        nk, nj = h["prim"].shape[2:4]
        nlev = len(np.unique(h["Levels"][()]))
    vol = np.broadcast_to(np.diff(x, axis=1)[:, None, None, :], (x.shape[0], nk, nj, x.shape[1] - 1)).ravel()
    if ("amr" in d or "smr" in d) and nlev < 2:
        return "NO REFINEMENT"
    dP = np.abs(c["P"] / Pad(c["rho"]) - 1)
    return dP.max(), float(np.sum(dP * vol) / np.sum(vol))


def fmt(m):
    return "MISSING" if m is None else m if isinstance(m, str) else f"max {m[0]:.2e} L1 {m[1]:.2e}"


rows = ["# F3c: dual_energy_pdv = face vs centred, D2a perpendicular field, dP = P/P_ad(rho) - 1 (last dump)"]
for b in ("0.01", "0.001"):
    for n in (1024, 2048, 4096):
        cen = f"D2a/D2a_perp_b{b}_N{n}_on" if n == 1024 else f"D2y/D2a_perp_b{b}_N{n}_on"
        rows.append(f"uniform  beta {b:5s} N {n:5d}  centred: {fmt(metric(cen))} | face: {fmt(metric(f'F3b/D2a_perp_b{b}_N{n}_face'))}")
    for mesh in ("smr", "amr"):
        prev = None
        for fine in (1024, 2048, 4096):
            tag = mesh if fine == 1024 else f"{mesh}{fine}"
            mc, mf = metric(f"F3c/D2a_perp_b{b}_{tag}_centred"), metric(f"F3c/D2a_perp_b{b}_{tag}_face")
            ok = mf is not None and not isinstance(mf, str)
            order = f" | face max order {np.log2(prev / mf[0]):.2f}" if prev is not None and ok else ""
            prev = mf[0] if ok else None
            rows.append(f"{mesh:8s} beta {b:5s} fine {fine:4d} centred: {fmt(mc)} | face: {fmt(mf)}{order}")
a, b = final("F3c/D2a_perp_b0.01_N1024_face_uniform"), final("F3b/D2a_perp_b0.01_N1024_face")
if a and b:
    with h5py.File(a) as x, h5py.File(b) as y:
        same = np.array_equal(x["prim"][()], y["prim"][()])
        dmax = float(np.abs(x["prim"][()] - y["prim"][()]).max())
    rows.append(f"uniform rerun with dual_vflux (0445a0fb) vs F3b: {'BITWISE' if same else f'DIFFER max|d prim| {dmax:.1e}'}")
txt = "\n".join(rows); print(txt)
open(os.path.join(TAB, "F3c_face_pdv_refined.txt"), "w").write(txt + "\n")
