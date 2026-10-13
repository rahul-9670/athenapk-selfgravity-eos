#!/usr/bin/env python
"""Fix campaign (2026-10-12, user: "fix imf1 and imf2 properly. improve the glm damping order"), binary d4ca78e9.
New keys (all default off; key-off = bitwise):
  GLM  hydro/glmmhd_damping = midpoint                     (IM-F3: exponential-midpoint psi damping)
  F1   hydro/dual_energy_shock_eta1 = true + hydro/dual_energy_shock_pjump = 1.0   (IM-F1: dt-independent shock flag)
  F2   hydro/boundary_flux_clamp_wall = true (now with the acoustic wall pressure) + hydro/diode_energy_consistent = true
Writes inputs/F_runs*.txt ("rundir | deck | overrides | nranks") and the wall-test decks inputs/F/wall_*.in.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write

GLM = "hydro/glmmhd_damping=midpoint"
F1 = "hydro/dual_energy_shock_eta1=true hydro/dual_energy_shock_pjump=1.0"
F2 = "hydro/boundary_flux_clamp_wall=true hydro/diode_energy_consistent=true"


def wall_deck(u0, By):
    """Uniform rho 1, P 0.6, By, gas moving AWAY from both walls at u0 (vx = +u0 left half, -u0 right half -> a
    compression at the centre, irrelevant): each no-inflow wall must produce the exact reflecting-wall
    rarefaction (analyze_fix.py). gamma 5/3, [0,1], N 256, diode + clamp, t 0.3."""
    h = hydro_block("S", clamp=True)
    st = (f"rhoL = 1.0\nrhoR = 1.0\nvxL = {u0!r}\nvxR = {-u0!r}\nvyL = 0.0\nvyR = 0.0\nvzL = 0.0\nvzR = 0.0\n"
          f"PL = 0.6\nPR = 0.6\nByL = {By!r}\nByR = {By!r}\nBzL = 0.0\nBzR = 0.0\n")
    return f"""<comment>
problem = ideal_mhd fix campaign: no-inflow wall test, gas leaving both walls at u0 = {u0}, By = {By}

<job>
problem_id = ideal_mhd

<problem/ideal_mhd>
mode = shocktube
x0 = 0.5
B0x = 0.0
{st}
{mesh_block((256, 1, 1), (0, 1), (-0.5, 0.5), (-0.5, 0.5), (256, 1, 1), bc='diode')}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 0.3
nlim = -1

{h}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = prim
dt = 0.1
"""


def main():
    r1, rbig = [], []
    add = lambda L, d, k, o="", n=1: L.append(f"F/{d} | {k} | {o} | {n}")
    # key-off bitwise against earlier runs
    for d, k in (("keyoff_I7_BW_diode", "I7/I7_BW_diode.in"), ("keyoff_I5_2d_prod_N128", "I5/I5_2d_prod_N128.in"),
                 ("keyoff_noh_1d_prod_N1024", "I6/noh_1d_prod_N1024.in"), ("keyoff_I1_1d_fast_N64", "I1/I1_1d_fast_N64.in")):
        add(r1, d, k)
    # GLM: splitting test, fixed-lambda ladders, production-alpha ladder, coarse pulse, field loop
    for dim in (1, 2):
        add(r1, f"glm_I5_{dim}d_lam10_N512", f"I5/I5_{dim}d_lam10_N512.in", GLM)
        for cfl in ("0.15", "0.075"):
            add(r1, f"glm_I5_{dim}d_lam10_N512_cfl{cfl}", f"I5/I5_{dim}d_lam10_N512_cfl{cfl}.in", GLM)
        for N in ((64, 128, 256, 1024) if dim == 1 else (64, 128, 256)):
            add(r1, f"glm_I5_{dim}d_lam10_N{N}", f"I5/I5_{dim}d_lam10_N{N}.in", GLM)
    for N in (64, 128, 256, 512):
        add(r1, f"glm_I5_2d_prod_N{N}", f"I5/I5_2d_prod_N{N}.in", GLM)
    for lev in range(4):
        add(r1, f"glm_I5c_2d_prod_N64_patch{lev}", f"I5/I5c_2d_prod_N64_patch{lev}.in", GLM)
    add(rbig, "glm_I4_2d_prod_N128", "I4/I4_2d_prod_N128.in", GLM, 8)
    add(rbig, "glm_I4_3d_prod_N64", "I4/I4_3d_prod_N64.in", GLM, 16)
    # F1: Mach scan, resolution, threshold sensitivity, pjump without shock_eta1, multi-D, tabulated D3, regression
    for P in ("1e-2", "1e-3", "1e-4", "1e-5", "1e-6"):
        add(r1, f"f1_noh1d_P{P}", "I6/noh_1d_prod_N1024.in", f"problem/ideal_mhd/P0={P} {F1}")
    for N in (2048, 4096):
        add(r1, f"f1_noh1d_P1e-6_N{N}", "I6/noh_1d_prod_N1024.in",
            f"{F1} parthenon/mesh/nx1={N} parthenon/meshblock/nx1={N}")
    for pj in ("0.5", "2.0"):
        add(r1, f"f1_noh1d_P1e-6_pj{pj}", "I6/noh_1d_prod_N1024.in",
            f"hydro/dual_energy_shock_eta1=true hydro/dual_energy_shock_pjump={pj}")
    add(r1, f"f1only_pjump_noh1d_P1e-6", "I6/noh_1d_prod_N1024.in", "hydro/dual_energy_shock_pjump=1.0")
    add(rbig, "f1_noh_2d_N256", "I6/noh_2d_prod_N256.in", F1, 64)
    add(rbig, "f1_noh_2d_N512", "I6/noh_2d_prod_N512.in", F1, 64)
    add(rbig, "f1_noh_3d_N128", "I6/noh_3d_prod_N128.in", F1, 64)
    add(rbig, "f1_noh_3d_N64", "I6/noh_3d_prod_N64.in", F1, 8)
    for m in (3, 10, 30):
        add(r1, f"f1_D3_M{m}", f"I6/D3_M{m}_N1024.in", f"hydro/vl2_predictor_recon=same {F1}")
    for tag, o in (("prod", ""), ("f1", F1)):
        add(r1, f"f1reg_D1b_M1000_N256_{tag}", "../../dual_energy/inputs/D1b_M1000_N256.in",
            f"hydro/vl2_predictor_recon=same {o}".strip())
    # F2: I7 suite with the full fix, the wall-only arm (no energy key), and the exact no-inflow wall test
    for p in ("BW", "RJ2a_div", "divb_a0", "divb_prod"):
        add(r1, f"f2_I7_{p}", f"I7/I7_{p}_diode.in", F2)
    for u0 in ("0.0", "0.01", "0.1"):
        for By in ("0.0", "1.0"):
            name = f"wall_u{u0}_By{By}"
            write(f"{M}/inputs/F/{name}.in", wall_deck(float(u0), float(By)))
            for tag, o in (("legacy", ""), ("f2", F2), ("wallonly", "hydro/boundary_flux_clamp_wall=true")):
                add(r1, f"{name}_{tag}", f"F/{name}.in", o)
    write(f"{M}/inputs/F_runs_r1.txt", "\n".join(r1) + "\n")
    write(f"{M}/inputs/F_runs_big.txt", "\n".join(rbig) + "\n")
    print(len(r1), "1-rank runs,", len(rbig), "multi-rank runs")


if __name__ == "__main__":
    main()
