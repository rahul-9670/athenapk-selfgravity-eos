#!/usr/bin/env python
"""R3: free streaming under Sn at full c -- searchlight, crossing beams, shadow (2D x-y, production Sn keys).

Domain [-2, 2] x [-1, 1], 128 x 64 cells (dx = 1/32), dt 0.01, 32^2 meshblocks (8 blocks, 8 ranks, srun: R-F7).
Static gas, sn_matter_coupling = false, arad_code = 1e-30 (no thermal emission), c_code = 1, creduc = 1.
Sources (radiation/sn_test_source*, R3 extension): box x in [-1.9, -1.8], |y - y0| < 0.05, rate Q = 1, into ONE
ordinate: beam A = the ordinate nearest (0.869, 0.360, 0.340) [mu = 0.340, phi = 22.5 deg], y0 = -0.766 so that the
beam centre passes (0, 0); beam B = nearest (0.869, -0.360, 0.340), y0 = +0.766.
Arms: S1 = A (vacuum, kappa 1e-10), S2 = B, S3 = A + B (crossing at the origin), S4 = A + an opaque column
|x|, |y| < 0.125 made by the isobaric top-hat (thermal_amplitude -0.999 -> rho 1000 inside, 0.5003 outside,
kappa 0.04: tau across the column 10, background tau over the box ~0.08). Each in SOU (production) and step.
S4 gas pressure p0 = 1e-6 (2026-10-08): with p0 = 1 the forced dt 0.01 is 2x (128 x 64) / 4x (256 x 128) above the
hydro CFL limit (c_s = 1.83 outside the column); the column edge went unstable (|v| 1.9e-11 at t 0.5 -> 0.51 at t 6,
rho 0.22..1026; the 256 x 128 run died at cycle 23 on a hydro NaN). p0 = 1e-6 puts the hydro dt limit at ~2.6 >> 0.01
so the column stays static; the radiation does not see p (no emission, no coupling). S1-S3 are uniform (v = 0 exactly).
Resolution study (_hr): 256 x 128 (dx = 1/64), tlim 3.5, arms S1 and S4 only.
writes inputs/R3/*.in, inputs/R3_runs.txt, inputs/R3hr_runs.txt
"""
import os

from rad_common import INP, prod_overlay, set_keys

TL = 6.0
SRC_A = ("-1.9 -1.8 -0.816 -0.716 -0.5 0.5", "0.869 0.360 0.340")
SRC_B = ("-1.9 -1.8 0.716 0.816 -0.5 0.5", "0.869 -0.360 0.340")


def deck(blob, hr=False):
    amp = "-0.999" if blob else "0.0"
    nx, ny, tl = (256, 128, 3.5) if hr else (128, 64, TL)
    return f"""<comment>
problem = R3 free streaming (scripts/make_decks_R3.py), blob = {blob}

<job>
problem_id = rad_pulse

<problem/rad_pulse>
rho0 = 1.0
p0 = {"1.0e-6" if blob else "1.0"}
E0 = 0.0
Eamp = 0.0
reduced_flux = 0.0
thermal_wave = true
thermal_isobaric = true
thermal_tophat = true
thermal_tophat_ndim = 2
thermal_tophat_halfwidth = 0.125
thermal_amplitude = {amp}
thermal_radiation_factor = 0.0

<parthenon/mesh>
refinement = none
nghost = 2
nx1 = {nx}
x1min = -2.0
x1max = 2.0
ix1_bc = outflow
ox1_bc = outflow
nx2 = {ny}
x2min = -1.0
x2max = 1.0
ix2_bc = outflow
ox2_bc = outflow
nx3 = 1
x3min = -0.5
x3max = 0.5

<parthenon/meshblock>
nx1 = 32
nx2 = 32
nx3 = 1

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tl!r}
nlim = 100000
dt_force = 0.01

<hydro>
fluid = glmmhd
eos = adiabatic
gamma = 1.6666666666666667
riemann = hlld
reconstruction = plm
dfloor = 1.0e-10
pfloor = 1.0e-12

<physics>
self_gravity = false
radiation = true

<radiation>
transport = sn
n_group = 1
creduc = 1.0
c_code = 1.0
arad_code = 1.0e-30
sn_matter_coupling = false
sn_rel_tol = 1.0e-10
tfloor_code = 1.0e-12
efloor = 1.0e-40
opacity_model = constant
kappa_a_code = {"0.04" if blob else "1.0e-10"}
kappa_s_code = 0.0
sn_test_source = true

<parthenon/output0>
file_type = hdf5
dt = 0.5
variables = prim, rad.Er, rad.Fr1, rad.Fr2, sn.u
"""


def src(t, n, box, dirv):
    return {f"sn_test_source{n}_q": "1.0", f"sn_test_source{n}_tend": "1.0e30", f"sn_test_source{n}_box": box,
            f"sn_test_source{n}_dir": dirv}


def main():
    d = os.path.join(INP, "R3")
    os.makedirs(d, exist_ok=True)
    runs, hruns = [], []
    arms = {"S1": (False, [SRC_A]), "S2": (False, [SRC_B]), "S3": (False, [SRC_A, SRC_B]), "S4": (True, [SRC_A])}
    for arm, (blob, srcs), hr in [(a, v, h) for h in (False, True) for a, v in arms.items() if not h or a in ("S1", "S4")]:
        for sch in ("sou", "step"):
            t = prod_overlay(deck(blob, hr))
            keys = {}
            for i, (box, dirv) in enumerate(srcs):
                keys.update(src(t, "" if i == 0 else "2", box, dirv))
            keys["sn_rel_tol"] = "1.0e-10"
            if sch == "step":
                keys["sn_spatial"] = "step"
            t = set_keys(t, "radiation", keys, "R3")
            tag = f"{arm}_{sch}" + ("_hr" if hr else "")
            open(os.path.join(d, tag + ".in"), "w").write(t)
            (hruns if hr else runs).append(f"R3/{tag} | R3/{tag}.in | | 8")
    open(os.path.join(INP, "R3_runs.txt"), "w").write("\n".join(runs) + "\n")
    open(os.path.join(INP, "R3hr_runs.txt"), "w").write("\n".join(hruns) + "\n")
    print(len(runs), "+", len(hruns), "runs")


if __name__ == "__main__":
    main()
