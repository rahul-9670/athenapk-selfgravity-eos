#!/usr/bin/env python
"""R2: Su-Olson non-equilibrium Marshak wave (transport + Fleck coupling), option (a) approved 2026-10-08:
LINEARISED about an equilibrium T0 (ideal gas, constant c_v) and run as +A / -A pairs; (U(+A) - U(-A))/2 cancels the
even orders of the linearisation (0-D estimate: residual ~0.6 x^2, x = peak dT/T0 ~ 1e-3 here).

Mapping (code units): eps = 4 a T0^3 / c_v,vol = 4 a T0^3 (gamma - 1) / rho = 1 fixes rho; kappa = 1/rho gives sigma = 1,
so z = x; tau = eps c sigma t = c t. Source Q = c sigma A a T0^4 for |x| < 0.5 while tau < 10 (radiation/sn_test_source,
radiation_sn_testsrc.cpp). U = (E_r - a T0^4)/(A a T0^4), V = (a T^4 - a T0^4)/(A a T0^4).
Domain x1 in [-20, 20] (1D), Planck incoming BC at T0 (the background), production Sn keys, sn_rel_tol 1e-12.
Arms: base (dz 0.05, dtau 0.01, SOU, 4x8) +A/-A; dz 0.025; dtau 0.005; tol 1e-10; step; 8x16 ordinates.
writes inputs/R2/*.in, inputs/R2_runs.txt
"""
import os

from rad_common import INP, prod_overlay, fmt

T_UNIT = 10.015143591970153
ARAD = 0.3856750655749156     # Params radiation/arad (R1 runs, default calibrated units)
C = 1577855.042105263
GAM = 5.0 / 3.0
T0K = 10.0
T0 = T0K / T_UNIT
RHO = 4 * ARAD * T0 ** 3 * (GAM - 1)
KAP = 1.0 / RHO
TAUS = [0.1, 1.0, 3.16228, 10.0]
TEND = 10.0 / C


def deck(A, nx, dtau, extra_rad=""):
    q = C * 1.0 * A * ARAD * T0 ** 4
    dt = dtau / C
    return f"""<comment>
problem = R2 Su-Olson (linearised, option a) -- scripts/make_decks_R2.py; A = {A!r}

<job>
problem_id = rad_pulse

<problem/rad_pulse>
rho0 = {fmt(RHO)}
p0 = {fmt(RHO * T0)}
E0 = 0.0
Eamp = 0.0
reduced_flux = 0.0
thermal_wave = true
thermal_amplitude = 0.0
thermal_tophat = false
temperature_K = {fmt(T0K)}
thermal_radiation_factor = 1.0

<parthenon/mesh>
refinement = none
nghost = 2
nx1 = {nx}
x1min = -20.0
x1max = 20.0
ix1_bc = outflow
ox1_bc = outflow
nx2 = 1
x2min = -0.5
x2max = 0.5
nx3 = 1
x3min = -0.5
x3max = 0.5

<parthenon/meshblock>
nx1 = {nx}

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {fmt(TAUS[-1] / C)}
nlim = 1000000
dt_force = {fmt(dt)}

<hydro>
fluid = glmmhd
eos = adiabatic
gamma = {fmt(GAM)}
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
sn_matter_coupling = true
sn_rel_tol = 1.0e-12
tfloor_code = 5.0e-2
efloor = 1.0e-15
opacity_model = constant
kappa_a_code = {fmt(KAP)}
kappa_s_code = 0.0
sn_test_source = true
sn_test_source_q = {fmt(q)}
sn_test_source_x0 = 0.5
sn_test_source_tend = {fmt(TEND)}
{extra_rad}
<parthenon/output0>
file_type = hdf5
dt = {fmt(TAUS[0] / C)}
variables = prim, rad.Er
"""


ARMS = {  # name: (nx, dtau, overrides baked into <radiation> after the production overlay)
    "base": (800, 0.01, {}),
    "dz2": (1600, 0.01, {}),
    "dt2": (800, 0.005, {}),
    "tol10": (800, 0.01, {"sn_rel_tol": "1.0e-10"}),
    "step": (800, 0.01, {"sn_spatial": "step"}),
    "ang816": (800, 0.01, {"sn_nzeta": "8", "sn_npsi": "16"}),
}


def main():
    from rad_common import set_keys
    d = os.path.join(INP, "R2")
    os.makedirs(d, exist_ok=True)
    runs = []
    for arm, (nx, dtau, ov) in ARMS.items():
        for sgn, A in (("p", 2.0e-3), ("m", -2.0e-3)):
            t = prod_overlay(deck(A, nx, dtau))
            t = set_keys(t, "radiation", {"sn_bc_tbg_K": fmt(T0K), "sn_rel_tol": "1.0e-12"}, "R2 (after the overlay)")
            if ov:
                t = set_keys(t, "radiation", ov, "R2 arm")
            name = f"su_{arm}_{sgn}.in"
            open(os.path.join(d, name), "w").write(t)
            runs.append(f"R2/{arm}_{sgn} | R2/{name} | | 1")
    open(os.path.join(INP, "R2_runs.txt"), "w").write("\n".join(runs) + "\n")
    print(len(runs), "runs; rho", RHO, "kappa", KAP, "tend", TEND)


if __name__ == "__main__":
    main()
