#!/usr/bin/env python
"""R1: 0-D multigroup gas-radiation relaxation (uniform 8^3 periodic box, static gas) vs the exact multigroup ODE.

Collapse regime: rho = 1e-13 g/cm^3 (rho_code = 1e-13/rho_unit), n_group = 3 on the production edges
{0, 1e12, 1e15, inf} Hz, production Sn solver keys (rad_common.PROD_RAD + dual keys), creduc = 1, matter coupling on.
Gas: ideal, gamma = 5/3 (the EOS is not under test here; the tabulated EOS enters via dTde only, R1 keeps it exact).
IC: T_gas = temperature_K, radiation = Planck at thermal_trad_K (rad_pulse key added for R1), split into groups.
Opacity families: c0/c1/c2 = constant kappa 1 cm^2/g x analytic shape beta = 0/1/2 (radiation_groups.hpp), tab = v9
table (gray magnitude from the table, group multipliers from the file; no dust package -> kdust = 1).
dt ladder: dt = DT_PROD * 4^-k, k = 0..6, DT_PROD = 1.1e-5 code (production dt, RAD_ACCURACY #5), to t = 4 DT_PROD.
At k = 0 the stiffness is S ~ 3e2 (production-like). Fleck alpha = 1 everywhere, + alpha = 0.5 on one family.
writes inputs/R1/base_<pair>_<op>.in, inputs/R1_runs.txt
"""
import os

from rad_common import INP, prod_overlay, fmt
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root

RHO_UNIT = 5.467e-19      # g/cm^3 (Params Hydro/phys_units/rho_unit_cgs)
OPAC_UNIT = 0.01536227    # cm^2/g -> code (Params Hydro/phys_units/opacity_unit)
T_UNIT = 10.015143591970153
RHO = 1.0e-13 / RHO_UNIT
DT_PROD = 1.1e-5
KS = range(7)
PAIRS = {"g100r30": (100.0, 30.0), "g30r100": (30.0, 100.0), "g15r10": (15.0, 10.0)}
OPS = {"c0": ("constant", 0.0), "c1": ("constant", 1.0), "c2": ("constant", 2.0), "tab": ("tabulated", 0.0)}
TABLE = _REPO + "/src/radiation/opacity_table_v9_pollack_fmt3.bin"


def deck(Tg, Tr, model, beta):
    op = (f"opacity_model = tabulated\nopacity_table_file = {TABLE}\n" if model == "tabulated" else
          f"opacity_model = constant\nkappa_a_code = {fmt(OPAC_UNIT)}   # 1 cm^2/g\nkappa_s_code = 0.0\n"
          f"opacity_beta = {fmt(beta)}\n")
    return f"""<comment>
problem = R1 0-D multigroup relaxation (validation/radiation_sn, scripts/make_decks_R1.py)

<job>
problem_id = rad_pulse

<problem/rad_pulse>
rho0 = {fmt(RHO)}
p0 = {fmt(RHO * Tg / T_UNIT)}
E0 = 0.0
Eamp = 0.0
reduced_flux = 0.0
thermal_wave = true
thermal_amplitude = 0.0
thermal_tophat = false
temperature_K = {fmt(Tg)}
thermal_trad_K = {fmt(Tr)}
thermal_radiation_factor = 1.0

<parthenon/mesh>
refinement = none
nghost = 2
nx1 = 8
x1min = -0.5
x1max = 0.5
ix1_bc = periodic
ox1_bc = periodic
nx2 = 8
x2min = -0.5
x2max = 0.5
ix2_bc = periodic
ox2_bc = periodic
nx3 = 8
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic

<parthenon/meshblock>
nx1 = 8
nx2 = 8
nx3 = 8

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {fmt(4 * DT_PROD)}
nlim = 100000
dt_force = {fmt(DT_PROD)}

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
n_group = 3
nu_min_hz = 1.0e12
nu_max_hz = 1.0e15
creduc = 1.0
sn_matter_coupling = true
sn_rel_tol = 1.0e-8
tfloor_code = 5.0e-2
efloor = 1.0e-15
{op}
<parthenon/output0>
file_type = hdf5
dt = {fmt(DT_PROD / 4)}
variables = prim, rad.Er, rad.Er_g1, rad.Er_g2
"""


def main():
    d = os.path.join(INP, "R1")
    os.makedirs(d, exist_ok=True)
    runs = []
    for pn, (Tg, Tr) in PAIRS.items():
        for on, (model, beta) in OPS.items():
            name = f"base_{pn}_{on}.in"
            open(os.path.join(d, name), "w").write(prod_overlay(deck(Tg, Tr, model, beta)))
            alphas = ("1.0", "0.5") if (pn, on) == ("g100r30", "c2") else ("1.0",)
            for a in alphas:
                for k in KS:
                    dt = DT_PROD * 4.0 ** -k
                    runs.append(f"R1/{pn}_{on}_a{a}_k{k} | R1/{name} | parthenon/time/dt_force={fmt(dt)} "
                                f"radiation/sn_fleck_alpha={a} | 1")
    open(os.path.join(INP, "R1_runs.txt"), "w").write("\n".join(runs) + "\n")
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
