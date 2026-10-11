#!/usr/bin/env python
"""R5: convergence in group count / group edges (R5a, 1-D slab) and in ordinate count (R5b, 3-D cube), production
solver = the ADOPTED fix set (rad_common.adopted_overlay), v11-recipe opacity tables.

Problem ("hot core in a cold envelope"): a dense hot top-hat |x| < HW (rho_core = 1e-11 g/cm^3, T_core = 300 or 1000 K)
in a dilute envelope (rho_env = 1e-13 g/cm^3, 10 K), radiation initially Planck at the local T, outflow boundaries with
the production Planck incoming BC at 10 K. The core is optically thick (tau ~ 1e3) and cools slowly (t_cool ~ 1e3 x the
run), so its photosphere drives a quasi-steady heating of the envelope whose temperature profile depends on how the
envelope absorbs the core's (much hotter) spectrum: kappa_nu ~ nu^1.8 for dust, so a group-mean opacity evaluated at the
envelope's own 10 K under-represents the absorption of 300-1000 K photons. That is the multigroup question for the collapse run.
Gas: ideal, gamma 5/3 (the cores are not pressure-balanced; the run is 1.2e-5 code = 1.8e7 s, the core edge moves
<= ~0.3 AU, the same hydro in every arm). dt_force 2e-7 code (hydro CFL ~0.25 at the fastest v + c_s).

R5a group partitions (name: n_group, nu_min_hz, nu_max_hz; interior edges log-spaced as radiation_groups.hpp
BuildRadGroups): nested L3/L4/L6/L10/L18 over [1e11, 1e15] (4 / 2 / 1 / 0.5 / 0.25 dex; L18 = reference, test binary
with ATHENAPK_MAX_GROUP = 24), the production P3 {1e12, 1e15}, edge-placement variants E3 {1e12, 1e14}, E4 {1e12, 1e13,
1e14}, E8 (0.5 dex over [10^11.5, 10^14.5]), gray G1, and (added after the first results: L6 -> L10 -> L18 converge
only ~linearly in the group width) F10 / F18 = 0.25 / 0.125 dex over [1e12, 1e14], where the core's photons are absorbed. Mesh 1-D 640 cells over +-20 AU (dx 0.0625 AU), HW = 2 AU.
R5b ordinate counts (nzeta x npsi): 4x8 (production), 6x12, 8x16, 12x24 (reference); production groups P3; 3-D cube
|x|,|y|,|z| < 2 AU in a +-6 AU box, 48^3 (dx 0.25 AU), T_core = 300 K.
writes inputs/R5/*.in, inputs/R5/tables.txt (name | --edges for gen_tables_R5.sh), inputs/R5a_runs.txt, inputs/R5b_runs.txt
"""
import os

import numpy as np

from rad_common import INP, RUNS, adopted_overlay, fmt

RHO_UNIT = 5.467e-19                 # g/cm^3
T_UNIT = 10.015143591970153          # K
AU = 1.495978707e13 / 2.81e16        # code length
RHO_ENV, RFAC = 1.0e-13, 100.0     # rho_core = RFAC rho_env = 1e-11 g/cm^3
T_ENV = 10.0
HW = 2.0 * AU
DT = 2.0e-7
NSTEP = 60
TAB = os.path.join(RUNS, "R5", "tables")
PC = "radiation/sn_precond_stencil=sou"   # R-F12 fix (binary 93752d2d and later)

FAMS = {"L3": (3, 1e11, 1e15), "L4": (4, 1e11, 1e15), "L6": (6, 1e11, 1e15), "L10": (10, 1e11, 1e15),
        "L18": (18, 1e11, 1e15), "P3": (3, 1e12, 1e15), "E3": (3, 1e12, 1e14), "E4": (4, 1e12, 1e14),
        "E8": (8, 10 ** 11.5, 10 ** 14.5), "F10": (10, 1e12, 1e14), "F18": (18, 1e12, 1e14)}
TCORE = (300.0, 1000.0)
ORDS = ((4, 8), (6, 12), (8, 16), (12, 24))


def edges(n, lo, hi):
    """The runtime partition (radiation_groups.hpp BuildRadGroups), as the generator's --edges string."""
    ll, lh = np.log10(lo), np.log10(hi)
    e = [0.0] + [float(10.0 ** (ll + (lh - ll) * (i - 1) / (n - 2))) for i in range(1, n)] + [1e30]
    return ",".join(repr(x) for x in e)


def deck(Tc, fam, nd, ords=(4, 8)):
    n, lo, hi = FAMS[fam] if fam != "G1" else (1, 1e12, 1e15)
    table = os.path.join(TAB, f"opac_{fam if fam != 'G1' else 'P3'}.bin")
    T0 = 0.5 * (Tc + T_ENV)
    amp = (Tc - T_ENV) / (Tc + T_ENV)
    if nd == 1:
        L, nx, ny, nz, mb = 20.0 * AU, 640, 1, 1, (640, 1, 1)
    else:
        L, nx, ny, nz, mb = 6.0 * AU, 48, 48, 48, (16, 16, 16)
    rho = RHO_ENV / RHO_UNIT
    mesh = "".join(f"""nx{d} = {N}
x{d}min = {fmt(-L if N > 1 else -0.5)}
x{d}max = {fmt(L if N > 1 else 0.5)}
""" + (f"ix{d}_bc = outflow\nox{d}_bc = outflow\n" if N > 1 else "") for d, N in ((1, nx), (2, ny), (3, nz)))
    return f"""<comment>
problem = R5 hot core in a cold envelope (validation/radiation_sn, scripts/make_decks_R5.py): {fam}, T_core {Tc} K, {nd}-D

<job>
problem_id = rad_pulse

<problem/rad_pulse>
rho0 = {fmt(rho)}
p0 = {fmt(rho * T_ENV / T_UNIT)}
E0 = 0.0
Eamp = 0.0
reduced_flux = 0.0
thermal_wave = true
thermal_tophat = true
thermal_tophat_ndim = {nd}
thermal_tophat_halfwidth = {fmt(HW)}
thermal_tophat_rho_factor = {fmt(RFAC)}
temperature_K = {fmt(T0)}
thermal_amplitude = {fmt(amp)}
thermal_radiation_factor = 1.0

<parthenon/mesh>
refinement = none
nghost = 2
{mesh}
<parthenon/meshblock>
nx1 = {mb[0]}
nx2 = {mb[1]}
nx3 = {mb[2]}

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {fmt(NSTEP * DT)}
nlim = 100000
dt_force = {fmt(DT)}

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
n_group = {n}
nu_min_hz = {fmt(lo)}
nu_max_hz = {fmt(hi)}
creduc = 1.0
sn_matter_coupling = true
sn_rel_tol = 1.0e-8
tfloor_code = 5.0e-2
efloor = 1.0e-15
opacity_model = tabulated
opacity_table_file = {table}
sn_nzeta = {ords[0]}
sn_npsi = {ords[1]}

<parthenon/output0>
file_type = hdf5
dt = {fmt(NSTEP * DT / 4)}
variables = prim, {", ".join(["rad.Er"] + [f"rad.Er_g{g}" for g in range(1, n)])}
"""


def main():
    d = os.path.join(INP, "R5")
    os.makedirs(d, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    open(os.path.join(d, "tables.txt"), "w").write(
        "".join(f"{f} | {edges(*FAMS[f])}\n" for f in FAMS))
    ra = []
    for Tc in TCORE:
        for fam in list(FAMS) + ["G1"]:
            name = f"a_{fam}_T{int(Tc)}"
            # the sn_nzeta / sn_npsi lines are rewritten by the overlay (PROD_RAD sets 4 x 8); R5b sets them per run
            open(os.path.join(d, name + ".in"), "w").write(adopted_overlay(deck(Tc, fam, 1)))
            ra.append(f"R5/{name} | R5/{name}.in |  | 1")
    rb = []
    open(os.path.join(d, "b_P3_T300.in"), "w").write(adopted_overlay(deck(300.0, "P3", 3)))
    for nz, npsi in ORDS:
        rb.append(f"R5/b_o{nz}x{npsi} | R5/b_P3_T300.in | radiation/sn_nzeta={nz} radiation/sn_npsi={npsi} {PC} | 27")
    open(os.path.join(INP, "R5a_runs.txt"), "w").write("\n".join(ra) + "\n")
    # R-F12 (found by the first R5a pass: every 1000 K arm failed in BiCGSTAB): the full R5a set again with the
    # SOU-consistent preconditioner, run dirs R5/c_*; the step-preconditioned 300 K arms (R5/a_*) stay as the check that
    # the converged answer does not depend on the preconditioner.
    open(os.path.join(INP, "R5c_runs.txt"), "w").write("\n".join(
        l.replace("R5/a_", "R5/c_", 1).replace(" |  | ", f" | {PC} | ") for l in ra) + "\n")
    open(os.path.join(INP, "R5b_runs.txt"), "w").write("\n".join(rb) + "\n")
    print(len(ra), "R5a runs,", len(rb), "R5b runs")


if __name__ == "__main__":
    main()
