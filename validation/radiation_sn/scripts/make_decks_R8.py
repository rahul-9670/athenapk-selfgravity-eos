#!/usr/bin/env python
"""R8 decks (benchmarks with literature answers).

R8a GANAPOL PLANE SOURCE: infinite purely, isotropically scattering medium; unit isotropic plane pulse at x = 0;
reference = scripts/ganapol_ref.py (Ganapol 2001 closed form, verified against Monte Carlo). Code units chosen so the
benchmark's units are the code's: c_code = 1, rho = 1, kappa_s_code = 1 (sigma_s = 1 per code length), kappa_a = 0,
matter coupling off, vacuum incoming BC. 1-D, x in [-1.5, 1.5], 1200 cells (dx 0.0025); pulse = Gaussian, width 0.005,
integral 1 (the reference is convolved with it). Ordinates 4x8 (production) / 8x16 / 16x32; dt 2e-3 (production-like
c dt / dx = 0.8) and 5e-4; SOU (adopted set incl. the limiter) and step. Output at t = 0.5, 1.

R8b RADIATIVE SHOCK (Lowrie & Edwards 2008 model problem, gray, absorption only): colliding flows (rad_shock pgen),
upstream rho0 = 1, a0 = 1 (T0 = 1/gamma code), P0 = 1e-4 (a_code = P0/T0^4), c_code = 1000, kappa_a = 577.35 per mass;
lab inflow speed chosen so the shock-frame upstream Mach number is M0 (static-model RH: u_in = M0 (1 - 1/rho1)).
Reference = scripts/radshock_ref.py (non-equilibrium diffusion phase-plane solution, static and full O(u/c) models).
1-D x in [-0.04, 0.04], 4000 cells; M0 = 2 and 3; t = 0.01, 0.015, 0.02.

R8c 3-D POINT SOURCE (Ganapol point-source benchmark; replaces the DUSTY-type dust-shell benchmark of the plan, which
needs a stellar point source, a power-law density pgen and a 1000x radial range the code does not have): the R8a medium
in 3-D (pure isotropic scattering == steady gray radiative equilibrium with negligible heat capacity), an isotropic
3-D Gaussian pulse (width 0.05, integral 1) at the origin; reference = ganapol_ref.point_smoothed (plane -> point
relation, exact). Box [-1.25, 1.25]^3, 100^3 cells (dx 0.025), 20^3 blocks, dt 0.02 (c dt/dx 0.8), t = 0.5, 1.
Ordinates 4x8 / 8x16 (SOU adopted set) and 4x8 step.
writes inputs/R8/*.in, inputs/R8a_runs.txt, inputs/R8b_runs.txt, inputs/R8c_runs.txt
"""
import os

from rad_common import INP, adopted_overlay, fmt

W = 0.005
WC = 0.05
ORDS = ((4, 8), (8, 16), (16, 32))
DTS = (2e-3, 5e-4)


def deck_plane():
    import math
    return f"""<comment>
problem = R8a Ganapol plane source (validation/radiation_sn, scripts/make_decks_R8.py)

<job>
problem_id = rad_pulse

<problem/rad_pulse>
rho0 = 1.0
p0 = 1.0
E0 = 1.0e-12
Eamp = {fmt(1.0 / (math.sqrt(2.0 * math.pi) * W))}
x0 = 0.0
width = {fmt(W)}
reduced_flux = 0.0

<parthenon/mesh>
refinement = none
nghost = 2
nx1 = 1200
x1min = -1.5
x1max = 1.5
ix1_bc = outflow
ox1_bc = outflow
nx2 = 1
x2min = -0.5
x2max = 0.5
nx3 = 1
x3min = -0.5
x3max = 0.5

<parthenon/meshblock>
nx1 = 1200
nx2 = 1
nx3 = 1

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 1.0
nlim = 1000000
dt_force = 2.0e-3

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
c_code = 1.0
creduc = 1.0
sn_matter_coupling = false
sn_rel_tol = 1.0e-10
opacity_model = constant
kappa_a_code = 0.0
kappa_s_code = 1.0
efloor = 1.0e-30
sn_bc_incoming = vacuum

<parthenon/output0>
file_type = hdf5
dt = 0.5
variables = prim, rad.Er
"""


def deck_point():
    import math
    t = deck_plane()
    t = t.replace(f"Eamp = {fmt(1.0 / (math.sqrt(2.0 * math.pi) * W))}", f"Eamp = {fmt(1.0 / (2.0 * math.pi * WC * WC) ** 1.5)}")
    t = t.replace(f"width = {fmt(W)}", f"width = {fmt(WC)}\npulse_ndim = 3")
    mesh = "".join(f"nx{d} = 100\nx{d}min = -1.25\nx{d}max = 1.25\nix{d}_bc = outflow\nox{d}_bc = outflow\n" for d in (1, 2, 3))
    a = t.index("nx1 = 1200")
    b = t.index("<parthenon/meshblock>")
    t = t[:a] + mesh + "\n" + t[b:]
    t = t.replace("<parthenon/meshblock>\nnx1 = 1200\nnx2 = 1\nnx3 = 1", "<parthenon/meshblock>\nnx1 = 20\nnx2 = 20\nnx3 = 20")
    t = t.replace("dt_force = 2.0e-3", "dt_force = 2.0e-2").replace("R8a Ganapol plane source", "R8c Ganapol point source")
    assert "pulse_ndim = 3" in t and "nx3 = 20" in t and "x3max = 1.25" in t
    return t


RS_P0, RS_C, RS_KAP = 1.0e-4, 1.0e3, 577.35
RS_M = (2.0, 3.0)


def deck_shock(M0):
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from radshock_ref import Shock
    g = 5.0 / 3.0
    sh = Shock(M0, RS_P0, RS_C, RS_KAP, full=False)
    E1 = sh.downstream()
    rho1 = sh.state(E1, 0.0, 0)[1]
    uin = M0 * (1.0 - 1.0 / rho1)
    T0 = 1.0 / g
    return f"""<comment>
problem = R8b radiative shock M0 = {M0} (validation/radiation_sn, scripts/make_decks_R8.py); shock-frame rho1 = {float(rho1)!r}

<job>
problem_id = rad_shock

<problem/rad_shock>
rho0 = 1.0
T0 = {fmt(T0)}
v0 = {fmt(uin)}
smooth = 1.0e-4

<parthenon/mesh>
refinement = none
nghost = 2
nx1 = 4000
x1min = -0.04
x1max = 0.04
ix1_bc = outflow
ox1_bc = outflow
nx2 = 1
x2min = -0.5
x2max = 0.5
nx3 = 1
x3min = -0.5
x3max = 0.5

<parthenon/meshblock>
nx1 = 4000
nx2 = 1
nx3 = 1

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 0.02
nlim = 1000000

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
c_code = {fmt(RS_C)}
arad_code = {fmt(RS_P0 / T0 ** 4)}
creduc = 1.0
sn_matter_coupling = true
sn_rel_tol = 1.0e-8
opacity_model = constant
kappa_a_code = {fmt(RS_KAP)}
kappa_s_code = 0.0
efloor = 1.0e-30
sn_bc_incoming = outflow
tfloor_code = 1.0e-3

<parthenon/output0>
file_type = hdf5
dt = 0.005
variables = prim, rad.Er
"""


def main():
    d = os.path.join(INP, "R8")
    os.makedirs(d, exist_ok=True)
    txt = adopted_overlay(deck_plane())
    # the overlay sets sn_bc_incoming = planck (production); R8a needs vacuum: put it back explicitly
    txt = txt.replace("sn_bc_incoming = planck   # R0 prod overlay", "sn_bc_incoming = vacuum   # R8a (not production)")
    assert "sn_bc_incoming = vacuum   # R8a" in txt
    open(os.path.join(d, "plane.in"), "w").write(txt)
    runs = []
    for nz, npsi in ORDS:
        for dt in DTS:
            for sp in ("sou", "step"):
                o = f"radiation/sn_nzeta={nz} radiation/sn_npsi={npsi} parthenon/time/dt_force={fmt(dt)}"
                if sp == "step":
                    o += " radiation/sn_spatial=step radiation/sn_sou_limiter=none"
                runs.append(f"R8/plane_o{nz}x{npsi}_dt{dt:.0e}_{sp} | R8/plane.in | {o} | 1")
    open(os.path.join(INP, "R8a_runs.txt"), "w").write("\n".join(runs) + "\n")
    print(len(runs), "R8a runs")
    rb = []
    for M0 in RS_M:
        t = adopted_overlay(deck_shock(M0))
        t = t.replace("sn_bc_incoming = planck   # R0 prod overlay", "sn_bc_incoming = outflow   # R8b (not production)")
        assert "sn_bc_incoming = outflow   # R8b" in t
        nm = f"shock_M{M0:.0f}"
        open(os.path.join(d, nm + ".in"), "w").write(t)
        rb.append(f"R8/{nm} | R8/{nm}.in | radiation/sn_precond_stencil=sou | 1")
        rb.append(f"R8/{nm}_step | R8/{nm}.in | radiation/sn_precond_stencil=sou radiation/sn_spatial=step | 1")
    open(os.path.join(INP, "R8b_runs.txt"), "w").write("\n".join(rb) + "\n")
    t = adopted_overlay(deck_point()).replace("sn_bc_incoming = planck   # R0 prod overlay",
                                              "sn_bc_incoming = vacuum   # R8c (not production)")
    assert "sn_bc_incoming = vacuum   # R8c" in t
    open(os.path.join(d, "point.in"), "w").write(t)
    rc = [f"R8/point_o4x8 | R8/point.in | radiation/sn_precond_stencil=sou | 64",
          f"R8/point_o8x16 | R8/point.in | radiation/sn_nzeta=8 radiation/sn_npsi=16 radiation/sn_precond_stencil=sou | 64",
          f"R8/point_o4x8_step | R8/point.in | radiation/sn_spatial=step radiation/sn_sou_limiter=none | 64"]
    open(os.path.join(INP, "R8c_runs.txt"), "w").write("\n".join(rc) + "\n")


if __name__ == "__main__":
    main()
