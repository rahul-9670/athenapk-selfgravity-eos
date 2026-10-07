#!/usr/bin/env python
"""Write the input decks of the literature (L*) and deep-AMR (DA*) tests from blocks + the production
gravity include (inputs/_grav_production.inc, copied verbatim -- never edited here)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.normpath(os.path.join(HERE, "..", "inputs"))
TAB = os.path.join(INP, "tables")
GRAV = open(os.path.join(INP, "_grav_production.inc")).read()
GRAV = GRAV[GRAV.index("<parthenon/mesh>"):]  # starts with the timeout key, then <physics> ...
GRAV_MESH_KEY = "task_collection_timeout_in_seconds = 7200   # production value\n"
GRAV_REST = GRAV[GRAV.index("<physics>"):]
META = json.load(open(os.path.join(TAB, "tables_meta.json")))


def mesh(L, n, mb, refinement="none", numlevel=1, bc="outflow", lo=None, extra=""):
    lo = -L if lo is None else lo
    s = f"<parthenon/mesh>\n{GRAV_MESH_KEY}refinement = {refinement}\nnumlevel   = {numlevel}\nmultigrid  = true\n"
    for d in (1, 2, 3):
        s += (f"nx{d} = {n[d - 1]}\nx{d}min = {lo[d - 1] if isinstance(lo, (list, tuple)) else lo}\n"
              f"x{d}max = {L[d - 1] if isinstance(L, (list, tuple)) else L}\nix{d}_bc = {bc}\nox{d}_bc = {bc}\n")
    s += extra + f"\n<parthenon/meshblock>\nnx1 = {mb[0]}\nnx2 = {mb[1]}\nnx3 = {mb[2]}\n"
    return s


def static_levels(halfwidths):
    s = ""
    for l, h in enumerate(halfwidths, 1):
        s += (f"\n<parthenon/static_refinement{l}>\nlevel = {l}\nx1min = {-h}\nx1max = {h}\n"
              f"x2min = {-h}\nx2max = {h}\nx3min = {-h}\nx3max = {h}\n")
    return s


def hydro(gamma, extra=""):
    return (f"<hydro>\nfluid          = glmmhd\nglmmhd_source  = dedner_extended\neos            = adiabatic\n"
            f"gamma          = {gamma}\nriemann        = hlld\nreconstruction = plm\ncons_diag      = true\n{extra}")


def time(tlim, cfl=0.3, integrator="vl2", extra=""):
    return f"<parthenon/time>\nintegrator = {integrator}\ncfl        = {cfl}\ntlim       = {tlim}\nnlim       = -1\n{extra}"


def outputs(dt0, dt1=None, variables="prim, grav.phi", single=True, dn0=None):
    s = f"<parthenon/output0>\nfile_type = hdf5\nvariables = {variables}\nsingle_precision_output = {'true' if single else 'false'}\n"
    s += f"dn = {dn0}\n" if dn0 else f"dt = {dt0}\n"
    s += f"\n<parthenon/output1>\nfile_type = hst\ndt = {dt1 if dt1 else dt0 / 10}\n"
    return s


def grav(four_pi_G=1.0, periodic=False):
    g = GRAV_REST.replace("four_pi_G          = 1.0", f"four_pi_G          = {four_pi_G!r}")
    if periodic:
        g = g.replace("use_swindle        = false", "use_swindle        = true")
        g = "\n".join(l for l in g.splitlines() if not (l.strip().startswith(("ix", "ox")) and "multipole" in l)) + "\n"
    return g


PROD_BC = "boundary_flux_clamp = true\n"   # production hydro BCs: diode + inward-flux clamp
# Production hydro keys for supersonic collapse / turbulence: floors. Without them the cold
# supersonic flows of L6 (xi = 9), L7 and L10 hit E - KE - ME < 0 ("Got negative pressure"), 2026-10-03.
PROD_HYDRO = PROD_BC + "dfloor = 1.0e-6\npfloor = 1.0e-8\n"
# (with pfloor > 0 a transient E - KE - ME < 0 is floored instead of aborting, and the barotropic step
#  resets e_th every stage.)


def jeans_ref(nj, numlevel):
    return f"\n<refinement>\ntype   = jeans\nnjeans = {nj}\n"


def write(name, comment, *blocks):
    """Existing decks are NOT overwritten when their content differs (runs record deck md5s), unless the
    name matches the regex in $SG_DECKS_REWRITE."""
    import re
    txt = f"<comment>\nproblem = {comment}\n\n" + "\n".join(blocks)
    path = os.path.join(INP, name + ".in")
    if os.path.exists(path):
        if open(path).read() == txt:
            return
        if not re.fullmatch(os.environ.get("SG_DECKS_REWRITE", "(?!)"), name):
            print("  KEPT (differs, not rewritten)", name + ".in")
            return
    open(path, "w").write(txt)
    print("  wrote", name + ".in")


def prof(table, extra=""):
    return (f"<job>\nproblem_id = profile1d\n\n<problem/profile1d>\ntable = {os.path.join(TAB, table)}\n{extra}")


# ---------------------------------------------------------------- L1 polytrope + atmosphere
m = META["L1_polytrope_atm"]
write("L1_polytrope_atm", "n = 1 polytrope (gamma = 2) with a hydrostatic self-gravitating atmosphere (Mullen,\n"
      "          Hanawa and Gammie 2021 Sec. 4.3); G = M = R = 1, Ma = 0.01, box 8R, multipole BC. tau = R/v_esc = 0.7071",
      prof("L1_polytrope_atm.txt", "geometry = spherical\nnsub = 2\n"),
      mesh(4.0, (80, 80, 80), (16, 16, 16)), time(7.0710678), hydro(2.0, "pfloor = 1e-14\ndfloor = 1e-12\n"),
      outputs(0.70710678), grav(m["four_pi_G"]))

# ---------------------------------------------------------------- L2 Spitzer sheet
m = META["L2_spitzer"]
write("L2_spitzer", "advected Spitzer sheet (Mullen et al. 2021 Sec. 4.1): Gamma = gamma = 1.2, G = K = 1,\n"
      "          mean rho 0.3, period 4 along x, contrast 1.45e4, advected at vx = 1 for one period (t = 4).",
      prof("L2_spitzer.txt", "geometry = planar\nvx = 1.0\nnsub = 4\n"),
      mesh((4.0, 0.5, 0.5), (64, 8, 8), (32, 8, 8), bc="periodic", lo=(0.0, 0.0, 0.0)),
      time(4.0), hydro(1.2, "pfloor = 1e-14\ndfloor = 1e-12\n"), outputs(0.5), grav(m["four_pi_G"], periodic=True))

# ---------------------------------------------------------------- L3 n = 5 polytrope (equilibrium, collapse)
write("L3_n5_polytrope", "n = 5 (Plummer) polytrope in equilibrium (Hanawa and Mullen 2025 Sec. 4.4): rho_c = R = 1,\n"
      "          4 pi G = 1, P_c = 1/6, gamma = 5/3, box [-8, 8]^3, multipole BC; 10 dynamical times.",
      prof("L3_n5_polytrope.txt", "geometry = spherical\nnsub = 2\n"),
      mesh(8.0, (64, 64, 64), (16, 16, 16)), time(10.0), hydro(5 / 3), outputs(1.0), grav(1.0))
write("L3_n5_collapse", "n = 5 polytrope collapse (Hanawa and Mullen 2025 Sec. 4.6): pressure made uniform inside 0.9 R,\n"
      "          gamma = 1.3, h = 0.08 R (box [-5.12, 5.12]^3 at 128^3); bounce and accretion shock; vs 1D reference.",
      prof("L3_n5_collapse.txt", "geometry = spherical\nnsub = 2\n"),
      mesh(5.12, (128, 128, 128), (32, 32, 32)), time(6.0), hydro(1.3), outputs(0.1), grav(1.0))

# ---------------------------------------------------------------- L4 Lin-Mestel-Shu spheroid
write("L4_lms_spheroid", "pressure-free collapse of a homogeneous oblate spheroid (Lin, Mestel and Shu 1965): a = 4,\n"
      "          c = 2, rho = 1 (+0.01 floor), four_pi_G = 1; exact semi-axes from the index-symbol ODE.",
      "<job>\nproblem_id = grav_analytic\n\n<problem/grav_analytic>\nprofile = ellipsoid\na1 = 4.0\na2 = 4.0\na3 = 2.0\n"
      "rho_in = 1.0\nrho_out = 1.0e-2\npressure = 1.0e-6\nnsub = 8\n",
      mesh(8.0, (128, 128, 128), (32, 32, 32)),
      time(2.5, cfl=0.2, extra="dt_ceil = 0.0005\ndt_init = 1.0e-4\n"), hydro(1.4, "dfloor = 1e-10\npfloor = 1e-14\n"),
      outputs(0.05, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- L5 Evrard
m = META["L5_evrard"]
write("L5_evrard", "Evrard (1988) adiabatic collapse: M = R = G = 1, rho = 1/(2 pi r), u = 0.05, gamma = 5/3;\n"
      "          energies (kinetic, thermal, potential) vs the 1D Lagrangian reference; multipole BC.",
      prof("L5_evrard.txt", "geometry = spherical\nrho_out = 1.6e-5\np_out = 5.3e-7\nnsub = 2\n"),
      mesh(2.0, (128, 128, 128), (32, 32, 32)), time(3.0), hydro(5 / 3, "dfloor = 1e-8\npfloor = 1e-12\n"),
      outputs(0.1), grav(m["four_pi_G"]))

# ---------------------------------------------------------------- L6 BE stability (collapse_be, exact profile)
for xi, f in [(3.0, 1.1), (6.45, 1.0), (9.0, 1.0), (9.0, 1.1)]:
    L = max(8.0, np.ceil(1.35 * xi))
    write(f"L6_be_xi{xi:g}_f{f:g}", f"exact isothermal Bonnor-Ebert sphere xi_max = {xi}, overdensity f = {f}, pressure-bounded\n"
          "          (collapse_be exterior momentum BC), production barotropic EOS, Jeans AMR.",
          "<job>\nproblem_id = collapse_be\n\n<problem/collapse_be>\nmass = 6.0\ntemperature = 10.0\n"
          f"f = {f}\nrhocrit = 1.0e-6\nbe_profile = exact\nxi_max = {xi}\namp = 0.0\nomegatff = 0.0\nturb_mach = 0.0\n",
          mesh(L, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=6, bc="diode"), jeans_ref(8, 6),
          time(30.0), hydro(1.4, PROD_HYDRO), outputs(1.0, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- L7 / DA3: production IC, deep Jeans AMR
for nj in (4, 8, 16):
    write(f"L7_be_collapse_nj{nj}", "production collapse_be IC (Tomida profile, f = 5, M = 6 Msun, T = 10 K), ideal hydro, no\n"
          f"          rotation/turbulence/B, production barotropic EOS (rhocrit 1e-13 g/cc), adaptive Jeans AMR njeans = {nj}.\n"
          "          Deep-AMR test DA3: vs the 1D Lagrangian reference and the Larson-Penston similarity solution.",
          "<job>\nproblem_id = collapse_be\n\n<problem/collapse_be>\nmass = 6.0\ntemperature = 10.0\nf = 5.0\n"
          "rhocrit = 1.0e-13\namp = 0.0\nomegatff = 0.0\nturb_mach = 0.0\n",
          mesh(8.0, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=14, bc="diode"), jeans_ref(nj, 14),
          time(2.0), hydro(1.4, PROD_HYDRO), outputs(None, dt1=1e-3, variables="prim, grav.phi", dn0=25), grav(1.0))

# ---------------------------------------------------------------- L8 SIS (Shu)
write("L8_sis", "singular isothermal sphere rho = 2/r^2 (cs = 1, 4 pi G = 1), at rest, core capped at r = 0.02;\n"
      "          production barotropic enforcement (rho_crit 1e6: the capped core, rho = 5000, starts <1 % over-pressured; with 1e4 it started 25 % over and bounced); vs Shu (1977) expansion wave.",
      prof("L8_sis.txt", "geometry = spherical\nnsub = 8\nbarotropic = true\nrho_crit = 1.0e6\n"),
      mesh(4.0, (128, 128, 128), (16, 16, 16), refinement="static", numlevel=3, extra=static_levels([1.0, 0.5])),
      time(2.0), hydro(1.4, PROD_HYDRO), outputs(0.25), grav(1.0))

# ---------------------------------------------------------------- L9 Boss-Bodenheimer
m = META["L9_boss_bodenheimer"]
for nj in (4, 8):
    write(f"L9_boss_bodenheimer_nj{nj}", "Boss and Bodenheimer (1979) standard isothermal test: uniform sphere alpha = 0.26,\n"
          f"          beta = 0.16, m = 2 perturbation 10 %; rho0 = cs = 1, 4 pi G = 1; barotropic at 2.6e4; Jeans AMR njeans = {nj}.",
          prof("L9_boss_bodenheimer.txt", f"geometry = spherical\nnsub = 2\nomega = {m['Omega']}\nm2_amp = {m['m2_amp']}\n"
               "rho_out = 0.01\np_out = 0.01\nbarotropic = true\nrho_crit = 2.6e4\n"
               f"zero_mom_radius = {m['R']:.10f}\n"),
          mesh(12.0, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=12), jeans_ref(nj, 12),
          time(2.6), hydro(1.4, PROD_HYDRO), outputs(0.05, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- DA1 static deep nesting, production IC Poisson
write("DA1_be_ic_deep_static", "production collapse_be IC, single Poisson solve on 6 nested static levels (finest dx = 0.0039):\n"
      "          phi and force error per level vs the exact potential.",
      "<job>\nproblem_id = collapse_be\n\n<problem/collapse_be>\nmass = 6.0\ntemperature = 10.0\nf = 5.0\nrhocrit = 1.0e-13\n"
      "amp = 0.0\nomegatff = 0.0\nturb_mach = 0.0\n",
      mesh(8.0, (64, 64, 64), (16, 16, 16), refinement="static", numlevel=7, bc="diode",
           extra=static_levels([4.0, 2.0, 1.0, 0.5, 0.25, 0.125])),
      "<parthenon/time>\nintegrator = vl2\ncfl = 1.0e-6\ntlim = 1.0e30\nnlim = 1\n",
      hydro(1.4, PROD_BC), "<parthenon/output0>\nfile_type = hdf5\ndt = 1.0e30\nvariables = prim, grav.phi\n", grav(1.0))

# ---------------------------------------------------------------- DA2 free fall through 4 nested levels
write("DA2_freefall_deep", "pressure-free sphere (R = 4, rho0 = 1.01 incl. floor) collapsing through 4 nested static levels\n"
      "          to rho/rho0 ~ 1e3; central density vs the exact cycloid. dt_ceil 5e-4 suppresses the SG-F4 term.",
      "<job>\nproblem_id = grav_analytic\n\n<problem/grav_analytic>\nprofile = spheres\nm1 = 268.082573106329\nr1 = 4.0\n"
      "rho_out = 1.0e-2\npressure = 1.0e-6\nnsub = 8\n",
      mesh(8.0, (64, 64, 64), (16, 16, 16), refinement="static", numlevel=5, extra=static_levels([5.0, 2.5, 1.25, 0.625])),
      time(1.92, cfl=0.2, extra="dt_ceil = 0.0005\ndt_init = 1.0e-4\n"),
      hydro(1.4, "dfloor = 1e-10\npfloor = 1e-14\n"), outputs(0.02, dt1=0.002, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- L10 Hennebelle et al. 2011
P10 = json.load(open(os.path.join(TAB, "L10_params.json")))
m = META["L10_hennebelle"]
for mu in (120, 5, 2):
    write(f"L10_hennebelle_mu{mu}", "Hennebelle et al. (2011) massive magnetised turbulent barotropic core: rho = rho_c/(1+(r/r0)^2),\n"
          f"          contrast 10, alpha_th = 0.12 (=> 142 Msun at their rho_c), E_turb = 0.2 |W| (Kolmogorov, seed 42), mu = {mu},\n"
          "          barotropic rho_crit 1e-13 g/cc, uniform B (theirs ~ column density), Jeans AMR. Statistical benchmark.",
          "<job>\nproblem_id = collapse_be\n\n<problem/collapse_be>\n"
          f"mass = {P10['mass_param']!r}\ntemperature = 10.0\nf = 1.0\nrhocrit = 1.0e-13\nbe_profile = hennebelle\n"
          f"xi_max = {m['R']!r}\nprofile_r0 = {m['r0']!r}\namp = 0.0\nomegatff = 0.0\nB0z = {P10[f'B0z_mu{mu}']!r}\n"
          f"turb_mach = {P10['turb_mach']!r}\nturb_alpha = 3.667\nturb_ksample = k2\nturb_seed = 42\nturb_kmin = 1\n"
          "turb_kmax = 8\nturb_nmodes = 2048\nturb_region = sphere\n",
          mesh(21.0, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=10, bc="diode"), jeans_ref(8, 10),
          time(2.6), hydro(1.4, PROD_HYDRO), outputs(0.05, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- L6' BE stability, light hot ambient (profile1d)
for xi, f in [(3.0, 1.1), (6.45, 1.0), (9.0, 1.0), (9.0, 1.1)]:
    m = META[f"L6b_be_xi{xi:g}_f{f:g}"]
    L = 8.0 if xi < 7 else 12.0
    write(f"L6b_be_xi{xi:g}_f{f:g}", f"exact isothermal Bonnor-Ebert sphere xi_max = {xi}, f = {f}, confined by a LIGHT hot ambient\n"
          "          (1/100 edge density at the edge pressure), gamma = 1.001 (near-isothermal), outflow BC, Jeans AMR;\n"
          "          vs the 1D Lagrangian reference (constant external pressure). Replaces the collapse_be L6 (SG-F5).",
          prof(f"L6b_be_xi{xi:g}_f{f:g}.txt", f"geometry = spherical\nnsub = 2\nrho_out = {m['rho_edge'] / 100!r}\n"
               f"p_out = {m['rho_edge']!r}\n"),
          mesh(L, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=6), jeans_ref(8, 6),
          time(30.0), hydro(1.001, "dfloor = 1.0e-6\npfloor = 1.0e-8\n"), outputs(1.0, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- L6c = L6' with the PRODUCTION hydro BCs
# L6' used zero-gradient outflow faces: hot ambient streamed in (box mass +5.1 %, confining pressure +21 % by
# t = 25) and lifted the oscillation's minima (verified with the 1D model driven by the measured pressure ramp).
# Production uses diode + boundary_flux_clamp; L6c uses the same.
m = META["L6b_be_xi3_f1.1"]
for nj in (8, 32):
    write(f"L6c_be_xi3_f1.1_nj{nj}", "exact isothermal Bonnor-Ebert sphere xi_max = 3, f = 1.1, light hot ambient, gamma = 1.001,\n"
          "          PRODUCTION hydro BCs (diode + boundary_flux_clamp: no inflow through the faces); vs the 1D reference\n"
          "          with the same ambient and a constant outer pressure (scripts/ref1d_L6amb.py case a).",
          prof("L6b_be_xi3_f1.1.txt", f"geometry = spherical\nnsub = 2\nrho_out = {m['rho_edge'] / 100!r}\n"
               f"p_out = {m['rho_edge']!r}\n"),
          mesh(8.0, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=6, bc="diode"), jeans_ref(nj, 6),
          time(30.0), hydro(1.001, "dfloor = 1.0e-6\npfloor = 1.0e-8\n" + PROD_BC), outputs(1.0, variables="prim"), grav(1.0))

# ---------------------------------------------------------------- L6d = L6' in a CLOSED box (reflecting_val walls)
# L6c (diode + flux clamp) DRAINED the light ambient (box mass -24..-29 %, confining pressure -> 7 %) and the
# sphere came apart. Neither open BC conserves the ambient; reflecting_val walls do. 1D reference: same
# ambient, rigid wall at the volume-equivalent radius (scripts/ref1d_L6rigid.py).
for nj in (8, 32):
    write(f"L6d_be_xi3_f1.1_nj{nj}", "exact isothermal Bonnor-Ebert sphere xi_max = 3, f = 1.1, light hot ambient, gamma = 1.001,\n"
          "          CLOSED box (reflecting_val walls: mirror, normal momentum and normal B negated); vs the 1D\n"
          "          reference with the same ambient and a rigid wall at the volume-equivalent radius.",
          prof("L6b_be_xi3_f1.1.txt", f"geometry = spherical\nnsub = 2\nrho_out = {m['rho_edge'] / 100!r}\n"
               f"p_out = {m['rho_edge']!r}\n"),
          mesh(8.0, (64, 64, 64), (16, 16, 16), refinement="adaptive", numlevel=6, bc="reflecting_val"), jeans_ref(nj, 6),
          time(30.0), hydro(1.001, "dfloor = 1.0e-6\npfloor = 1.0e-8\n"), outputs(1.0, variables="prim"), grav(1.0))
