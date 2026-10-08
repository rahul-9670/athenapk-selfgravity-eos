"""Shared deck builder for the dual-energy module (validation_tests/dual_energy).

Every deck carries the production keys of this module verbatim (Tfloor 2.73 K, double flux, rho26 table, continuity
keys vl2_predictor_recon = dc and eos_interp = p) and the production hydro numerics (hlld / plm / vl2 / cfl 0.3 /
dfloor 1e-6 / pfloor 1e-8 / boundary_flux_clamp). Exceptions, each stated in the deck: boundary_flux_clamp = false on
periodic/inflow boxes (DE-F2); glmmhd_source = dedner_plain in 1D (E-F3: dedner_extended indexes y/z neighbours and
segfaults in 1D).
The dual-energy OFF variant of a run is the same deck with the CLI override hydro/dual_energy=false."""
import os
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root
import sys

sys.path.insert(0, _REPO + "/validation/eos_tabulated/scripts")
import eostab as E          # noqa: E402
import riemann_exact as RX  # noqa: E402

MOD = _REPO + "/validation/dual_energy"
INP = os.path.join(MOD, "inputs")
RUNS = os.path.join(MOD, "runs")
TAB = os.path.join(MOD, "results", "tables")
FIG = os.path.join(MOD, "results", "figures")
TABLE = _REPO + "/src/eos/eos_table_hires_v5_cold_rho26.bin"  # md5 cb68bd0f
EOS_INTERP = "p"  # production continuity key

# production dual-energy keys (submit_prod.sh, all default OFF in the code)
DUAL_KEYS = """dual_energy = true
dual_energy_eta1 = 5e-2
dual_energy_sync_step_end = true
dual_energy_unsigned_prim = true
dual_energy_exact_resync = true
dual_energy_eta2 = 0.1
dual_energy_kappa = 3.0
dual_energy_shock_comp = 0.02
dual_comp_ghost = true
boundary_clamp_dual = true
ghost_sync_rounds = 1"""


def table_eos():
    return RX.TableEOS(E.load(TABLE), EOS_INTERP)


def fmt(x):
    return repr(float(x))  # numpy 2 would write np.float64(x), which Parthenon's atof reads as 0 (trap)


def hydro_block(ndim, double_flux=True, extra="", clamp=False):
    src = "dedner_plain   # 1D: dedner_extended segfaults (E-F3); B.grad(psi) needs >= 2D" if ndim == 1 else \
        "dedner_extended   # production"
    return f"""<hydro>
fluid = glmmhd
eos = hydrogen
eos_table_file = {TABLE}
gamma = 1.4
riemann = hlld
reconstruction = plm
glmmhd_source = {src}
dfloor = 1.0e-6
pfloor = 1.0e-8
Tfloor = 0.2726   # code T units (10.0 K) -> 2.73 K, production 2026-10-06
boundary_flux_clamp = {"true   # production (outflow BCs)" if clamp else "false   # production = true, but it zeroes inflow on EVERY mesh-edge face without checking the BC type: a periodic box loses its through-flow (DE-F2)"}
cons_diag = true
eos_floor_consistent = true
eos_interp = {EOS_INTERP}
vl2_predictor_recon = dc
double_flux = {"true" if double_flux else "false"}
{DUAL_KEYS}
{extra}
"""


def mesh_block(nx, x1=(0.0, 1.0), nx2=1, nx3=1, x2=(-0.5, 0.5), x3=(-0.5, 0.5), bc="periodic", mb=None, nghost=None):
    mb = mb or (nx, nx2, nx3)
    g = f"nghost = {nghost}\n" if nghost else ""
    return f"""<parthenon/mesh>
refinement = none
{g}nx1 = {nx}
x1min = {fmt(x1[0])}
x1max = {fmt(x1[1])}
ix1_bc = {bc}
ox1_bc = {bc}
nx2 = {nx2}
x2min = {fmt(x2[0])}
x2max = {fmt(x2[1])}
ix2_bc = periodic
ox2_bc = periodic
nx3 = {nx3}
x3min = {fmt(x3[0])}
x3max = {fmt(x3[1])}
ix3_bc = periodic
ox3_bc = periodic

<parthenon/meshblock>
nx1 = {mb[0]}
nx2 = {mb[1]}
nx3 = {mb[2]}
"""


def deck(comment, problem, mesh, tlim, dt_out, hydro, extra_blocks="", hst_dt=None, vars_out="prim, cons, dual_switched"):
    hst = f"""
<parthenon/output1>
file_type = hst
dt = {fmt(hst_dt)}
""" if hst_dt else ""
    return f"""<comment>
problem = {comment}

<job>
problem_id = eos_test

{problem}
{mesh}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {fmt(tlim)}
nlim = -1

{hydro}
<physics>
self_gravity = false

<parthenon/output0>
file_type = hdf5
variables = {vars_out}
dt = {fmt(dt_out)}
{hst}{extra_blocks}"""
