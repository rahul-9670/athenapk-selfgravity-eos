"""Shared deck pieces for the ideal-MHD suite (validation_tests/ideal_mhd, 2026-10-12).

PRODUCTION hydro keys of this module, verbatim from the research code's production collapse run (its c1365 restart
Input + the OVR line of its launcher, 2026-10-11; last duplicate wins).

Two tiers (PLAN.md):
  S (scheme)     eos = adiabatic, gamma 5/3. FORCED divergences, each enforced by the code:
                   double_flux = false   (hydro.cpp:965 "double_flux needs eos = hydrogen")
                   eos_interp absent     (tabulated-only key)
                   Tfloor = -1           (ideal-gas Tfloor needs a units block + He fraction, hydro.cpp:1172;
                                          pfloor 1e-8 stays and is far below every S state)
                 boundary_flux_clamp = false in PERIODIC boxes, true with diode BCs as in production. The periodic
                 'false' changes nothing: since DE-F2 was fixed (2026-10-07) the clamp skips periodic faces
                 (hydro.cpp ClampBoundaryFluxes, c1l..c3u). Decks written 2026-10-12 before this note carry the stale
                 comment "DE-F2 (clamp ignores the BC type)"; their behaviour is identical.
  P (production) eos = hydrogen with the rho26 table, all keys verbatim.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import os

M = _VT + "/ideal_mhd"
F = _REPO
EOS_TABLE = F + "/src/eos/eos_table_hires_v5_cold_rho26.bin"

# production scheme keys (every tier)
SCHEME = """fluid = glmmhd
riemann = hlld
reconstruction = plm
divergence_control = glm
glmmhd_source = dedner_extended
glmmhd_alpha = 0.1
first_order_flux_correct = false
vl2_predictor_recon = same
dfloor = 1.0e-6
pfloor = 1.0e-8
eos_floor_consistent = true
lean_registers = true
ghost_sync_rounds = 1
"""
# production dual-energy keys (launcher OVR)
DUAL = """dual_energy = true
dual_energy_eta1 = 5e-2
dual_energy_eta2 = 0.1
dual_energy_kappa = 3.0
dual_energy_shock_comp = 0.02
dual_energy_sync_step_end = true
dual_energy_unsigned_prim = true
dual_energy_exact_resync = true
dual_energy_shared_select = true
dual_comp_ghost = true
dual_energy_heat_rkl2 = true
dual_energy_heat_exact = true
dual_energy_flux = density
boundary_clamp_dual = true
"""
EOS_S = """eos = adiabatic
gamma = 1.6666666666666667
Tfloor = -1             # forced (S tier): ideal-gas Tfloor needs units + He fraction
double_flux = false     # forced (S tier): double_flux needs eos = hydrogen
eos_table_allow_no_radiation = true   # validation-only escape: dual energy without radiation (hydro.cpp:2195)
"""
EOS_P = f"""eos = hydrogen
eos_table_file = {EOS_TABLE}
gamma = 1.4             # production deck value; unused by the table
eos_interp = p
Tfloor = 0.2726
double_flux = true
eos_table_allow_no_radiation = true   # validation-only escape (no radiation in these tests)
"""


def hydro_block(tier, clamp, dual=True, extra=""):
    s = "<hydro>\n" + SCHEME + (EOS_S if tier == "S" else EOS_P)
    s += f"boundary_flux_clamp = {'true' if clamp else 'false'}"
    s += "" if clamp else "   # periodic box: the clamp skips periodic faces anyway (DE-F2 fixed)"
    s += "\n" + (DUAL if dual else "dual_energy = false\n") + extra
    return s


def mesh_block(nx, x1, x2, x3, mb, bc="periodic", smr=None):
    """smr = list of (level, (x1min, x1max), (x2min, x2max), (x3min, x3max)) static-refinement regions."""
    nlev = 1 + max([r[0] for r in smr]) if smr else 1
    s = "<parthenon/mesh>\n" + (f"refinement = static\nnumlevel = {nlev}\n" if smr else "refinement = none\n") + "nghost = 2\n"
    for d, (n, (lo, hi)) in enumerate(zip(nx, (x1, x2, x3)), 1):
        s += f"nx{d} = {n}\nx{d}min = {lo}\nx{d}max = {hi}\nix{d}_bc = {bc}\nox{d}_bc = {bc}\n"
    s += "\n<parthenon/meshblock>\n" + "".join(f"nx{d} = {m}\n" for d, m in enumerate(mb, 1))
    for n, (lev, *box) in enumerate(smr or []):
        s += f"\n<parthenon/static_refinement{n}>\nlevel = {lev}\n"
        s += "".join(f"x{d}min = {lo}\nx{d}max = {hi}\n" for d, (lo, hi) in enumerate(box, 1))
    return s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
