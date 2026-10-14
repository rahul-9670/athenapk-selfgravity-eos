"""Shared deck pieces for the dust suite (validation_tests/dust, 2026-10-12).

PRODUCTION keys verbatim from the research code's production collapse run (its c1365 restart Input + the OVR line of
its launcher, 2026-10-12; last duplicate wins), as in ../ideal_mhd/scripts/im_common.py PLUS the ideal-MHD fix keys
(glmmhd_damping = midpoint, dual_energy_shock_eta1 + kv detector, boundary_flux_clamp_wall, diode_energy_consistent)
and the dust block. The decks were generated BEFORE the dust fix keys existed: every dust fix key is absent (= off)
unless a deck sets it as a test key.

Forced divergences (each says why):
  * glmmhd_source = dedner_plain in 1D decks: dedner_extended indexes y/z neighbours and segfaults in 1D (E-F3).
  * boundary_flux_clamp = false in periodic boxes (the clamp skips periodic faces since DE-F2; behaviour identical).
  * eos_table_allow_no_radiation = true: validation-only escape (no radiation in these tests).
  * radiation / gravity off: these tests isolate the dust package; T comes from the hydro EOS state, exactly the
    temperature ReactDust reads in production.
Units: eos_test has no BE units block, so PhysUnits uses the default FHC scales (rho_unit 5.467e-19 g/cm^3, v 1.9e4
cm/s, l 2.81e16 cm -> t_unit = l/v); the analysis reads the scales actually used from the run's own output Params.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import os

DU = _VT + "/dust"
F = _REPO
EOS_TABLE = F + "/src/eos/eos_table_hires_v5_cold_rho26.bin"
BIN = os.environ.get("BIN", _REPO + "/build/bin/athenaPK")   # the athenaPK under test

SCHEME = """fluid = glmmhd
riemann = hlld
reconstruction = plm
divergence_control = glm
glmmhd_source = {glm}
glmmhd_alpha = 0.1
glmmhd_damping = midpoint
first_order_flux_correct = false
vl2_predictor_recon = same
dfloor = 1.0e-6
pfloor = 1.0e-8
eos_floor_consistent = true
lean_registers = true
ghost_sync_rounds = 1
"""
DUAL = """dual_energy = true
dual_energy_eta1 = 5e-2
dual_energy_eta2 = 0.1
dual_energy_kappa = 3.0
dual_energy_shock_comp = 0.02
dual_energy_shock_eta1 = true
dual_energy_shock_detector = kv
dual_energy_sync_step_end = true
dual_energy_unsigned_prim = true
dual_energy_exact_resync = true
dual_energy_shared_select = true
dual_comp_ghost = true
dual_energy_heat_rkl2 = true
dual_energy_heat_exact = true
dual_energy_flux = density
boundary_clamp_dual = true
diode_energy_consistent = true
"""
EOS_P = f"""eos = hydrogen
eos_table_file = {EOS_TABLE}
gamma = 1.4             # production deck value; unused by the table
eos_interp = p
Tfloor = 0.2726
double_flux = true
eos_table_allow_no_radiation = true   # validation-only escape (no radiation in these tests)
"""

# production <dust> block (Input + launcher); every key explicit, including the run-time defaults
DUST = """<dust>
evolve = true
scalar_index = 0
a_ref_cm = 1.0e-5
f_dg_ref = 0.01
T_subl_K = 1.5e3
opacity_size_law = bridged
opacity_gas_split = true
rho_grain_cgs = 3
alpha_turb = 0.01
omega_ref_cgs = 1e-13
mu_n = 2.33
growth_cap = 0.1
omega_local = true
subl_irreversible = true
freeze_growth = false
dual_energy = true
"""


def hydro_block(dim, clamp, nscalars=2, extra=""):
    s = "<hydro>\n" + SCHEME.format(glm="dedner_plain   # 1D: dedner_extended segfaults (E-F3)" if dim == 1
                                      else "dedner_extended") + EOS_P
    s += f"boundary_flux_clamp = {'true' if clamp else 'false'}\n"
    if clamp:
        s += "boundary_flux_clamp_wall = true\n"
    s += DUAL + f"nscalars = {nscalars}\n" + extra
    return s


def dust_block(**over):
    s = DUST
    for k, v in over.items():
        lines = s.splitlines()
        hit = [i for i, l in enumerate(lines) if l.split("=")[0].strip() == k]
        if hit:
            lines[hit[0]] = f"{k} = {v}   # test value (production: {lines[hit[0]].split('=')[1].strip()})"
        else:
            lines.append(f"{k} = {v}   # test key (production: absent = default)")
        s = "\n".join(lines) + "\n"
    return s


def mesh_block(nx, x1, x2, x3, mb, bc="periodic", bcs=None, extra=""):
    s = "<parthenon/mesh>\nrefinement = none\nnghost = 2\n" + extra
    for d, (n, (lo, hi)) in enumerate(zip(nx, (x1, x2, x3)), 1):
        b = (bcs or {}).get(d, (bc, bc))
        s += f"nx{d} = {n}\nx{d}min = {lo}\nx{d}max = {hi}\nix{d}_bc = {b[0]}\nox{d}_bc = {b[1]}\n"
    s += "\n<parthenon/meshblock>\n" + "".join(f"nx{d} = {m}\n" for d, m in enumerate(mb, 1))
    return s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
