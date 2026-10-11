"""Shared helpers for the radiation (Sn) + opacity validation module (validation/radiation_sn).

PROD_RAD = the Sn SOLVER keys of the production collapse runs (as of 2026-10-08, incl. sn_dual_energy, sn_fallback,
sn_krylov_dot_scale, sn_skip_extinction, sn_skip_group_rel).
Gate-defining keys (sn_rel_tol, sn_matter_coupling, c_code, arad_code, opacities, n_group, creduc, floors) are NOT
overlaid: each gate keeps its own and says why. PROD_HYDRO_DUAL = the adopted dual-energy keys
(dual_energy/scripts/de_common.py), needed because radiation/sn_dual_energy is inert without hydro/dual_energy.
"""
import os
import re
import sys
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root

# RAD_MOD (2026-10-09): a mirror module root (inputs and scripts symlinked; own runs/ and results/), used for the
# production-configuration regression. Default = this module.
MOD = os.environ.get("RAD_MOD", _REPO + "/validation/radiation_sn")
INP = os.path.join(MOD, "inputs")
RUNS = os.path.join(MOD, "runs")
TAB = os.path.join(MOD, "results", "tables")
FIG = os.path.join(MOD, "results", "figures")
PY = sys.executable
OLD = os.environ.get("RAD_OLD_GATES", "")   # run dirs of the earlier Sn gates (R0 archived numbers; not shipped)
BIN_CPU = _REPO + "/validation/_build/bin/athenaPK_cpu_c69ee834"

sys.path.insert(0, _REPO + "/validation/dual_energy/scripts")
from de_common import DUAL_KEYS  # noqa: E402

PROD_RAD = {
    "transport": "sn",
    "sn_solver": "bicgstab",
    "sn_smoother": "ordered",
    "sn_precond_sweeps": "1",
    "sn_spatial": "sou",
    "sn_sou_kappa": "1.0",
    "sn_sou_linear_prolong": "true",
    "sn_red_black": "true",
    "sn_diamond_fixup": "true",
    "sn_criterion": "residual",
    "sn_rosseland_excess": "true",
    "sn_fleck_alpha": "1.0",
    "sn_max_iterations": "4000",
    "sn_nzeta": "4",
    "sn_npsi": "8",
    "sn_bc_incoming": "planck",
    "sn_bc_tbg_K": "10.0",
    "sn_radiation_force": "false",
    "sn_restart_reproject": "false",
    "sn_dual_energy": "true",
    "sn_fallback": "true",
    "sn_krylov_dot_scale": "true",
    "sn_skip_extinction": "true",
    "sn_skip_group_rel": "1e-12",
}
PROD_HYDRO_DUAL = dict(l.split(" = ") for l in DUAL_KEYS.splitlines())


def fmt(x):
    return repr(float(x))  # numpy 2 writes np.float64(x), which Parthenon's atof reads as 0 (trap)


def set_keys(text, block, keys, note="R0 prod overlay"):
    """Set key = value inside <block> of a Parthenon deck: replace an existing assignment (comments on that line are
    dropped) or append the key at the end of the block. Returns the new text."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == f"<{block}>"), None)
    if start is None:
        lines += ["", f"<{block}>"]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("<")), len(lines))
    todo = dict(keys)
    for i in range(start + 1, end):
        m = re.match(r"\s*([A-Za-z0-9_]+)\s*=", lines[i])
        if m and m.group(1) in todo:
            lines[i] = f"{m.group(1)} = {todo.pop(m.group(1))}   # {note}"
    ins = [f"{k} = {v}   # {note}" for k, v in todo.items()]
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    lines[end:end] = ins
    return "\n".join(lines) + "\n"


def prod_overlay(text):
    text = set_keys(text, "radiation", PROD_RAD)
    return set_keys(text, "hydro", PROD_HYDRO_DUAL)


# The ADOPTED radiation fix set (fix rounds 1+2, adopted 2026-10-08):
# solver/opacity-lookup keys on top of PROD_RAD. dust/opacity_evap = pollack is the eighth adopted key but belongs to the
# dust package (inert without <dust>); table v11 replaces v9 (TABLE_V11).
ADOPTED_RAD = {
    "opacity_interp": "log",
    "sn_norm": "b",
    "sn_local_tol": "1.0e-3",
    "sn_conserve_residual": "true",
    "sn_fc_scatter": "true",
    "sn_sou_limiter": "positive",
}
TABLE_V11 = _REPO + "/src/radiation/opacity_table_v11_pollack_acc_gasfrac_fmt3.bin"


def adopted_overlay(text):
    """prod_overlay + the adopted fix keys (the production configuration after the switch)."""
    return set_keys(prod_overlay(text), "radiation", ADOPTED_RAD, note="adopted fix set")
