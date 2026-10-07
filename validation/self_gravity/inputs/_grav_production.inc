# ---- self-gravity block: PRODUCTION keys, verbatim from the self-gravity block of the collapse
# ---- production runs. Tests override only *_bc / four_pi_G / use_swindle
# ---- where the problem requires it (periodic boxes), and residual_tolerance in the
# ---- tolerance sweeps. Do NOT edit without re-reading the production restart.
<parthenon/mesh>
task_collection_timeout_in_seconds = 7200   # production value (default 300 hung a 256^3 solve)

<physics>
self_gravity = true

<self_gravity>
units_override     = true
four_pi_G          = 1.0
use_swindle        = false
packed_bc          = true
ix1_bc             = multipole
ox1_bc             = multipole
ix2_bc             = multipole
ox2_bc             = multipole
ix3_bc             = multipole
ox3_bc             = multipole
mp_deterministic   = true
solver_diag        = true
phi_operator_split = true
phi_prolongation   = linear

<self_gravity/solver_params>
solver             = BiCGSTAB
max_iterations     = 200
residual_tolerance = 1.0e-6
relative_residual  = true
do_FAS             = true
max_coarsenings    = 20
smoother           = SRJ2
presmoother        = SRJ2
postsmoother       = SRJ2
preconditioner     = Multigrid
seed_from_phi      = true
