#!/usr/bin/env python
"""R6: GPU (H100) vs CPU on the module's gates, SAME source (validation branch d98fe50d + rad_pulse pulse_ndim),
SAME deck, SAME rank count (1 rank: every block on one device / one core), so the only difference is the device.
Gates: R0 G9 (prod, creduc 1), R0 K4 (prod, tau_cell 0.5, verify-operator on), R1 g100r30 tab k0, R3 S1 / S3 with the
limiter (2-D 128x64), R5 c_P3_T1000 (1-D 640-cell block: E-F8b), R8a plane 4x8 (1-D 1200-cell block > 1024: E-F8b),
R8c point 4x8 (3-D 100^3, 125 blocks on one device). The adopted fix set is on wherever the deck has it.
writes inputs/R6_cpu_runs.txt, inputs/R6_gpu_runs.txt (rundir | deck | overrides | nranks)
"""
import os

from rad_common import INP

PC = "radiation/sn_precond_stencil=sou"
G = [("G9", "R0/G9_prod.in", "radiation/creduc=1.0"),
     ("K4", "R0/K4_prod.in", "radiation/kappa_s_code=64.0 parthenon/time/tlim=0.076800 parthenon/time/cfl=0.2115 "
      "parthenon/output0/dt=0.007680 radiation/sn_solver=bicgstab radiation/sn_verify_operator=true"),
     ("R1", "R1/base_g100r30_tab.in", "parthenon/time/dt_force=1.1e-05 radiation/sn_fleck_alpha=1.0"),
     ("S1", "R3/S1_sou.in", "radiation/sn_sou_limiter=positive"),
     ("S3", "R3/S3_sou.in", "radiation/sn_sou_limiter=positive"),
     ("R5", "R5/a_P3_T1000.in", PC),
     ("R8a", "R8/plane.in", "parthenon/time/dt_force=0.002"),
     ("R8c", "R8/point.in", PC)]
for dev in ("cpu", "gpu"):
    open(os.path.join(INP, f"R6_{dev}_runs.txt"), "w").write(
        "".join(f"R6/{dev}/{n} | {d} | {o} | 1\n" for n, d, o in G))
print(len(G), "gates x 2 devices")
