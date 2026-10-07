#!/bin/bash
#SBATCH --job-name=sg_f1
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# SG-F1 fix verification with PRODUCTION keys only (no absolute floor, warm start): the legs that
# stalled into NaN / negative pressure on binary 7b6b6cd1. New binary: roundoff_floor (swindle only)
# + safe_fallback. Last leg: floor forced OFF to exercise the fallback on the original failure.
P1="problem/grav_analytic/profile=sinusoid problem/grav_analytic/rho_bg=2 problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172"
RS $W/F1_fix/S1_n16_prod 8 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P1
RS $W/F1_fix/S1_n64_prod 64 $I/static_poisson.in $(mesh 64 64 64 16) $cube01 $periodic $P1
for n in 64 128; do
  RS $W/F1_fix/jeans_nJ0.5_n${n}_prod 64 $I/jeans_wave.in $(mesh $n $((n/2)) $((n/2)) 16) self_gravity/four_pi_G=88.82643960980423 parthenon/time/tlim=0.3849001794597505 parthenon/output0/dt=0.019245
  RS $W/F1_fix/jeans_nJ0.5_n${n}_zeroguess 64 $I/jeans_wave.in $(mesh $n $((n/2)) $((n/2)) 16) self_gravity/four_pi_G=88.82643960980423 parthenon/time/tlim=0.3849001794597505 parthenon/output0/dt=0.019245 self_gravity/solver_params/seed_from_phi=false self_gravity/solver_params/zero_initial_guess=true
done
RS $W/F1_fix/slab_n32_prod 16 $I/grav_slab.in $(mesh 32 16 16 8) parthenon/time/integrator=rk2 parthenon/time/cfl=0.327
RS $W/F1_fix/dbg_g88_np8 8 $I/jeans_wave.in $(mesh 32 16 16 8) self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=40
RS $W/F1_fix/dbg_g88_np8_floor0 8 $I/jeans_wave.in $(mesh 32 16 16 8) self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=40 self_gravity/solver_params/roundoff_floor=0
echo SUITE-F1-DONE
