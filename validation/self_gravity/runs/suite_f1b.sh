#!/bin/bash
#SBATCH --job-name=sg_f1b
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_97a07fdd
for v in prod zeroguess; do X=""; [ $v = zeroguess ] && X="self_gravity/solver_params/seed_from_phi=false self_gravity/solver_params/zero_initial_guess=true"
  $R $W/F1_fix/jeans_nJ0.5_n64_$v 16 $I/jeans_wave.in $(mesh 64 32 32 16) self_gravity/four_pi_G=88.82643960980423 parthenon/time/tlim=0.3849001794597505 parthenon/output0/dt=0.019245 $X
done
