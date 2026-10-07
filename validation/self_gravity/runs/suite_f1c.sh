#!/bin/bash
#SBATCH --job-name=sg_f1c
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
for n in 64 128; do NP=$(( n==64 ? 16 : 64 ))
  for v in prod zeroguess; do X=""; [ $v = zeroguess ] && X="self_gravity/solver_params/seed_from_phi=false self_gravity/solver_params/zero_initial_guess=true"
    $R $W/F1_fix_sqrtN/jeans_nJ0.5_n${n}_$v $NP $I/jeans_wave.in $(mesh $n $((n/2)) $((n/2)) 16) self_gravity/four_pi_G=88.82643960980423 parthenon/time/tlim=0.3849001794597505 parthenon/output0/dt=0.019245 $X
  done; done
$R $W/F1_fix_sqrtN/dbg_g88_np8 8 $I/jeans_wave.in $(mesh 32 16 16 8) self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=40
P1="problem/grav_analytic/profile=sinusoid problem/grav_analytic/rho_bg=2 problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172"
for n in 16 64 256; do mb=$(( n<64 ? 8 : 16 )); [ $n -ge 128 ] && mb=32; $R $W/F1_fix_sqrtN/S1_n${n}_prod $(np_for $n $n $n $mb) $I/static_poisson.in $(mesh $n $n $n $mb) $cube01 $periodic $P1; done
