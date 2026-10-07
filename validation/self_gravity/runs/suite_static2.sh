#!/bin/bash
#SBATCH --job-name=sg_static2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-S1 periodic sinusoid (Tomida & Stone 2023 Sec 4.1; Athena++ poisson iprob 1): G = 1
# SG-S1 corrected legs (finding SG-F1): production keys + absolute floor 1e-6 rms(rhs) = 4.44e-6
# (rms(4 pi G A sss) = 4 pi / sqrt 8); tolerance sweep with a COLD start so residual_tolerance is
# relative to rms(rhs), as in Tomida and Stone 2023 Fig. 5.
P="problem/grav_analytic/profile=sinusoid problem/grav_analytic/rho_bg=2 problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172"
for n in 16 32 64 128 256; do mb=$(( n<64 ? 8 : 16 )); [ $n -ge 128 ] && mb=32
  $R $W/S1_sinusoid/n${n}_prodabs $(np_for $n $n $n $mb) $I/static_poisson.in $(mesh $n $n $n $mb) $cube01 $periodic $P self_gravity/solver_params/absolute_residual_tolerance=4.44e-6
done
for tol in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-10 1e-12; do
  $R $W/S1_sinusoid/cold_n128_$tol 64 $I/static_poisson.in $(mesh 128 128 128 32) $cube01 $periodic $P \
    self_gravity/solver_params/residual_tolerance=$tol self_gravity/solver_params/max_iterations=1000 \
    self_gravity/solver_params/seed_from_phi=false self_gravity/solver_params/zero_initial_guess=true self_gravity/solver_params/print_per_step=true
done
echo SUITE-STATIC2-DONE
