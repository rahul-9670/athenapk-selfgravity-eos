#!/bin/bash
#SBATCH --job-name=sg_slab
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D4 advected self-gravitating slab: artemis-matched (rk2, cfl 0.9) and production integrator (vl2, cfl 0.3)
SA="self_gravity/solver_params/absolute_residual_tolerance=2.136e-7"   # 1e-6 rms(4piG drho), eps=0.3 (SG-F1)
for n in 16 32 64 128; do mb=$(( n/4 )); [ $mb -gt 32 ] && mb=32
  M="$(mesh $n $((n/2)) $((n/2)) $mb)"; NP=$(np_for $n $((n/2)) $((n/2)) $mb)
  # artemis CFL is cfl / sum_d (|v_d|+cs)/dx_d (artemis/src/gas/gas.cpp:575); AthenaPK takes the max over
  # directions. For v = (0.8,0.6,0), cs = sqrt(gamma P0/rho0) = 3.16: artemis 0.9 == AthenaPK 0.327.
  $R $W/D4_slab/rk2_cfl0.327_n$n $NP $I/grav_slab.in $M $SA parthenon/time/cfl=0.327
  $R $W/D4_slab/vl2_cfl0.3_n$n $NP $I/grav_slab.in $M $SA parthenon/time/integrator=vl2 parthenon/time/cfl=0.3
done
echo SUITE-SLAB-DONE
