#!/bin/bash
#SBATCH --job-name=sg_ff
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D7 pressure-free collapse (WP-15 re-run, production solver keys, sub-sampled sphere)
for n in 64 128 256; do mb=$(( n<=64 ? 16 : 32 ))
  $R $W/D7_freefall/n$n $(np_for $n $n $n $mb) $I/freefall.in $(mesh $n $n $n $mb)
done
# stair-step control (nsub = 1 = WP-15's IC) at 64^3: isolates the surface-sampling error
$R $W/D7_freefall/n64_nsub1 64 $I/freefall.in $(mesh 64 64 64 16) problem/grav_analytic/nsub=1
echo SUITE-FF-DONE
