#!/bin/bash
#SBATCH --job-name=sg_fforder
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D7 pressure-free collapse (WP-15 re-run, production solver keys, sub-sampled sphere)
# SG-F4 bisection: temporal order of the pressure-free collapse per integrator (vl2 measured: order 1.00-1.09).
# rk1 must give 1; if rk2/rk3 also give 1 the gravity-hydro coupling (not VL2) limits the order.
NPMAX=16
for integ in rk1 rk2 rk3; do
  ( for dc in 0.002 0.001; do RS $W/D7_freefall/n64_${integ}_dtceil$dc 16 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/integrator=$integ parthenon/time/dt_ceil=$dc; done ) &
done
wait
echo SUITE-FFORDER-DONE
