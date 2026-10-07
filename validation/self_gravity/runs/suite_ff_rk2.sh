#!/bin/bash
#SBATCH --job-name=sg_ff_rk2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D7 pressure-free collapse (WP-15 re-run, production solver keys, sub-sampled sphere)
# SG-F4 bisection (temporal order per integrator), one integrator per job (concurrent runs oversubscribed).
for dc in 0.002 0.001; do RS $W/D7_freefall/n64_rk2_dtceil$dc 64 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/integrator=rk2 parthenon/time/dt_ceil=$dc; done
echo SUITE-FF-rk2-DONE
