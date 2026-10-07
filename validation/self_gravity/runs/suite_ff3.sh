#!/bin/bash
#SBATCH --job-name=sg_ff3
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D7 pressure-free collapse (WP-15 re-run, production solver keys, sub-sampled sphere)
# Temporal order of the gravity-hydro coupling: dt_ceil ladder at 64^3 (0.002 and 0.0005 already run).
RS $W/D7_freefall/n64_dtceil0.001 64 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/dt_ceil=0.001
RS $W/D7_freefall/n64_dtceil0.00025 64 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/dt_ceil=0.00025
echo SUITE-FF3-DONE
