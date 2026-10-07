#!/bin/bash
#SBATCH --job-name=sg_ff2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D7 pressure-free collapse (WP-15 re-run, production solver keys, sub-sampled sphere)
# SG-F3 follow-up: rho_c error +0.42 % at t = 1.5 is resolution-independent (64-256). Candidates:
# (a) time-step error (dt_ceil = 0.002 at every N), (b) multipole shell/uniform-split BC floor.
RS $W/D7_freefall/n64_dtceil0.0005 64 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/dt_ceil=0.0005
RS $W/D7_freefall/n64_nosplit 64 $I/freefall.in $(mesh 64 64 64 16) self_gravity/mp_shell_split=false self_gravity/mp_uniform_split=false
echo SUITE-FF2-DONE
