#!/bin/bash
#SBATCH --job-name=sg_L9_nj8
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_b76b9e6b
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
RS $W/L9_boss_bodenheimer/nj8 64 $I/L9_boss_bodenheimer_nj8.in
echo SUITE-L9_nj8-DONE
