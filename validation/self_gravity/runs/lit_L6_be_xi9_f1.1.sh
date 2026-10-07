#!/bin/bash
#SBATCH --job-name=sg_L6_be_xi9_f1.1
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_b76b9e6b
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
RS $W/L6_be/xi9_f1.1 64 $I/L6_be_xi9_f1.1.in
echo SUITE-L6_be_xi9_f1.1-DONE
