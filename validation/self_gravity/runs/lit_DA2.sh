#!/bin/bash
#SBATCH --job-name=sg_DA2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_b76b9e6b
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
RS $W/DA2_freefall_deep/root64_L4 64 $I/DA2_freefall_deep.in
echo SUITE-DA2-DONE
