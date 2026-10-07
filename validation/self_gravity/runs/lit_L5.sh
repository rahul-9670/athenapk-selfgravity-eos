#!/bin/bash
#SBATCH --job-name=sg_L5
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_b76b9e6b
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
for n in 64 128 256; do mb=$(( n<=64 ? 16 : 32 )); RS $W/L5_evrard/n$n 64 $I/L5_evrard.in $(mesh $n $n $n $mb); done
echo SUITE-L5-DONE
