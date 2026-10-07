#!/bin/bash
#SBATCH --job-name=sg_L5a
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
C=$REPO/validation/self_gravity/scripts/run_chain.sh; NEED=0
RC() { $C "$@"; rc=$?; [ $rc -eq 99 ] && NEED=1; return 0; }
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
RC $W/L5_evrard/n64 64 $I/L5_evrard.in $(mesh 64 64 64 16); RC $W/L5_evrard/n128 64 $I/L5_evrard.in $(mesh 128 128 128 32)
if [ $NEED = 1 ]; then echo "unfinished legs -> resubmitting"; sbatch $REPO/validation/self_gravity/runs/chain_L5a.sh; else echo SUITE-L5a-DONE; fi
