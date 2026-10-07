#!/bin/bash
#SBATCH --job-name=sg_L5b
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
RC $W/L5_evrard/n256 64 $I/L5_evrard.in $(mesh 256 256 256 32)
if [ $NEED = 1 ]; then echo "unfinished legs -> resubmitting"; sbatch $REPO/validation/self_gravity/runs/chain_L5b.sh; else echo SUITE-L5b-DONE; fi
