#!/bin/bash
#SBATCH --job-name=sg_L3b256
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
RC $W/L3_n5_collapse/n256 64 $I/L3_n5_collapse.in $(mesh 256 256 256 32) parthenon/output0/dt=0.02; RC $W/L3_n5_collapse/n128_dt0.02 64 $I/L3_n5_collapse.in parthenon/output0/dt=0.02
if [ $NEED = 1 ]; then echo "unfinished legs -> resubmitting"; sbatch $REPO/validation/self_gravity/runs/chain_L3b256.sh; else echo SUITE-L5b-DONE; fi
