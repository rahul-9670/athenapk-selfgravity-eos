#!/bin/bash
#SBATCH --job-name=sg_L7_nj8_rk2
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
RC $W/L7_be_collapse/nj8_rk2 64 $I/L7_be_collapse_nj8.in parthenon/time/integrator=rk2 parthenon/time/tlim=1.03 parthenon/output0/dn=8 parthenon/output0/variables=prim
if [ $NEED = 1 ]; then echo "unfinished legs -> resubmitting"; sbatch $REPO/validation/self_gravity/runs/chain_L7_nj8_rk2.sh; else echo SUITE-L7_nj8_rk2-DONE; fi
