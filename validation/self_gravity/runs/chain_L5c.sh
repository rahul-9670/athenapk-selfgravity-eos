#!/bin/bash
#SBATCH --job-name=sg_L5c
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
RC $W/L5_evrard/box4_n128 64 $I/L5_evrard.in $(mesh 128 128 128 32) parthenon/mesh/x1min=-4 parthenon/mesh/x1max=4 parthenon/mesh/x2min=-4 parthenon/mesh/x2max=4 parthenon/mesh/x3min=-4 parthenon/mesh/x3max=4; RC $W/L5_evrard/box4_n256 64 $I/L5_evrard.in $(mesh 256 256 256 32) parthenon/mesh/x1min=-4 parthenon/mesh/x1max=4 parthenon/mesh/x2min=-4 parthenon/mesh/x2max=4 parthenon/mesh/x3min=-4 parthenon/mesh/x3max=4
if [ $NEED = 1 ]; then echo "unfinished legs -> resubmitting"; sbatch $REPO/validation/self_gravity/runs/chain_L5c.sh; else echo SUITE-L5b-DONE; fi
