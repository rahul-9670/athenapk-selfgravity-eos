#!/bin/bash
#SBATCH --job-name=sg_L9hang
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# Reproduce the L9 nj4 GMG restriction-receive hang (job 2956209 aborted at cycle 1439 after
# "MPI probably hanging after 1e9 receive tries", ReceiveBoundBufs<gmg_restrict_recv>).
# Same binary, same 64 ranks, same restart (cycle 1000); restarts every 100 cycles to bisect later.
source $REPO/validation/self_gravity/scripts/suite_common.sh
BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
cd $REPO/validation/self_gravity/runs/L9_boss_bodenheimer/_repro_hang_nj4
{ echo "job $SLURM_JOB_ID $(date) host $(hostname)"; md5sum $BIN start_c1000.rhdf; } > cmd.txt
mpirun -n 64 --bind-to core $BIN -r start_c1000.rhdf -t 01:50:00 parthenon/time/nlim=1460 parthenon/output2/file_type=rst parthenon/output2/dn=100 > run.log 2>&1
echo "rc $?" >> cmd.txt
