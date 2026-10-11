#!/bin/bash
#SBATCH --job-name=rad_hang
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/hang_%j.out
# R0 SEAM hang bisection (2026-10-08): the static-AMR seam arms (16^3 root, 4^3 meshblocks, nghost 2) spin forever in
# parthenon::SendBoundBufsWithRestrictOption after the first Sn solve. nlim = 3, 300 s timeout per variant.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
ulimit -c 0
M=$REPO/validation/radiation_sn; W=$M/runs/R0/HANG; D=$M/inputs/R0/SEAM_step.in
BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_4135083d}
C="parthenon/time/nlim=3 radiation/sn_solver=bicgstab"
v () { name=$1; np=$2; launch=$3; shift 3; d=$W/$name; mkdir -p $d; cd $d; rm -f *.phdf *.xdmf *.hst run.log
  T0=$(date +%s)
  if [ "$launch" = srun ]; then timeout 300 srun --mpi=pmix -n $np $BIN -i $D $C "$@" > run.log 2>&1; else timeout 300 mpirun -n $np --bind-to none $BIN -i $D $C "$@" > run.log 2>&1; fi
  rc=$?; echo "$name np=$np $launch rc=$rc wall=$(( $(date +%s)-T0 ))s cycles=$(grep -ac '^cycle=' run.log) solves=$(grep -ac 'TRUE rel' run.log) :: $*"; }
v e1_np8_mpirun 8 mpirun
v e2_np1 1 mpirun
v e3_mb8 8 mpirun parthenon/meshblock/nx1=8 parthenon/meshblock/nx2=8 parthenon/meshblock/nx3=8
v e4_np8_srun 8 srun
v e5_coal_on 8 mpirun parthenon/mesh/do_coalesced_comms=true
v e6_coal_off 8 mpirun parthenon/mesh/do_coalesced_comms=false
v e7_norad 8 mpirun physics/radiation=false
v e8_ng3 8 mpirun parthenon/mesh/nghost=3
echo HANG-DONE
