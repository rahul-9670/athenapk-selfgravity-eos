#!/bin/bash
#SBATCH --job-name=sg_poly
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D5 static n=1 polytrope (production-like box, multipole BC) ; SG-D6 offset collision
for n in 32 64 128; do mb=$(( n<=64 ? 16 : 32 ))
  $R $W/D5_polytrope/n${n}_mp $(np_for $n $n $n $mb) $I/polytrope.in $(mesh $n $n $n $mb)
done
$R $W/D5_polytrope/artemis_static_amr 64 $I/polytrope_artemis.in $(mesh 64 64 64 16)
$R $W/D6_collision/n128_box16 64 $I/polytrope.in $(mesh 128 128 128 16) problem/polytrope/iprob=2 \
  problem/polytrope/x10a=4 problem/polytrope/x20a=2.5 parthenon/mesh/x1min=-16 parthenon/mesh/x1max=16 \
  parthenon/mesh/x2min=-16 parthenon/mesh/x2max=16 parthenon/mesh/x3min=-16 parthenon/mesh/x3max=16 \
  parthenon/time/tlim=10 parthenon/output0/dt=1.0
echo SUITE-POLY-DONE
