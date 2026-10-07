#!/bin/bash
#SBATCH --job-name=sg_L1
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_b76b9e6b
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
for n in 40 80 160; do mb=$(( n/5 )); RS $W/L1_polytrope_atm/n$n 64 $I/L1_polytrope_atm.in $(mesh $n $n $n $mb); done; RS $W/L1_polytrope_atm/n80_100tau 64 $I/L1_polytrope_atm.in $(mesh 80 80 80 16) parthenon/time/tlim=70.710678 parthenon/output0/dt=3.5355339
echo SUITE-L1-DONE
