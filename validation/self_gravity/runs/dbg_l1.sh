#!/bin/bash
#SBATCH --job-name=sg_dbgl1
#SBATCH --partition=std
#SBATCH --ntasks=64
#SBATCH --time=00:20:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
$R $W/debug_L1/n80_c1 64 $I/L1_polytrope_atm.in $(mesh 80 80 80 16) parthenon/time/nlim=1 parthenon/output0/dt=1e-12 parthenon/output0/single_precision_output=false
