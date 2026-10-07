#!/bin/bash
#SBATCH --job-name=sg_dbgl8hc
#SBATCH --partition=std
#SBATCH --ntasks=64
#SBATCH --time=00:40:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
$R $W/debug_L8hc/to0.11 64 $I/L8_sis_heavycore.in parthenon/time/tlim=0.11 parthenon/output0/dn=1 parthenon/output0/dt=-1 parthenon/output0/variables=prim,grav.phi,cons parthenon/output0/single_precision_output=false
