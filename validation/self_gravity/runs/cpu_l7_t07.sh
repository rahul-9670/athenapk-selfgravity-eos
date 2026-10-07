#!/bin/bash
#SBATCH --job-name=sg_cpuL7t07
#SBATCH --partition=std
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# CPU twin of GPU/L7_nj8_t0.7 (same deck, same CLI) for a matched deep-AMR GPU/CPU comparison.
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
$R $W/L7_cpu_t0.7/nj8 64 $I/L7_be_collapse_nj8.in parthenon/time/tlim=0.7
