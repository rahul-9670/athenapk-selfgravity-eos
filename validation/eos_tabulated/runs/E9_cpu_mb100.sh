#!/bin/bash
#SBATCH --job-name=eos_E9cpu
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9 CPU reference: E7e N = 400 decks as 4 meshblocks of 100 (the GPU decomposition), binary a6870124
# (same source as the GPU rebuild; E7 code path identical to 890d7f56).
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_a6870124
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9_cpu_mb100
for d in $I/E7e_*_n400.in; do $R $W/$(basename $d .in)_mb100 1 $d parthenon/meshblock/nx1=100 & done; wait
# same decks, ONE block, this binary (decomposition + binary check vs runs/E7e)
for d in $I/E7e_*_n400.in; do $R $W/$(basename $d .in)_mb400 1 $d & done; wait
echo E9CPU-DONE
