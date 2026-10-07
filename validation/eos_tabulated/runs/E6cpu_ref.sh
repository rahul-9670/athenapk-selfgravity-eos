#!/bin/bash
#SBATCH --job-name=eos_E6ref
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9 CPU reference for the E6 n128 decks with the CURRENT binary (E6 itself ran on 39c0d06a, default eos_interp = p).
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_890d7f56
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E6cpu
for d in $I/E6_wave_S?_n128.in; do $R $W/$(basename $d .in) 1 $d & done; wait
echo E6REF-DONE
