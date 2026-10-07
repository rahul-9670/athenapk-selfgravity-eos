#!/bin/bash
#SBATCH --job-name=eos_EF5T
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E-F5 physical floor: hydro/Tfloor = 0.2726 code (= 2.73 K, CMB) with the extrapolating EOS; cons-nTfloor column.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_04db22ca
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/EF5_tfloor
for t in control negative subedge; do
  $R $W/E10_probe_${t}_fctrue_Tfloor 1 $I/E10_probe_${t}_fctrue.in hydro/Tfloor=0.2726 &
done
wait; echo EF5T-DONE
