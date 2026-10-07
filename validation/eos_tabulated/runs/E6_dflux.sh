#!/bin/bash
#SBATCH --job-name=eos_E6d
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# Falsification of the double-flux contact sensor: smooth acoustic waves (E6, states S1-S5, N 16-256) with
# hydro/double_flux=true must reproduce the standard runs (runs/E6_wave) -- the sensor must not fire in sound waves.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_54b1958b
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E6_dflux
for s in S1 S2 S3 S4 S5; do for N in 16 32 64 128 256; do
  $R $W/E6_wave_${s}_n$N 1 $I/E6_wave_${s}_n$N.in hydro/double_flux=true &
  $R $W/std_E6_wave_${s}_n$N 1 $I/E6_wave_${s}_n$N.in &
done; done
wait; echo E6D-DONE
