#!/bin/bash
#SBATCH --job-name=eos_EF56f
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E-F5/E-F6 fix (binary a15514d6 (final default), hydro/eos_extrapolate default true): E10 probe with the fix and with the key off
# (must reproduce runs/E10 bitwise), and in-table bit-identity legs vs the a6870124/890d7f56 runs.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_a15514d6
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/EF56_final
for d in $I/E10_probe_*.in; do n=$(basename $d .in)
  $R $W/${n}_fix 1 $d &
  $R $W/${n}_noext 1 $d hydro/eos_extrapolate=false &
done
for d in $I/E7e_*_n400.in $I/E6_wave_S?_n128.in $I/E8_leg2.in; do $R $W/bitid_$(basename $d .in) 1 $d & done
wait
echo EF56-DONE
