#!/bin/bash
#SBATCH --job-name=eos_EF56r
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# Key-off regate of the double-flux build 54b1958b vs EF56_final (a15514d6): every output must be bitwise equal.
#
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_54b1958b
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/EF56_regate
for d in $I/E10_probe_*.in; do n=$(basename $d .in)
  $R $W/${n}_fix 1 $d &
done
for d in $I/E7e_*_n400.in $I/E6_wave_S?_n128.in $I/E8_leg2.in; do $R $W/bitid_$(basename $d .in) 1 $d & done
wait
echo EF56R-DONE
