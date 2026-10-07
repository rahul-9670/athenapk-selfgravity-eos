#!/bin/bash
#SBATCH --job-name=eos_E7y2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7y2: key-off regate of 85b625bf vs a15514d6: 20 1D legs.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_85b625bf
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs
for d in $I/E10_probe_*.in; do $R $W/EF56_regate_v8/$(basename $d .in)_fix 1 $d & done
for d in $I/E7e_*_n400.in $I/E6_wave_S?_n128.in $I/E8_leg2.in; do $R $W/EF56_regate_v8/bitid_$(basename $d .in) 1 $d & done
wait; echo E7Y2-DONE
