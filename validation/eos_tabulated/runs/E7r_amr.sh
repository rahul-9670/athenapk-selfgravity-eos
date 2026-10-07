#!/bin/bash
#SBATCH --job-name=eos_E7r
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7r: double flux with mesh refinement (coarse cells at fine/coarse faces are standard cells): R4 moving contact
# in 1D through a static patch and with adaptive refinement; oblique 2D slab with SMR and AMR. Key on and off.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_fe0caa0f
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7r
for d in $I/E7r_*.in; do n=$(basename $d .in)
  case $n in *2d*) np=4;; *) np=1;; esac
  $R $W/${n}_dflux $np $d hydro/double_flux=true &
  $R $W/${n}_std $np $d &
done
wait; echo E7R-DONE
