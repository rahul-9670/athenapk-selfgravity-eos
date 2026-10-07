#!/bin/bash
#SBATCH --job-name=eos_E7q
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=104
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7q: double flux in 2D/3D -- periodic oblique slabs of pure contacts (R4 star states), key on and off;
# decomposition invariance (mb32 / mb16 vs one block) and path neutrality (k = (1,0) on a 2D grid vs 1D).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_fe0caa0f
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7q
for d in $I/E7q_*.in; do n=$(basename $d .in)
  case $n in *3d_n128|*2d_n512) np=16;; *3d_n64_mb16) np=8;; *mb32) np=4;; *) np=1;; esac
  $R $W/${n}_dflux $np $d hydro/double_flux=true &
  $R $W/${n}_std $np $d &
done
wait; echo E7Q-DONE
