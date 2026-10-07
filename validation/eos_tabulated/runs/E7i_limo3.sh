#!/bin/bash
#SBATCH --job-name=eos_E7i
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7i: hydro/reconstruction = limo3 (nghost 3) on R2 (N 800-6400), R1 and R3 (N 1600, 3200): no-regression check
# of the reconstruction that removed the R4 slow convergence (E7h).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_23f9e39e
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7i
L="hydro/reconstruction=limo3 parthenon/mesh/nghost=3"
for N in 800 1600 3200 6400; do
  $R $W/R2_limo3_n$N 1 $I/E7e_R2_dissociation_p_over_rho_n3200.in parthenon/mesh/nx1=$N parthenon/meshblock/nx1=$N $L &
done
for N in 1600 3200; do
  $R $W/R1_limo3_n$N 1 $I/E7e_R1_molecular_p_over_rho_n$N.in $L &
  $R $W/R3_limo3_n$N 1 $I/E7e_R3_dissoc_shock_p_over_rho_n$N.in $L &
done
wait; echo E7i-DONE
