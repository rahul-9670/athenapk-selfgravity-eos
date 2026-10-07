#!/bin/bash
#SBATCH --job-name=eos_E7n
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7n: hydro/double_flux = true WITH the shock sensor (double_flux_shock_dp 0.1), contacts + R1-R4.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_783a5a10
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7n
for p in R4_ionization R2_dissociation R1_molecular; do for N in 400 800 1600 3200 6400; do
  $R $W/E7k_${p}_mov_n${N}_dflux 1 $I/E7k_${p}_mov_n$N.in hydro/double_flux=true &
done; done
for p in R4_ionization R2_dissociation R1_molecular R3_dissoc_shock; do for N in 400 800 1600 3200 6400; do
  if [ $N -le 3200 ]; then D=$I/E7e_${p}_p_over_rho_n$N.in; X=""; else D=$I/E7e_${p}_p_over_rho_n3200.in; X="parthenon/mesh/nx1=$N parthenon/meshblock/nx1=$N"; fi
  [ $p = R3_dissoc_shock ] && [ $N -gt 3200 ] && continue
  $R $W/E7e_${p}_n${N}_dflux 1 $D $X hydro/double_flux=true &
done; done
wait; echo E7m-DONE
