#!/bin/bash
#SBATCH --job-name=eos_E7h
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7h: the R4 plateau error is NOT from the table (E7c/E7d/E7g). Scheme bisection, one change per variant,
# R4 (p_over_rho, bilinear) N = 800..6400: dc; ppm, wenoz, limo3 (nghost 3); plm + dc predictor; plm + rk3.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_23f9e39e
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7h
D=$I/E7e_R4_ionization_p_over_rho_n3200.in
for N in 800 1600 3200 6400; do
  M="parthenon/mesh/nx1=$N parthenon/meshblock/nx1=$N"
  $R $W/R4_dc_n$N 1 $D $M hydro/reconstruction=dc &
  $R $W/R4_ppm_n$N 1 $D $M hydro/reconstruction=ppm parthenon/mesh/nghost=3 &
  $R $W/R4_wenoz_n$N 1 $D $M hydro/reconstruction=wenoz parthenon/mesh/nghost=3 &
  $R $W/R4_limo3_n$N 1 $D $M hydro/reconstruction=limo3 parthenon/mesh/nghost=3 &
  $R $W/R4_plmdcpred_n$N 1 $D $M hydro/vl2_predictor_recon=dc &
  $R $W/R4_plmrk3_n$N 1 $D $M parthenon/time/integrator=rk3 &
done
wait; echo E7h-DONE
