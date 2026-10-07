#!/bin/bash
#SBATCH --job-name=eos_E7f
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7f: why do R2/R4 (rarefaction through recombination) converge at order ~0.5-0.7 in the plateau while R1/R3 converge
# fast? One change per leg on R4 N = 1600 (p_over_rho), plus N = 6400/12800 for R2 and R4.
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_a6870124
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7f
D4=$I/E7e_R4_ionization_p_over_rho_n1600.in
$R $W/R4_n1600_rk2 1 $D4 parthenon/time/integrator=rk2 &
$R $W/R4_n1600_dcpred 1 $D4 hydro/vl2_predictor_recon=dc &
$R $W/R4_n1600_bis60 1 $D4 hydro/eos_bisections=60 &
$R $W/R4_n1600_cfl01 1 $D4 parthenon/time/cfl=0.1 &
$R $W/R4_n1600_hlle 1 $D4 hydro/riemann=hlle &
$R $W/R4_n1600_ref 1 $D4 &
for N in 6400 12800; do
  for P in R2_dissociation R4_ionization; do
    $R $W/${P}_n$N 1 $I/E7e_${P}_p_over_rho_n3200.in parthenon/mesh/nx1=$N parthenon/meshblock/nx1=$N &
  done
done
wait
echo E7f-DONE
