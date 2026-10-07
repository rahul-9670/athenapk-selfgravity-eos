#!/bin/bash
#SBATCH --job-name=eos_E7qb
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# slab-mode pgen change (c5dc1c1f) must leave riemann/wave/compress bitwise: vs E7p (54b1958b, dflux) and EF56_regate.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_c5dc1c1f
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7q_bitid
$R $W/E7e_R4_ionization_n400_dflux 1 $I/E7e_R4_ionization_p_over_rho_n400.in hydro/double_flux=true &
$R $W/E7k_R4_ionization_mov_n400_dflux 1 $I/E7k_R4_ionization_mov_n400.in hydro/double_flux=true &
$R $W/bitid_E7e_R4_ionization_p_over_rho_n400 1 $I/E7e_R4_ionization_p_over_rho_n400.in &
$R $W/bitid_E6_wave_S3_n128 1 $I/E6_wave_S3_n128.in &
$R $W/bitid_E8_leg2 1 $I/E8_leg2.in &
wait; echo E7QB-DONE
