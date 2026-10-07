#!/bin/bash
#SBATCH --job-name=eos_E7u
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# 37c90051 (nvcc capture fix only) must reproduce fe0caa0f bitwise with double flux on.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_37c90051
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7u_bitid37
$R $W/E7k_R4_ionization_mov_n400_dflux 1 $I/E7k_R4_ionization_mov_n400.in hydro/double_flux=true &
$R $W/E7e_R4_ionization_n400_dflux 1 $I/E7e_R4_ionization_p_over_rho_n400.in hydro/double_flux=true &
$R $W/E7q_slab11_2d_n64_dflux 1 $I/E7q_slab11_2d_n64.in hydro/double_flux=true &
$R $W/E7r_R4_mov_1d_smr_dflux 1 $I/E7r_R4_mov_1d_smr.in hydro/double_flux=true &
wait; echo E7U-DONE
