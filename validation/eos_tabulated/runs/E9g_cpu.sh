#!/bin/bash
#SBATCH --job-name=eos_E9gc
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# CPU references for E9g (fe0caa0f, same decks/overrides, 1 rank each).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_fe0caa0f
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9g_cpu
$R $W/E7k_R4_mov_n400_mb100 1 $I/E7k_R4_ionization_mov_n400.in parthenon/meshblock/nx1=100 hydro/double_flux=true &
$R $W/E7e_R4_n400_mb100 1 $I/E7e_R4_ionization_p_over_rho_n400.in parthenon/meshblock/nx1=100 hydro/double_flux=true &
$R $W/E7q_slab11_2d_n128 1 $I/E7q_slab11_2d_n128.in hydro/double_flux=true &
$R $W/E7q_slab111_3d_n64 1 $I/E7q_slab111_3d_n64.in hydro/double_flux=true &
$R $W/E7r_R4_mov_1d_amr 1 $I/E7r_R4_mov_1d_amr.in hydro/double_flux=true &
wait; echo E9GC-DONE
