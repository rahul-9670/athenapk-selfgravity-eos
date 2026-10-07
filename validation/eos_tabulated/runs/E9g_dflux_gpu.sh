#!/bin/bash
#SBATCH --job-name=eos_E9g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9g: double flux v6 on the GPU (binary $1 = GPU build of the source of CPU fe0caa0f): 1D contact + R4 (mb100),
# 2D oblique slab, 3D slab, 1D AMR contact; key on. CPU references: runs/E9g_cpu (same decks/overrides).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
export BIN=$1
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9g_gpu
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
$R $W/E7k_R4_mov_n400_mb100 1 $I/E7k_R4_ionization_mov_n400.in parthenon/meshblock/nx1=100 hydro/double_flux=true
$R $W/E7e_R4_n400_mb100 1 $I/E7e_R4_ionization_p_over_rho_n400.in parthenon/meshblock/nx1=100 hydro/double_flux=true
$R $W/E7q_slab11_2d_n128 1 $I/E7q_slab11_2d_n128.in hydro/double_flux=true
$R $W/E7q_slab111_3d_n64 1 $I/E7q_slab111_3d_n64.in hydro/double_flux=true
$R $W/E7r_R4_mov_1d_amr 1 $I/E7r_R4_mov_1d_amr.in hydro/double_flux=true
echo E9G-DONE
