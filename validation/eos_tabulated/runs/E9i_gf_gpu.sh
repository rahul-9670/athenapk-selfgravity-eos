#!/bin/bash
#SBATCH --job-name=eos_E9i
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9i: double flux + transported-pressure fill (sensor-gated) on the GPU ($1 = GPU build of the source of CPU 85b625bf).
# E7r refinement decks; CPU references: runs/E7w4/gf.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
export BIN=$1
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9i_gpu
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
for n in E7r_R4_mov_1d_smr E7r_R4_mov_1d_amr E7r_slab11_2d_smr E7r_slab11_2d_amr; do
  $R $W/${n}_dflux 1 $I/$n.in hydro/double_flux=true
done
echo E9I-DONE
