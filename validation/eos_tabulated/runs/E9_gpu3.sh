#!/bin/bash
#SBATCH --job-name=eos_E9gpu3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9 part 3 (pinned GPU 4d8a6115 = source of CPU a6870124): E10 probes (thin 3D) + E7e N=400 as 4 x 100 blocks.
# (validation/physics-tests 4bf02bd7 + _build/bin/athenaPK_cpu_890d7f56.src.patch). CPU references: runs/E7,E8,E10,E6cpu.
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
G=$REPO/validation/_build/bin/athenaPK_gpu_4d8a6115

export BIN=$G
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9_gpu
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
for d in $I/E10_probe_*.in; do
  $R $W/$(basename $d .in) 1 $d
done
# 1D meshblocks of 400 exceed the CUDA team-scratch limit ("could not find a valid execution configuration"):
# E7e at N = 400 as 4 blocks of 100 (CPU reference with the same decomposition: runs/E9_cpu_mb100)
for d in $I/E7e_*_n400.in; do
  $R $W/$(basename $d .in)_mb100 1 $d parthenon/meshblock/nx1=100
done
echo E9-GPU-DONE $(date)
