#!/bin/bash
#SBATCH --job-name=eos_E9gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9: the EOS decks on one H100, binary built from the SAME source as CPU 890d7f56
# (validation/physics-tests 4bf02bd7 + _build/bin/athenaPK_cpu_890d7f56.src.patch). CPU references: runs/E7,E8,E10,E6cpu.
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
G=$REPO/build_gpu/bin/athenaPK
M=$(md5sum $G | cut -c1-8); cp -p $G $REPO/validation/_build/bin/athenaPK_gpu_$M
export BIN=$REPO/validation/_build/bin/athenaPK_gpu_$M
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9_gpu
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
for d in $I/E10_probe_*.in $I/E7_*_n400.in $I/E6_wave_S?_n128.in $I/E8_leg2.in $I/E8_leg3.in; do
  $R $W/$(basename $d .in) 1 $d
done
echo E9-GPU-DONE $(date)
