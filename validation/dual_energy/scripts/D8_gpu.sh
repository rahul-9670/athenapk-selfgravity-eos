#!/bin/bash
#SBATCH --job-name=de_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/dual_energy/runs/%x_%j.out
# D8: the GPU binary athenaPK_gpu_07357f75 (same source as CPU 4b5393c2, identical .src.patch) on a subset of the CPU
# decks/overrides (inputs/D8_gpu_runs.txt), sequentially on one H100. Long 1D blocks run as 64-cell meshblocks:
# the double-flux coefficient kernel's level-0 scratch (3 x nvar x (nx1+4) doubles) does not launch for nx1 >~ 200.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
export BIN=$REPO/validation/_build/bin/athenaPK_gpu_07357f75
M=$REPO/validation/dual_energy
R1=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
while IFS='|' read -r d k o n; do
  d=$(echo $d); k=$(echo $k); o=$(echo $o)
  $R1 $M/runs/$d 1 $M/inputs/$k $o < /dev/null   # mpirun must not eat the loop's stdin (only the first line ran)
done < $M/inputs/D8_gpu_runs.txt
echo D8-GPU-DONE
