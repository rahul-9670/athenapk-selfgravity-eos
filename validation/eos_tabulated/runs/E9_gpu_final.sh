#!/bin/bash
#SBATCH --job-name=eos_E9f
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E9 for the E-F5/E-F6 fix (final source; GPU binary from the same source as CPU a15514d6): the floor probe with the
# fix (all out-of-line helpers on CUDA) + in-table legs. CPU references: runs/EF56_final, runs/E9_cpu_mb100.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
export BIN=$1
R=$REPO/validation/self_gravity/scripts/run_one.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E9_gpu_final
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
for d in $I/E10_probe_*.in; do $R $W/$(basename $d .in)_fix 1 $d; done
$R $W/E7e_R4_ionization_p_over_rho_n400_mb100 1 $I/E7e_R4_ionization_p_over_rho_n400.in parthenon/meshblock/nx1=100
$R $W/E8_leg2 1 $I/E8_leg2.in
echo E9F-DONE
