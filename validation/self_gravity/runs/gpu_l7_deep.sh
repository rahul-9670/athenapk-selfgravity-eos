#!/bin/bash
#SBATCH --job-name=sg_gpuL7deep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# Deep-AMR GPU check: restart the GPU binary from the CPU L7 nj8 restart at cycle 1500 (levels 0-13,
# rho_c 1.3e11 code) and run to cycle 1600; compare with CPU dump parthenon.out0.00200.phdf (cycle 1600).
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
BIN=$REPO/validation/_build/bin/athenaPK_gpu_ee6c11d3
D=$REPO/validation/self_gravity/runs/GPU/L7_nj8_deep_c1500
cd $D
{ echo "job $SLURM_JOB_ID $(date) host $(hostname)"; md5sum $BIN start_c1500.rhdf; nvidia-smi -L; } > cmd.txt
mpirun -n 1 --mca mtl ^psm2 --mca btl tcp,self,sm $BIN -r start_c1500.rhdf parthenon/time/nlim=1600 parthenon/output0/dn=8 > run.log 2>&1
echo "rc $?" >> cmd.txt
