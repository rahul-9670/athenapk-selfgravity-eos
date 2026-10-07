#!/bin/bash
#SBATCH --job-name=sg_L2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_b76b9e6b
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
for n in 32 64 128 256 512; do mb=$(( n<32 ? n : 32 )); nb=$(( n/mb )); RS $W/L2_spitzer/n$n $nb $I/L2_spitzer.in parthenon/mesh/nx1=$n parthenon/meshblock/nx1=$mb parthenon/mesh/x2max=$(python3 -c "print(32/$n)") parthenon/mesh/x3max=$(python3 -c "print(32/$n)"); done
echo SUITE-L2-DONE
