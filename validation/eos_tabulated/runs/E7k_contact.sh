#!/bin/bash
#SBATCH --job-name=eos_E7k
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7k: pure contact between EOS regimes (uniform P, u exact), PLM vs LimO3. Final default binary.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_a15514d6
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7k
for d in $I/E7k_*.in; do n=$(basename $d .in)
  $R $W/${n}_plm 1 $d &
  $R $W/${n}_limo3 1 $d hydro/reconstruction=limo3 parthenon/mesh/nghost=3 &
done
wait; echo E7k-DONE
