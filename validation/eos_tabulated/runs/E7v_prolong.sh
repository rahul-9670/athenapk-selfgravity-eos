#!/bin/bash
#SBATCH --job-name=eos_E7v
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7v: isolate the residual double-flux contact error on refined meshes: piecewise-constant (pc) vs minmod (mm)
# prolongation of cons (hydro/cons_prolong_pc), E7r decks, key on/off. mm runs must reproduce runs/E7r bitwise.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_a5cd4e41
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7v
for n in E7r_R4_mov_1d_smr E7r_R4_mov_1d_amr E7r_slab11_2d_smr E7r_slab11_2d_amr; do
  case $n in *2d*) np=4;; *) np=1;; esac
  $R $W/pc/${n}_dflux $np $I/$n.in hydro/double_flux=true hydro/cons_prolong_pc=true &
  $R $W/pc/${n}_std $np $I/$n.in hydro/cons_prolong_pc=true &
  $R $W/mm/${n}_dflux $np $I/$n.in hydro/double_flux=true &
done
wait; echo E7V-DONE
