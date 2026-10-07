#!/bin/bash
#SBATCH --job-name=eos_E7w
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7w: EOS-consistent ghost fill (hydro/double_flux_ghost_fill) on the E7r refinement decks. gf = fill on, gfchk = fill
# on + index self-check, nogf = fill off (must reproduce runs/E7r bitwise).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_11d1eb2b
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7w
for n in E7r_R4_mov_1d_smr E7r_R4_mov_1d_amr E7r_slab11_2d_smr E7r_slab11_2d_amr; do
  case $n in *2d*) np=4;; *) np=1;; esac
  $R $W/gf/${n}_dflux $np $I/$n.in hydro/double_flux=true &
  $R $W/gfchk/${n}_dflux $np $I/$n.in hydro/double_flux=true hydro/double_flux_ghost_fill_check=true &
  $R $W/nogf/${n}_dflux $np $I/$n.in hydro/double_flux=true hydro/double_flux_ghost_fill=false &
done
wait; echo E7W-DONE
