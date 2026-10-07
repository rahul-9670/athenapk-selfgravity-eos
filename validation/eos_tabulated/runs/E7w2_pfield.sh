#!/bin/bash
#SBATCH --job-name=eos_E7w2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7w2: EOS-consistent fill via the transported pressure dflux_p (ghosts at level jumps incl. restricted ones, regridded
# interiors). Binary ac833be5. gf = on, nogf = off (= E7r), std = key off (= E7r std); E7x acoustic wave through SMR.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_ac833be5
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7w2
for n in E7r_R4_mov_1d_smr E7r_R4_mov_1d_amr E7r_slab11_2d_smr E7r_slab11_2d_amr; do
  case $n in *2d*) np=4;; *) np=1;; esac
  $R $W/gf/${n}_dflux $np $I/$n.in hydro/double_flux=true &
  $R $W/nogf/${n}_dflux $np $I/$n.in hydro/double_flux=true hydro/double_flux_ghost_fill=false &
  $R $W/gf/${n}_std $np $I/$n.in &
done
for N in 32 64 128; do
  $R $W/wave/E7x_wave_S3_smr_n${N}_dflux 1 $I/E7x_wave_S3_smr_n$N.in hydro/double_flux=true &
done
wait; echo E7W2-DONE
