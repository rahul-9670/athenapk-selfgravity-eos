#!/bin/bash
#SBATCH --job-name=eos_E7t
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7t: 1D regression of double flux v6 (X,Y,Z face sums as flux-corrected fluxes; scratch sized from prim; binary
# fe0caa0f): contacts + R1-R4 as E7p (N <= 3200), E6 waves key on/off, key-off legs vs a15514d6 (EF56_regate set).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_fe0caa0f
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs
for p in R4_ionization R2_dissociation R1_molecular; do for N in 400 800 1600 3200; do
  $R $W/E7t/E7k_${p}_mov_n${N}_dflux 1 $I/E7k_${p}_mov_n$N.in hydro/double_flux=true &
done; done
for p in R4_ionization R2_dissociation R1_molecular R3_dissoc_shock; do for N in 400 800 1600 3200; do
  $R $W/E7t/E7e_${p}_n${N}_dflux 1 $I/E7e_${p}_p_over_rho_n$N.in hydro/double_flux=true &
done; done
for s in S1 S2 S3 S4 S5; do for N in 16 32 64 128 256; do
  $R $W/E6_dflux_v6/E6_wave_${s}_n$N 1 $I/E6_wave_${s}_n$N.in hydro/double_flux=true &
done; done
for d in $I/E10_probe_*.in; do $R $W/EF56_regate_v6/$(basename $d .in)_fix 1 $d & done
for d in $I/E7e_*_n400.in $I/E6_wave_S?_n128.in $I/E8_leg2.in; do $R $W/EF56_regate_v6/bitid_$(basename $d .in) 1 $d & done
wait; echo E7T-DONE
