#!/bin/bash
#SBATCH --job-name=eos_E7x
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7x: acoustic wave through a static refined patch: double flux + ghost fill vs standard (the sensor never fires in
# sound waves, so the difference is the ghost fill alone). Binary 11d1eb2b.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false; unset OMP_PLACES
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_11d1eb2b
R=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E7x
for N in 32 64 128; do
  $R $W/E7x_wave_S3_smr_n${N}_dflux 1 $I/E7x_wave_S3_smr_n$N.in hydro/double_flux=true &
  $R $W/E7x_wave_S3_smr_n${N}_std 1 $I/E7x_wave_S3_smr_n$N.in &
  $R $W/E7x_wave_S3_smr_n${N}_dfluxnogf 1 $I/E7x_wave_S3_smr_n$N.in hydro/double_flux=true hydro/double_flux_ghost_fill=false &
done
wait; echo E7X-DONE
