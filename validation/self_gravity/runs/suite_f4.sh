#!/bin/bash
#SBATCH --job-name=sg_f4
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# SG-F4 falsification. (a) Ballistic converging flow, gravity OFF: v = -0.5 r, rho = 1 -> exact
# rho_c(t) = (1 - 0.5 t)^-3. If vl2 is first order here too, the defect is the VL2 donor-cell
# predictor at the converging stagnation point and gravity is not involved. (b) the free fall
# with the predictor reconstruction = PLM (hydro/vl2_predictor_recon = same).
BAL="problem/grav_analytic/profile=uniform problem/grav_analytic/rho_bg=1 problem/grav_analytic/v_homologous=0.5 problem/grav_analytic/pressure=1e-6 physics/self_gravity=false parthenon/time/nlim=-1 parthenon/time/tlim=1.4 parthenon/time/cfl=0.2 parthenon/output0/dt=0.05 parthenon/output0/variables=prim parthenon/output0/single_precision_output=true hydro/dfloor=1e-10 hydro/pfloor=1e-14"
for dc in 0.002 0.001 0.0005; do
  for v in vl2 rk2 vl2same; do
    X="parthenon/time/integrator=${v%same}"; [ $v = vl2same ] && X="parthenon/time/integrator=vl2 hydro/vl2_predictor_recon=same"
    RS $W/F4_ballistic/n64_${v}_dtceil$dc 64 $I/static_poisson.in $(mesh 64 64 64 16) $BAL parthenon/time/dt_ceil=$dc $X
  done
  RS $W/D7_freefall/n64_vl2same_dtceil$dc 64 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/dt_ceil=$dc hydro/vl2_predictor_recon=same
done
echo SUITE-F4-DONE
