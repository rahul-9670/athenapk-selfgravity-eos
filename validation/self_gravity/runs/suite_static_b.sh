#!/bin/bash
#SBATCH --job-name=sg_staticB
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-S1 periodic sinusoid (Tomida & Stone 2023 Sec 4.1; Athena++ poisson iprob 1): G = 1
for n in 16 32 64 128 256; do mb=$(( n<64 ? 8 : 16 )); [ $n -ge 128 ] && mb=32
  for tol in prod tight; do X=""; [ $tol = tight ] && X="$tight"
    RS $W/S1_sinusoid/n${n}_$tol $(np_for $n $n $n $mb) $I/static_poisson.in $(mesh $n $n $n $mb) $cube01 $periodic \
      problem/grav_analytic/profile=sinusoid problem/grav_analytic/rho_bg=2 problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172 $X
  done; done
# tolerance sweep at 128^3
# tolerance sweep moved to suite_static2.sh (cold start, SG-F1)
for tol in; do
  RS $W/S1_sinusoid/tol_n128_$tol 64 $I/static_poisson.in $(mesh 128 128 128 32) $cube01 $periodic \
    problem/grav_analytic/profile=sinusoid self_gravity/four_pi_G=12.566370614359172 \
    self_gravity/solver_params/residual_tolerance=$tol self_gravity/solver_params/max_iterations=1000
done
# ---- SG-S2 Gaussian sphere, SG-S3 uniform sphere, SG-S5 Maclaurin spheroid, SG-S6 truncated Plummer (box [-8,8]^3)
for n in 32 64 128 256; do mb=$(( n<=64 ? 16 : 32 ))
  M="$(mesh $n $n $n $mb)"; NP=$(np_for $n $n $n $mb)
  RS $W/S2_gaussian/n${n}_mp $NP $I/static_poisson.in $M problem/grav_analytic/profile=gaussian problem/grav_analytic/mass=1 problem/grav_analytic/sigma=1
  RS $W/S3_sphere/n${n}_mp $NP $I/static_poisson.in $M
  RS $W/S5_maclaurin/n${n}_mp $NP $I/static_poisson.in $M problem/grav_analytic/profile=ellipsoid problem/grav_analytic/a1=4 problem/grav_analytic/a2=4 problem/grav_analytic/a3=1.7435595774162693
  RS $W/S6_plummer/n${n}_mp $NP $I/static_poisson.in $M problem/grav_analytic/profile=plummer problem/grav_analytic/mass=1 problem/grav_analytic/a_plummer=1 problem/grav_analytic/r_trunc=6
  RS $W/S7_be_ic/n${n}_mp $NP $I/be_ic.in $M
  RS $W/S3_sphere/n${n}_mp_tight $NP $I/static_poisson.in $M $tight
  RS $W/S7_be_ic/n${n}_mp_tight $NP $I/be_ic.in $M $tight
done
# zero-Dirichlet contrast (the pre-WS-5a BC) at 64 and 128
for n in 64 128; do mb=$(( n<=64 ? 16 : 32 )); M="$(mesh $n $n $n $mb)"; NP=$(np_for $n $n $n $mb)
  RS $W/S3_sphere/n${n}_zero $NP $I/static_poisson.in $M $zerobc
  RS $W/S5_maclaurin/n${n}_zero $NP $I/static_poisson.in $M $zerobc problem/grav_analytic/profile=ellipsoid problem/grav_analytic/a1=4 problem/grav_analytic/a2=4 problem/grav_analytic/a3=1.7435595774162693
  RS $W/S7_be_ic/n${n}_zero $NP $I/be_ic.in $M $zerobc
done
# box-size sensitivity of the multipole BC (Maclaurin, fixed dx = 0.125): half-width 6, 8, 12, 16
for L in 6 8 12 16; do n=$(( 2*L*8 )); M="$(mesh $n $n $n 16)"
  RS $W/S5_maclaurin/box${L}_dx0.125 64 $I/static_poisson.in $M parthenon/mesh/x1min=-$L parthenon/mesh/x1max=$L parthenon/mesh/x2min=-$L parthenon/mesh/x2max=$L parthenon/mesh/x3min=-$L parthenon/mesh/x3max=$L problem/grav_analytic/profile=ellipsoid problem/grav_analytic/a1=4 problem/grav_analytic/a2=4 problem/grav_analytic/a3=1.7435595774162693
done
# ---- SG-S4 binary on nested static AMR (Tomida & Stone 2023 Sec 4.2); SG-S3a sphere on static AMR
for n in 32 64 128; do
  RS $W/S4_binary/root${n} 8 $I/binary_amr.in $(mesh $n $n $n 16)
  RS $W/S4_binary/root${n}_tight 8 $I/binary_amr.in $(mesh $n $n $n 16) $tight
done
for n in 32 64 128; do RS $W/S3_sphere_amr/root${n} 8 $I/sphere_amr.in $(mesh $n $n $n 16); done
# ---- MPI decomposition invariance (Athena++ jeans_3d.py checks 1/2/4 ranks): sphere 64^3
for np in 1 8 64; do RS $W/S3_sphere/n64_np$np $np $I/static_poisson.in $(mesh 64 64 64 16); done
echo SUITE-STATIC-DONE
