#!/bin/bash
#SBATCH --job-name=sg_dbgs1
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=$REPO/validation/self_gravity/runs/dbg_s1_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
W=$S/runs/debug_s1; P="problem/grav_analytic/profile=sinusoid problem/grav_analytic/rho_bg=2 problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172 self_gravity/solver_params/print_per_step=true"
$R $W/np8_mb8 8 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P
$R $W/np1_mb8 1 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P
$R $W/np1_mb16 1 $I/static_poisson.in $(mesh 16 16 16 16) $cube01 $periodic $P
$R $W/np8_mb8_noFAS 8 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P self_gravity/solver_params/do_FAS=false
$R $W/np8_mb8_tol1e-8 8 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P self_gravity/solver_params/residual_tolerance=1e-8
