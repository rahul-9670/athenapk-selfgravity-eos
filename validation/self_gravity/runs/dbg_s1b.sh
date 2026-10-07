#!/bin/bash
#SBATCH --job-name=sg_dbgs1b
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=$REPO/validation/self_gravity/runs/dbg_s1b_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
W=$S/runs/debug_s1; P="problem/grav_analytic/profile=sinusoid problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172 self_gravity/solver_params/print_per_step=true self_gravity/solver_params/max_iterations=40"
$R $W/np8_rhobg20 8 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P problem/grav_analytic/rho_bg=20
$R $W/np8_rhobg200 8 $I/static_poisson.in $(mesh 16 16 16 8) $cube01 $periodic $P problem/grav_analytic/rho_bg=200
