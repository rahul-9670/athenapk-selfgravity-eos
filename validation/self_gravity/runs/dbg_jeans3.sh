#!/bin/bash
#SBATCH --job-name=sg_dbg3
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/self_gravity/runs/dbg_jeans3_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
W=$S/runs/debug_jeans; M="$(mesh 32 16 16 8)"; A="self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=40"
$R $W/m_a_prod 8 $I/jeans_wave.in $M $A
$R $W/m_b_abstol 8 $I/jeans_wave.in $M $A self_gravity/solver_params/absolute_residual_tolerance=1e-10
$R $W/m_c_zeroguess 8 $I/jeans_wave.in $M $A self_gravity/solver_params/seed_from_phi=false self_gravity/solver_params/zero_initial_guess=true
$R $W/m_d_amp1e-3 8 $I/jeans_wave.in $M $A problem/jeans_wave/amp=1e-3
