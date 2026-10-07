#!/bin/bash
#SBATCH --job-name=sg_dbg2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=$REPO/validation/self_gravity/runs/dbg_jeans2_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
W=$S/runs/debug_jeans; A="self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=10 problem/jeans_wave/amp=0"
$R $W/a_np2 2 $I/jeans_wave.in $(mesh 32 16 16 8) $A
$R $W/a_np4 4 $I/jeans_wave.in $(mesh 32 16 16 8) $A
$R $W/a_np8_rk2 8 $I/jeans_wave.in $(mesh 32 16 16 8) $A parthenon/time/integrator=rk2
$R $W/a_np8_coal 8 $I/jeans_wave.in $(mesh 32 16 16 8) $A parthenon/mesh/do_coalesced_comms=true
$R $W/a_np2_mb16 2 $I/jeans_wave.in $(mesh 32 16 16 16) $A
$R $W/a_np8_cube 8 $I/jeans_wave.in $(mesh 32 32 32 16) $A parthenon/mesh/x2max=1 parthenon/mesh/x3max=1
$R $W/a_np8_1e-10 8 $I/jeans_wave.in $(mesh 32 16 16 8) $A self_gravity/solver_params/residual_tolerance=1e-10
$R $W/a_np8_nophisplit 8 $I/jeans_wave.in $(mesh 32 16 16 8) $A self_gravity/phi_operator_split=false
$R $W/a_np8_noseed 8 $I/jeans_wave.in $(mesh 32 16 16 8) $A self_gravity/solver_params/seed_from_phi=false
