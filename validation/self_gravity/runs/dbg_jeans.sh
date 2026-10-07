#!/bin/bash
#SBATCH --job-name=sg_dbg
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=$REPO/validation/self_gravity/runs/dbg_jeans_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
W=$S/runs/debug_jeans; M="$(mesh 32 16 16 8)"
$R $W/nograv 8 $I/jeans_wave.in $M physics/self_gravity=false parthenon/time/nlim=20
$R $W/g1 8 $I/jeans_wave.in $M self_gravity/four_pi_G=1.0 parthenon/time/nlim=20
$R $W/g88 8 $I/jeans_wave.in $M self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=20
$R $W/g88_np1 1 $I/jeans_wave.in $M self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=20
$R $W/g88_amp0 8 $I/jeans_wave.in $M self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=20 problem/jeans_wave/amp=0
