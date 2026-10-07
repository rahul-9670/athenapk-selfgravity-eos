#!/bin/bash
#SBATCH --job-name=sg_smoke2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/self_gravity/runs/smoke2_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
W=$S/runs/smoke
$R $W/jeans 8 $I/jeans_wave.in $(mesh 32 16 16 8) self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=5
$R $W/slab 8 $I/grav_slab.in $(mesh 16 8 8 4) parthenon/time/nlim=5
$R $W/binary 8 $I/binary_amr.in $(mesh 32 32 32 16)
$R $W/sphere_amr 8 $I/sphere_amr.in $(mesh 32 32 32 16)
$R $W/poly_art 16 $I/polytrope_artemis.in $(mesh 64 64 64 16) parthenon/time/nlim=3
