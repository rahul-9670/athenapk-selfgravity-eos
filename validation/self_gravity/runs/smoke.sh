#!/bin/bash
#SBATCH --job-name=sg_smoke
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/self_gravity/runs/smoke_%j.out
source ~/athenapk_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_mtl=^psm2 OMPI_MCA_btl=tcp,self,sm PMIX_MCA_gds=hash
S=$REPO/validation/self_gravity; R=$S/scripts/run_one.sh; I=$S/inputs; W=$S/runs/smoke
M32="parthenon/mesh/nx1=32 parthenon/mesh/nx2=32 parthenon/mesh/nx3=32"
$R $W/sphere 8 $I/static_poisson.in $M32
$R $W/gauss 8 $I/static_poisson.in $M32 problem/grav_analytic/profile=gaussian problem/grav_analytic/mass=1 problem/grav_analytic/sigma=1
$R $W/sinus 8 $I/static_poisson.in $M32 problem/grav_analytic/profile=sinusoid parthenon/mesh/x1min=0 parthenon/mesh/x1max=1 parthenon/mesh/x2min=0 parthenon/mesh/x2max=1 parthenon/mesh/x3min=0 parthenon/mesh/x3max=1 parthenon/mesh/ix1_bc=periodic parthenon/mesh/ox1_bc=periodic parthenon/mesh/ix2_bc=periodic parthenon/mesh/ox2_bc=periodic parthenon/mesh/ix3_bc=periodic parthenon/mesh/ox3_bc=periodic self_gravity/use_swindle=true self_gravity/ix1_bc=default self_gravity/ox1_bc=default self_gravity/ix2_bc=default self_gravity/ox2_bc=default self_gravity/ix3_bc=default self_gravity/ox3_bc=default self_gravity/four_pi_G=12.566370614359172
$R $W/ellip 8 $I/static_poisson.in $M32 problem/grav_analytic/profile=ellipsoid problem/grav_analytic/a1=4 problem/grav_analytic/a2=4 problem/grav_analytic/a3=1.7436
$R $W/be 8 $I/be_ic.in parthenon/mesh/nx1=32 parthenon/mesh/nx2=32 parthenon/mesh/nx3=32
$R $W/jeans 8 $I/jeans_wave.in parthenon/mesh/nx1=32 parthenon/mesh/nx2=16 parthenon/mesh/nx3=16 parthenon/meshblock/nx1=8 parthenon/meshblock/nx2=8 parthenon/meshblock/nx3=8 self_gravity/four_pi_G=11.843525281307231 parthenon/time/nlim=5
$R $W/slab 8 $I/grav_slab.in parthenon/mesh/nx1=16 parthenon/mesh/nx2=8 parthenon/mesh/nx3=8 parthenon/meshblock/nx1=4 parthenon/meshblock/nx2=4 parthenon/meshblock/nx3=4 parthenon/time/nlim=5
$R $W/poly 8 $I/polytrope.in $M32 parthenon/time/nlim=5
$R $W/poly2 8 $I/polytrope.in $M32 parthenon/time/nlim=5 problem/polytrope/iprob=2 problem/polytrope/x10a=4 problem/polytrope/x20a=2.5
$R $W/ff 8 $I/freefall.in $M32 parthenon/time/nlim=5
