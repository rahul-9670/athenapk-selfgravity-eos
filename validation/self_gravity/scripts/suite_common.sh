# sourced by every suite job script
source ~/athenapk_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_mtl=^psm2 OMPI_MCA_btl=tcp,self,sm PMIX_MCA_gds=hash OMPI_MCA_btl_sm_fbox_max=0  # SG-F7: Open MPI 5.0.7 sm fast-box deadlock (ompi #12979); fbox off = no hang in 1339 L9 cycles, same speed
S=$REPO/validation/self_gravity; R=$S/scripts/run_one.sh; I=$S/inputs; W=$S/runs
NPMAX=${SLURM_NTASKS:-64}
np_for() { local nb=$(( ($1/$4) * ($2/$4) * ($3/$4) )); [ $nb -lt $NPMAX ] && echo $nb || echo $NPMAX; }  # nx ny nz mb
mesh() { echo "parthenon/mesh/nx1=$1 parthenon/mesh/nx2=$2 parthenon/mesh/nx3=$3 parthenon/meshblock/nx1=$4 parthenon/meshblock/nx2=$4 parthenon/meshblock/nx3=$4"; }
cube01="parthenon/mesh/x1min=0 parthenon/mesh/x1max=1 parthenon/mesh/x2min=0 parthenon/mesh/x2max=1 parthenon/mesh/x3min=0 parthenon/mesh/x3max=1"
periodic="parthenon/mesh/ix1_bc=periodic parthenon/mesh/ox1_bc=periodic parthenon/mesh/ix2_bc=periodic parthenon/mesh/ox2_bc=periodic parthenon/mesh/ix3_bc=periodic parthenon/mesh/ox3_bc=periodic self_gravity/use_swindle=true self_gravity/ix1_bc=default self_gravity/ox1_bc=default self_gravity/ix2_bc=default self_gravity/ox2_bc=default self_gravity/ix3_bc=default self_gravity/ox3_bc=default"
zerobc="self_gravity/ix1_bc=zero self_gravity/ox1_bc=zero self_gravity/ix2_bc=zero self_gravity/ox2_bc=zero self_gravity/ix3_bc=zero self_gravity/ox3_bc=zero"
# "tight" = converged reference: COLD start, 1e-10 relative to rms(rhs) (finding SG-F1: a warm-start
# relative target below the round-off floor stalls and BiCGSTAB returns its last iterate)
tight="self_gravity/solver_params/residual_tolerance=1.0e-10 self_gravity/solver_params/max_iterations=400 self_gravity/solver_params/seed_from_phi=false self_gravity/solver_params/zero_initial_guess=true"
# skip a leg whose final dump already exists (resubmission after a cancelled job)
done_leg() { ls $1/*.out0.final.phdf >/dev/null 2>&1; }
RS() { if done_leg "$1"; then echo "[skip] $(basename $1) (final dump exists)"; else $R "$@"; fi; }
