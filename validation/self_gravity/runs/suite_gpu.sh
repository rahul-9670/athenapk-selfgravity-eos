#!/bin/bash
#SBATCH --job-name=sg_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# GPU-path verification: the same decks/keys as the CPU legs, binary built from the same source
# (validation/physics-tests + uncommitted fixes, patch _build/bin/athenaPK_cpu_b76b9e6b.src.patch).
S=$REPO/validation/self_gravity
source $REPO/validation/self_gravity/scripts/suite_common.sh
module load cuda/12.5.1 >/dev/null 2>&1
G=$REPO/build_gpu/bin/athenaPK
M=$(md5sum $G | cut -c1-8); cp -p $G $REPO/validation/_build/bin/athenaPK_gpu_$M
export BIN=$REPO/validation/_build/bin/athenaPK_gpu_$M
echo "job $SLURM_JOB_ID $(date) GPU binary $(md5sum $BIN)"; nvidia-smi -L
W=$S/runs/GPU
# 1. static Poisson, production IC + production keys (CPU ref: S7_be_ic/n128_mp)
$R $W/S7_be_ic_n128_mp 1 $I/be_ic.in $(mesh 128 128 128 32)
# 2. deep static AMR, 6 levels (CPU ref: DA1_deep_static/root64_L6)
$R $W/DA1_root64_L6 1 $I/DA1_be_ic_deep_static.in
# 3. Jeans wave, periodic + swindle + new round-off floor (CPU ref: F1_fix/jeans_nJ0.5_n64_prod)
$R $W/jeans_nJ0.5_n64_prod 1 $I/jeans_wave.in $(mesh 64 32 32 16) self_gravity/four_pi_G=88.82643960980423 parthenon/time/tlim=0.3849001794597505 parthenon/output0/dt=0.019245
# 4. fallback path with the floor forced off (CPU ref: F1_fix/dbg_g88_np8_floor0)
$R $W/dbg_g88_floor0 1 $I/jeans_wave.in $(mesh 32 16 16 8) self_gravity/four_pi_G=88.826439609804 parthenon/time/nlim=40 self_gravity/solver_params/roundoff_floor=0
# 5. free fall 64^3 (CPU ref: D7_freefall/n64)
$R $W/freefall_n64 1 $I/freefall.in $(mesh 64 64 64 16)
# 6. deep adaptive AMR collapse, production IC, njeans 8, until t = 0.7 (CPU ref: L7_be_collapse/nj8)
$R $W/L7_nj8_t0.7 1 $I/L7_be_collapse_nj8.in parthenon/time/tlim=0.7
echo SUITE-GPU-DONE
