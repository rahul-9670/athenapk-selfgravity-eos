#!/bin/bash
#SBATCH --job-name=rad_gpu
#SBATCH --partition=gpu
#SBATCH --account=banerjee_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/gpu_%j.out
# R6: run a list (rundir | deck | overrides | nranks) SEQUENTIALLY on ONE H100, 1 MPI rank (nranks ignored).
export TMPDIR=/tmp
source ~/athenapk_env.sh
module load cuda/12.5.1 >/dev/null 2>&1
ulimit -c 0
export OMP_NUM_THREADS=1 OMPI_MCA_btl=self OMPI_MCA_mtl=^psm2
M=${RAD_MOD:-$REPO/validation/radiation_sn}
BIN=${BIN:?}
nvidia-smi -L
echo "BIN $(md5sum $BIN)  LIST $LIST"
grep -E "^${FILTER:-.}" "$LIST" | while IFS='|' read -r d k o n; do
  d=$(echo $d); k=$(echo $k); o=$(echo $o)
  dir=$M/runs/$d; deck=$M/inputs/$k; mkdir -p $dir; cd $dir; rm -f *.phdf *.rhdf *.xdmf *.hst run.log
  { echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none} gpu $CUDA_VISIBLE_DEVICES"; echo "bin  $(md5sum $BIN)";
    echo "deck $deck $(md5sum < $deck | cut -c1-32)"; echo "cmd  $BIN -i $deck $o"; } > cmd.txt
  T0=$(date +%s.%N); timeout ${CAP:-3000} $BIN -i $deck $o > run.log 2>&1 < /dev/null; rc=$?
  echo "rc $rc wall $(echo "$(date +%s.%N) - $T0" | bc) s" >> cmd.txt
  echo "[$rc] $d $(grep -aE '^cycle=' run.log | tail -1 | cut -c1-90)"
done
echo LIST-DONE
