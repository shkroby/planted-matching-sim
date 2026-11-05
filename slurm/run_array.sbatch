#!/bin/bash
# One shard of the experiment per array task; submit via slurm/submit.sh.
#SBATCH --job-name=pmsim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --requeue
set -euo pipefail
CONFIG="${1:?usage: run_array.sbatch CONFIG}"
cd "${SLURM_SUBMIT_DIR}"
# one numba/BLAS thread per worker process: parallelism is across instances
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} cpus=${SLURM_CPUS_PER_TASK}"
slurm/pyrun.sh python -m pmsim.experiment run "$CONFIG" \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${SLURM_ARRAY_TASK_COUNT}" \
    --workers "${SLURM_CPUS_PER_TASK}"
