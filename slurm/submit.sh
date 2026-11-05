#!/bin/bash
# Submit an experiment to Torch:  bash slurm/submit.sh configs/recovery.yaml [NSHARDS]
# Launches a job array (one shard per task) plus a dependent aggregate+plot job.
# Re-running the same command later only computes tasks that are missing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source slurm/config.sh
CONFIG="${1:?usage: submit.sh CONFIG [NSHARDS]}"
if [[ "$TORCH_ACCOUNT" == *XXX* ]]; then
  echo "Set TORCH_ACCOUNT (see 'my_slurm_accounts') in slurm/config.sh or the environment." >&2
  exit 1
fi
mkdir -p logs
slurm/pyrun.sh python -m pmsim.experiment plan "$CONFIG" --cpus 64 | tee logs/plan.txt
NSHARDS="${2:-$(grep '^NSHARDS=' logs/plan.txt | cut -d= -f2)}"

EXTRA=()
if [[ "$PMSIM_PREEMPT" == "1" ]]; then
  EXTRA+=(--comment="preemption=yes;requeue=true")
fi
JOB=$(sbatch --parsable --account="$TORCH_ACCOUNT" --array=0-$((NSHARDS - 1)) \
      "${EXTRA[@]}" slurm/run_array.sbatch "$CONFIG")
POST=$(sbatch --parsable --account="$TORCH_ACCOUNT" --dependency=afterany:"$JOB" \
       slurm/postprocess.sbatch "$CONFIG")
echo "array job $JOB ($NSHARDS shards), postprocess job $POST"
echo "monitor: squeue -u \$USER ; tail -f logs/pmsim_${JOB}_0.out"
