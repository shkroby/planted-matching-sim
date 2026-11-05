#!/bin/bash
# Run a command inside the project environment, from the repository root:
#   slurm/pyrun.sh python -m pmsim.experiment plan configs/recovery.yaml
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/slurm/config.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR
CMD=$(printf '%q ' "$@")
cd "$REPO"
if [[ "$PMSIM_ENV_MODE" == "overlay" ]]; then
  exec apptainer exec --overlay "$PMSIM_OVERLAY:ro" \
       --bind "$REPO,$SCRATCH" "$PMSIM_IMAGE" \
       /bin/bash -c "source /ext3/env.sh; cd '$REPO'; $CMD"
else
  if [[ -n "$PMSIM_PY_MODULE" ]]; then module purge; module load "$PMSIM_PY_MODULE"; fi
  source "$PMSIM_VENV/bin/activate"
  exec "$@"
fi
