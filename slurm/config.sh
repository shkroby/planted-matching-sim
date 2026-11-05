# Site settings for NYU Torch. Edit once (or export these before submitting).
# Your Slurm account: run `my_slurm_accounts` on a login node.
export TORCH_ACCOUNT="${TORCH_ACCOUNT:-torch_pr_XXX_XXXXX}"

# Environment: "overlay" (Apptainer + ext3 overlay, recommended by NYU HPC)
# or "venv" (plain Python virtualenv on $SCRATCH).
export PMSIM_ENV_MODE="${PMSIM_ENV_MODE:-overlay}"
export PMSIM_HOME="${PMSIM_HOME:-$SCRATCH/pmsim}"
export PMSIM_OVERLAY="${PMSIM_OVERLAY:-$PMSIM_HOME/pmsim.ext3}"
export PMSIM_IMAGE="${PMSIM_IMAGE:-/share/apps/images/cuda-13.3.1-ubuntu-26.04.sif}"
export PMSIM_VENV="${PMSIM_VENV:-$PMSIM_HOME/venv}"
# venv mode only: Python module to load (see `module avail python`).
export PMSIM_PY_MODULE="${PMSIM_PY_MODULE:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PMSIM_HOME/numba_cache}"

# Set to 1 to also use preemptible partitions (faster start; jobs are
# requeued automatically and resume from finished tasks).
export PMSIM_PREEMPT="${PMSIM_PREEMPT:-0}"
