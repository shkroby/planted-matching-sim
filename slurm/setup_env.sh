#!/bin/bash
# One-time environment setup on a Torch login node:
#   bash slurm/setup_env.sh
# Builds either an Apptainer overlay with Miniforge (default) or a venv.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/slurm/config.sh"
mkdir -p "$PMSIM_HOME" "$NUMBA_CACHE_DIR"
PKGS="numpy scipy numba pandas pyyaml matplotlib pytest"

if [[ "$PMSIM_ENV_MODE" == "overlay" ]]; then
  if [[ ! -f "$PMSIM_OVERLAY" ]]; then
    cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz "$PMSIM_OVERLAY.gz"
    gunzip "$PMSIM_OVERLAY.gz"
  fi
  apptainer exec --fakeroot --overlay "$PMSIM_OVERLAY:rw" "$PMSIM_IMAGE" /bin/bash -c "
    set -e
    if [[ ! -d /ext3/miniforge3 ]]; then
      cd /tmp
      wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
      bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
      rm -f Miniforge3-Linux-x86_64.sh
    fi
    cat > /ext3/env.sh <<'ENV'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:\$PATH
ENV
    source /ext3/env.sh
    conda install -y -q -c conda-forge python=3.12 $PKGS
    conda clean -y -a
    python -c 'import numba, scipy; print(\"numba\", numba.__version__, \"scipy\", scipy.__version__)'
  "
else
  if [[ -n "$PMSIM_PY_MODULE" ]]; then module purge; module load "$PMSIM_PY_MODULE"; fi
  python3 -m venv "$PMSIM_VENV"
  source "$PMSIM_VENV/bin/activate"
  pip install -q --upgrade pip
  pip install -q $PKGS
fi

# Compile and cache the numba kernels once, and run the exactness tests.
"$REPO/slurm/pyrun.sh" python -m pytest -q tests
echo "Environment ready ($PMSIM_ENV_MODE)."
