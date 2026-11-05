#!/bin/bash
# Aggregate shard outputs and draw all figures (runs after the array).
#SBATCH --job-name=pmsim-post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
CONFIG="${1:?usage: postprocess.sbatch CONFIG}"
cd "${SLURM_SUBMIT_DIR}"
slurm/pyrun.sh python -m pmsim.experiment aggregate "$CONFIG"
slurm/pyrun.sh python -m pmsim.experiment plot "$CONFIG"
