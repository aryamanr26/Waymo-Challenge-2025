#!/usr/bin/env bash
#SBATCH --job-name=waymo-dataloader-check
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=logs/slurm_check_dataloader.out
#SBATCH --error=logs/slurm_check_dataloader.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Usage:
#   sbatch scripts/slurm_check_dataloader.sh
#
# Optional overrides at submit time:
#   sbatch --export=ALL,CONDA_ENV=waymo,CUDA_MODULE=cuda/12.1,CUDNN_MODULE=cudnn/8.9 \
#          scripts/slurm_check_dataloader.sh
#
#   sbatch --export=ALL,WAYMO_NUM_TEMPORAL_FRAMES=5,WAYMO_SMOKE_BATCH_SIZE=2 \
#          scripts/slurm_check_dataloader.sh
# -----------------------------------------------------------------------------

# In some SLURM setups, BASH_SOURCE points to a spool copy path.
# Prefer submission directory for stable/writable project root resolution.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

# If local waymo-open-dataset checkout exists, expose its src path.
if [[ -d "${REPO_ROOT}/waymo-open-dataset/src" ]]; then
  export PYTHONPATH="${REPO_ROOT}/waymo-open-dataset/src:${PYTHONPATH:-}"
fi

echo "== SLURM job context =="
echo "HOSTNAME: ${HOSTNAME:-unknown}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "PWD: $(pwd)"
echo

# ---- Modules (adjust names for your cluster if needed) ----
module purge
module load gcc/11 || true
module load "${CUDA_MODULE:-cuda/12.1}" || true
# Optional: load only if present on your cluster
if [[ -n "${CUDNN_MODULE:-}" ]]; then
  module load "${CUDNN_MODULE}" || true
fi

echo "== Loaded modules =="
module list || true
echo

# ---- Conda environment ----
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate waymo

echo "== Runtime versions =="
which python
python -V
nvidia-smi || true
python -c "import torch, tensorflow as tf; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('tf', tf.__version__)"
echo

# ---- Dataloader smoke test ----
echo "== Running dataloader smoke test =="
echo "WAYMO_NUM_TEMPORAL_FRAMES=${WAYMO_NUM_TEMPORAL_FRAMES:-1}"
echo "WAYMO_SMOKE_BATCH_SIZE=${WAYMO_SMOKE_BATCH_SIZE:-2}"
echo "WAYMO_SMOKE_SHARDS=${WAYMO_SMOKE_SHARDS:-1}"

./scripts/check_waymo_dataloader.sh

echo
echo "Done."
