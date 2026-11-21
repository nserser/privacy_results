#!/bin/bash
#SBATCH --job-name=finmem1-fs-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/gpu_investigation/logs/fs_gpu_%j.out
#SBATCH --error=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/gpu_investigation/logs/fs_gpu_%j.err

set -euo pipefail
ROOT=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121
PKG=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121
VENV=/users/anis.bkakria/privacy/.venv-cpu3-corr1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/gpu_investigation/vendor"
cd "$ROOT"
srun --cpu-bind=cores "$VENV/bin/python" -u "$ROOT/gpu_investigation/run_fs_gpu.py"
"$VENV/bin/python" -u "$ROOT/gpu_investigation/aggregate_fs_gpu.py"
