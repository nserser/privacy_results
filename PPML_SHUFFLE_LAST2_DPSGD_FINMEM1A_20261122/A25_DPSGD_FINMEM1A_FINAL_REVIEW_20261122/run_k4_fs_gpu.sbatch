#!/bin/bash
#SBATCH --job-name=finmem1a-k4-fs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/FINMEM1A_INVESTIGATION_20261122/logs/k4_gpu_%j.out
#SBATCH --error=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/FINMEM1A_INVESTIGATION_20261122/logs/k4_gpu_%j.err
set -euo pipefail
ROOT=/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/FINMEM1A_INVESTIGATION_20261122
VENV=/users/anis.bkakria/privacy/.venv-cpu3-corr1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/input/CORE_SOURCE/src:/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/gpu_investigation/vendor"
cd "$ROOT"
srun --cpu-bind=cores "$VENV/bin/python" -u "$ROOT/run_k4_fs_gpu.py"
