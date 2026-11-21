# FINMEM1 Ubuntu resume package

This is a portable, resumable snapshot of the CPU-only FINMEM1 investigation. It includes the frozen MEMFRONT2 input artifacts, the exact THEORYCOMPLY source used by this run, the pinned Feldman–Shenfeld repository at commit `fde83c45bf12c4f7821832c7d5d2ae2841fc0910`, all completed direct/memory/robustness outputs, and the accountant queries completed when the package was assembled.

## Snapshot status

The package was assembled while the original host's accountant batch was still running. It contains the durable completed query files from that instant; any queries that were in progress will be recomputed on this server. The runner skips existing result files, so it resumes from completed results. It will not stop or update the original-host process.

Expected FS queries: 40. The runner verifies the pinned git commit, uses process-level parallelism, caps numerical-library threads at one, then rebuilds summary tables and figures. Use `WORKERS=56 ./RUN_ON_UBUNTU.sh` on a server that actually grants 56 CPUs; otherwise the runner accepts any requested value, so set `WORKERS` to the CPU allocation from SLURM (e.g. `srun --cpus-per-task=56 ...`). Never request/use a GPU for this milestone.

## Run

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git build-essential
# unpack the zip, then:
cd A25_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121
WORKERS=56 ./RUN_ON_UBUNTU.sh
```

The script creates `.venv-finmem1`, installs the small CPU dependency set, verifies source locks, resumes all outstanding FS queries, and regenerates tables/figures. To run manually inside an existing environment:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/input/CORE_SOURCE/src:$PWD/input/PLD_accounting_git"
python3 finmem1/verify_portable.py
python3 finmem1/run_fs.py --workers 56
python3 finmem1/finalize_tables.py
python3 finmem1/aggregate.py
python3 finmem1/make_figures.py
```

The batch is resumable from `final_state/fs/*.json`. Do not delete those files. `FINMEM1_PROGRESS.jsonl` and `FINMEM1_JOB_LEDGER.jsonl` are append-only execution records.

## Scientific scope

This archive only resumes the frozen FINMEM1 accountant/comparison stage. It does not alter the mechanism, grids, support, privacy definitions, or decision thresholds. Historical SCALECERT, MEMFRONT1 and MEMFRONT2 decisions remain unchanged. No GPU is part of this run.
