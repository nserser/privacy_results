#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-$(nproc)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON="${PYTHON:-python3}"
"$PYTHON" - <<'PYCODE'
import sys
assert sys.version_info >= (3,10), f"Python 3.10+ required, got {sys.version}"
PYCODE
# Create/use an isolated venv. Set VENV_DIR to override its location.
VENV_DIR="${VENV_DIR:-.venv-finmem1}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then "$PYTHON" -m venv "$VENV_DIR"; fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-ubuntu.txt
export PYTHONPATH="$PWD/input/CORE_SOURCE/src:$PWD/input/PLD_accounting_git${PYTHONPATH:+:$PYTHONPATH}"
"$VENV_DIR/bin/python" finmem1/verify_portable.py
"$VENV_DIR/bin/python" finmem1/run_fs.py --workers "$WORKERS"
"$VENV_DIR/bin/python" finmem1/finalize_tables.py
"$VENV_DIR/bin/python" finmem1/aggregate.py
"$VENV_DIR/bin/python" finmem1/make_figures.py
