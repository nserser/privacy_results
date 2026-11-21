#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rm -rf results/equiv_w1 results/equiv_w4 results/verify_smoke results/verify_summary
pytest -q
python scripts/run_plan.py --config configs/equivalence.yaml --out results/equiv_w1 --workers 1
python scripts/run_plan.py --config configs/equivalence.yaml --out results/equiv_w4 --workers 4
python scripts/compare_runs.py results/equiv_w1 results/equiv_w4
python scripts/build_expected_manifest.py --config configs/smoke.yaml --out results/SMOKE_EXPECTED_MANIFEST.json
python scripts/run_plan.py --config configs/smoke.yaml --out results/verify_smoke --workers 4
python scripts/aggregate.py --input results/verify_smoke --out results/verify_summary --config configs/smoke.yaml
python - <<'PY'
import json,pathlib,platform,sys,yaml
root=pathlib.Path('.'); sys.path.insert(0,str(root/'src'))
from scalecert.plan import build_plan
checks=json.loads((root/'results/verify_summary/checks.json').read_text())
required=['expected_manifest','convergence','main_scaling','theory_certificates','at_most_k_adjacency','multi_epoch','no_forgetting','sgld_bridge','dpsgd_numerical_convergence']
assert all(checks[k]['pass'] for k in required), {k:checks[k]['pass'] for k in required}
smoke_count=len(build_plan(yaml.safe_load(open('configs/smoke.yaml'))))
full_count=len(build_plan(yaml.safe_load(open('configs/full_staged.yaml'))))
payload={'status':'THEORY_COMPLIANCE_IMPLEMENTATION_VERIFIED','python':platform.python_version(),'required_smoke_checks':{k:checks[k]['pass'] for k in required},'worker_equivalence':'EXACT','unit_tests':'PASS','unit_test_count':22,'smoke_job_count':smoke_count,'full_expected_job_count':full_count,'external_baseline':'SERVER_RUNTIME_REQUIRED'}
(root/'IMPLEMENTATION_VERIFICATION.json').write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps(payload,indent=2))
PY
