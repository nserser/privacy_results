#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/external/PLD_accounting"
BRANCH="${PLD_BRANCH:-MS-FFT-paper}"
mkdir -p "$ROOT/external" "$ROOT/results"
LOCK="$ROOT/external/FELDMAN_SHENFELD_RUNTIME_LOCK.json"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/moshenfeld/PLD_accounting "$DEST"
fi
git -C "$DEST" fetch --all --tags
if [ -f "$LOCK" ]; then
  COMMIT="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["commit"])' "$LOCK")"
  git -C "$DEST" checkout --detach "$COMMIT"
else
  git -C "$DEST" checkout "$BRANCH"
  git -C "$DEST" reset --hard "origin/$BRANCH"
  COMMIT="$(git -C "$DEST" rev-parse HEAD)"
  printf '{"repository":"https://github.com/moshenfeld/PLD_accounting","branch":"%s","commit":"%s"}\n' "$BRANCH" "$COMMIT" | tee "$LOCK"
fi
python -m pip install -e "$DEST"
PYTHONPATH="$ROOT/src" python - <<'PY'
import json
from scalecert.baseline_fs import package_probe, gaussian_allocation_epsilon
print(json.dumps(package_probe(),indent=2))
print(json.dumps(gaussian_allocation_epsilon(delta=1e-5,sigma=4.0,num_steps=100,num_selected=1,num_epochs=1),indent=2))
PY
