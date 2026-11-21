#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
RUN="${1:-runs/full}"
OUT="${2:-PPML_SHUFFLE_LAST2_EMP1_SCALECERT_RESULTS_$(date +%Y%m%d_%H%M%S).zip}"
[ -d "$RUN" ] || { echo "missing run directory $RUN"; exit 2; }
sha256sum "$(find "$RUN" -type f | sort | head -1)" >/dev/null 2>&1 || true
zip -qr "$OUT" "$RUN" logs external/FELDMAN_SHENFELD_RUNTIME_LOCK.json IMPLEMENTATION_VERIFICATION.json PROVENANCE.json -x 'external/PLD_accounting/*'
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
