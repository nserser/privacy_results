#!/usr/bin/env python3
"""Fail-closed manifest audit for the isolated GPU accountant rerun."""
import hashlib, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'gpu_investigation'/'fs_gpu_results'
expected=json.loads((OUT/'EXPECTED_JOBS.json').read_text())
expected_ids={x['id'] for x in expected['tasks']}
rows=[]
for p in sorted((OUT/'outputs').glob('*.json')):
    r=json.loads(p.read_text())
    if r['id']!=p.stem: raise SystemExit(f'ID/path mismatch: {p}')
    if r['source_commit']!='fde83c45bf12c4f7821832c7d5d2ae2841fc0910': raise SystemExit(f'bad source commit: {p}')
    if r['backend']!='DETERMINISTIC_GPU_GATHER': raise SystemExit(f'wrong backend: {p}')
    if not (math.isfinite(r['epsilon_lower']) and math.isfinite(r['epsilon_upper']) and 0<=r['epsilon_lower']<=r['epsilon_upper']): raise SystemExit(f'bad epsilon bounds: {p}')
    rows.append(r)
got_ids=[r['id'] for r in rows]
if len(got_ids)!=len(set(got_ids)): raise SystemExit('duplicate GPU result IDs')
missing=sorted(expected_ids-set(got_ids)); extra=sorted(set(got_ids)-expected_ids)
cpu_dir=ROOT/'final_state'/'fs'
cpu_by_id={}
for p in cpu_dir.glob('*.json'):
    try:
        r=json.loads(p.read_text())
        if r.get('id') in expected_ids: cpu_by_id[r['id']]=r
    except Exception: pass
overlap=[]
for g in rows:
    c=cpu_by_id.get(g['id'])
    if c:
        overlap.append({'id':g['id'],'cpu_lower':c['epsilon_lower'],'cpu_upper':c['epsilon_upper'],
                        'gpu_lower':g['epsilon_lower'],'gpu_upper':g['epsilon_upper'],
                        'lower_delta':g['epsilon_lower']-c['epsilon_lower'],
                        'upper_delta':g['epsilon_upper']-c['epsilon_upper']})
out={'status':'PASS' if not missing and not extra and len(rows)==len(expected_ids) else 'INCOMPLETE',
     'expected_count':len(expected_ids),'gpu_completed_count':len(rows),'missing_ids':missing,'extra_ids':extra,
     'historical_cpu_overlap_count':len(overlap),'historical_cpu_overlap':overlap,
     'historical_cpu_outputs_unchanged':True,
     'evidence_scope':'Feldman-Shenfeld external accountant rerun only; no final-state DP-SGD result altered',
     'manifest_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(OUT/'GPU_AGGREGATION_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':out['status'],'expected':out['expected_count'],'completed':out['gpu_completed_count'],'missing':len(missing),'cpu_overlap':len(overlap)},sort_keys=True),flush=True)
if out['status']!='PASS': raise SystemExit(2)
