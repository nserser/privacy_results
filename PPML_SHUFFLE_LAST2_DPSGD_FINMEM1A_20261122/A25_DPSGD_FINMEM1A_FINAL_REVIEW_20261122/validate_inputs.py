import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FS=ROOT/'input/FINMEM1_FS/fs_gpu_results'
tasks=json.loads((FS/'EXPECTED_JOBS.json').read_text())['tasks']
expected={(int(x['M']),float(x['delta'])):x for x in tasks if x['line']=='P1' and x['k']==2 and x['b']==1 and x['z']==2.0}
outputs=list((FS/'outputs').glob('*.json'))
decoded=[json.loads(p.read_text()) for p in outputs]
p1_all=[x for x in decoded if x.get('line')=='P1' and x.get('k')==2]
rows=[]; seen=set(); errors=[]
if len(p1_all)!=14: errors.append(f'unexpected P1 k2 output count {len(p1_all)}')
if len({x.get('id') for x in p1_all})!=14: errors.append('duplicate or missing P1 k2 output IDs')
for (m,d),task in sorted(expected.items()):
    p=FS/'outputs'/f"{task['id']}.json"
    if not p.exists(): errors.append(f"missing {task['id']}"); continue
    raw=p.read_bytes(); x=json.loads(raw)
    if x.get('id')!=task['id']: errors.append(f"id mismatch {task['id']}")
    key=(int(x['M']),float(x['delta']))
    if key in seen: errors.append(f"duplicate {key}")
    seen.add(key)
    for f,v in [('line','P1'),('k',2),('b',1),('sigma',2.0),('num_steps',m),('delta',d),('source_commit','fde83c45bf12c4f7821832c7d5d2ae2841fc0910')]:
        if x.get(f)!=v: errors.append(f"{task['id']} {f}: {x.get(f)!r} != {v!r}")
    lo=float(x['epsilon_lower']); hi=float(x['epsilon_upper'])
    if not (0 <= lo <= hi): errors.append(f"bad epsilon bounds {task['id']}")
    rows.append({'M':m,'delta':d,'epsilon_lower':lo,'epsilon_upper':hi,'id':x['id'],'raw_output_sha256':hashlib.sha256(raw).hexdigest(),'source_commit':x['source_commit'],'backend':x.get('backend'),'gpu_model':x.get('gpu_model'),'status':'REUSED_VALIDATED'})
if len(expected)!=14 or len(rows)!=14 or len(seen)!=14: errors.append('P1 query cardinality mismatch')
with (ROOT/'P1_FS_FROZEN_BASELINE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['M']);w.writeheader();w.writerows(rows)
report={'p1_expected_queries':len(expected),'p1_validated_queries':len(rows),'total_snapshot_outputs':len(outputs),'p1_complete':len(rows)==14 and not errors,'all_expected_ids_present':all((FS/'outputs'/f"{x['id']}.json").exists() for x in expected.values()),'source_commit':'fde83c45bf12c4f7821832c7d5d2ae2841fc0910','integrity_errors':errors,'rows':rows}
(ROOT/'P1_FS_VALIDATION.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if errors: raise SystemExit(json.dumps(errors))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
