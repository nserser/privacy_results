import concurrent.futures, json, os
from pathlib import Path
from final_state import run, smoke, ROOT

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
jobs = [('P1','BASE',m) for m in (256,512,1024,2048,4096,8192,16384)]
jobs += [('P1',g,m) for g in ('GRID','GRID2','DOMAIN') for m in (1024,4096,16384)]
jobs = list(dict.fromkeys(jobs))
(ROOT/'FINMEM1A_EXPECTED_JOBS.json').write_text(json.dumps([
    {'line':a,'grid':b,'M':m,'k':2,'b':1,'status':'REQUIRED'} for a,b,m in jobs
], indent=2)+'\n')
s = smoke()
(ROOT/'FINMEM1A_RECURRENCE_SMOKE.json').write_text(json.dumps(s,indent=2)+'\n')
if not s['pass']:
    raise SystemExit('recurrence smoke failed')
print(json.dumps({'event':'batch_start','n_jobs':len(jobs),'workers':min(8,os.cpu_count() or 1),'gpu':False}),flush=True)
with concurrent.futures.ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 1)) as pool:
    rows=[]
    for i,row in enumerate(pool.map(run,jobs),1):
        rows.append(row)
        print(json.dumps({'event':'job_complete','done':i,'total':len(jobs),'line':row['line'],'grid':row['grid'],'M':row['M']}),flush=True)
(ROOT/'FINMEM1A_JOBS_SUMMARY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
