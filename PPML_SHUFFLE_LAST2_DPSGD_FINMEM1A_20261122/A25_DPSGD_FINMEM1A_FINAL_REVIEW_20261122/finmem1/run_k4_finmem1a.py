import concurrent.futures, json, os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_state_k import run, ROOT
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
jobs=[(4,m) for m in (1024,2048,4096)]
(ROOT/'K4_EXPECTED_JOBS.json').write_text(json.dumps([{'line':'P1','k':k,'M':m,'b':1,'status':'CONDITIONAL_AUTHORIZED_AFTER_P1_K2_PASS'} for k,m in jobs],indent=2)+'\n')
print(json.dumps({'event':'k4_batch_start','n_jobs':len(jobs),'workers':3,'gpu':False}),flush=True)
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
    rows=[]
    for i,row in enumerate(pool.map(run,jobs),1):
        rows.append(row); print(json.dumps({'event':'k4_job_complete','done':i,'total':len(jobs),'M':row['M'],'tv':row['tv_grid']}),flush=True)
(ROOT/'K4_FINAL_STATE_SUMMARY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
