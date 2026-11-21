from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'input/CORE_SOURCE'
FS=ROOT/'input/PLD_accounting_git'
OUT=ROOT/'final_state/fs'; OUT.mkdir(parents=True,exist_ok=True)
LEDGER=ROOT/'FINMEM1_JOB_LEDGER.jsonl'
EXPECTED='fde83c45bf12c4f7821832c7d5d2ae2841fc0910'
sys.path.insert(0,str(CORE/'src'))

def log(x):
    with LEDGER.open('a') as f:
        f.write(json.dumps({'time_utc_epoch':time.time(),**x},sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())

def query(task):
    line,M,delta,z,k=task
    from scalecert.baseline_fs import gaussian_allocation_epsilon
    ident=hashlib.sha256(json.dumps({'line':line,'M':M,'delta':delta,'z':z,'k':k,'b':1},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    path=OUT/f'{ident}.json'
    if path.exists(): return json.loads(path.read_text())
    ans=gaussian_allocation_epsilon(delta=delta,sigma=z,num_steps=M,num_selected=k,num_epochs=1)
    row={'id':ident,'line':line,'M':M,'delta':delta,'sigma':z,'k':k,'b':1,'evidence_label':'EXTERNAL_ACCOUNTANT',**ans}
    path.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
    log({'event':'fs_complete','id':ident,'line':line,'M':M,'k':k,'delta':delta,'epsilon_lower':row['epsilon_lower'],'epsilon_upper':row['epsilon_upper']})
    return row

def main():
    ap=argparse.ArgumentParser(description='Resume pinned Feldman-Shenfeld accountant jobs from cached result files.')
    ap.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
    args=ap.parse_args()
    got=(FS/'PINNED_COMMIT.txt').read_text().strip()
    if got!=EXPECTED: raise SystemExit(f'FS commit mismatch: {got}')
    tasks=[(line,M,d,z,2) for line,z,Ms in [('P1',2.0,[256,512,1024,2048,4096,8192,16384]),('P2',4.0,[4096,8192,16384,32768,65536,131072,262144])] for M in Ms for d in (1e-3,1e-5)]
    tasks += [('P1',M,d,2.0,k) for k in (1,4) for M in (1024,2048,4096) for d in (1e-3,1e-5)]
    (OUT/'EXPECTED_JOBS.json').write_text(json.dumps([{'line':a,'M':m,'delta':d,'z':z,'k':k} for a,m,d,z,k in tasks],indent=2)+'\n')
    log({'event':'fs_batch_start','jobs':len(tasks),'workers':args.workers,'commit':got})
    os.environ['PYTHONPATH']=f"{CORE/'src'}:{FS}"
    sys.path.insert(0,str(FS))
    # Spawn workers with capped BLAS threads; the main process performs deterministic ordered aggregation.
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(query,tasks))
    (OUT/'SUMMARY.json').write_text(json.dumps({'status':'PASS','commit':got,'rows':rows},indent=2,sort_keys=True)+'\n')
    print(f'FS_COMPLETE {len(rows)}/{len(tasks)} commit={got}',flush=True)

if __name__=='__main__': main()
