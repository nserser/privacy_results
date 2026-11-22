from __future__ import annotations
import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import glob,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'input/CORE_SOURCE/src'
sys.path.insert(0,str(SRC))
from scalecert.cc_solver import Grid1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D,dpsgd_noise_std
from scalecert.occupancy import next_batch_probs
from scalecert.privacy import chi2_mass,privacy_profile_with_tv_defect,epsilon_upper_for_delta_with_tv_defect,epsilon_for_delta

RAW_DIR=ROOT/'input/MEMFRONT2_REVIEW/RAW_SPECTRAL'
OUT=ROOT/'final_state/k_robustness';OUT.mkdir(parents=True,exist_ok=True)
LEDGER=ROOT/'FINMEM1_JOB_LEDGER.jsonl'

def log(x):
    with LEDGER.open('a') as f:f.write(json.dumps({'time_utc_epoch':time.time(),**x},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def evolve(q0,K0,K1,lost0,lost1,M,k,ident,tag):
    states={k:np.asarray(q0,float).copy()};escaped=0.;initial=float(np.sum(q0))
    for t in range(M):
        nxt={}
        for r,v in states.items():
            for c,pr in next_batch_probs(r,M-t,1):
                K,lost=(K1,lost1) if c else (K0,lost0)
                out=np.asarray(K@v).ravel();escaped+=pr*float(np.dot(lost,v));out*=pr
                key=r-c
                if key in nxt:nxt[key]+=out
                else:nxt[key]=out.copy()
        states=nxt
        if (t+1)%1024==0 or t+1==M:log({'event':'krobust_progress','id':ident,'k':k,'tag':tag,'step':t+1,'M':M})
    if set(states)!={0}:raise RuntimeError('nonzero remaining occupancy states')
    p=states[0];le=initial-float(p.sum())-escaped
    return p,escaped,le

def run(job):
    k,M=job
    src=None
    for f in glob.glob(str(RAW_DIR/'*.json')):
        d0=json.load(open(f));c0=d0.get('config',{})
        if c0.get('line')=='P1' and c0.get('grid_label')=='BASE':src=Path(f[:-5]);break
    if src is None:raise RuntimeError('frozen P1 BASE kernel state not found')
    d=json.load(open(str(src)+'.json'));cfg=d['config'];npz=np.load(str(src)+'.npz');ident=hashlib.sha256(json.dumps({'line':'P1','grid':'BASE','M':M,'k':k,'b':1},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    path=OUT/f'{ident}.json'
    if path.exists():return json.loads(path.read_text())
    os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    grid=Grid1D.make(cfg['x_max'],cfg['nx']);ker=DiscreteGaussianKernel1D(grid,cfg['m'],cfg['A'],cfg['B'],cfg['eta'],dpsgd_noise_std(cfg['eta'],cfg['B'],cfg['z']),drop_tol=0.,tail_sigma=cfg['tail_sigma']);K0,l0,_,_=ker.kernel(0);K1,l1,_,_=ker.kernel(1);q0=npz['q']
    log({'event':'krobust_start','id':ident,'k':k,'M':M,'workers':True})
    p,ep,lp=evolve(q0,K0,K1,l0,l1,M,k,ident,'present');q=q0.copy();eq=0.
    for t in range(M):
        eq+=float(np.dot(l0,q));q=np.asarray(K0@q).ravel()
        if (t+1)%4096==0 or t+1==M:log({'event':'krobust_absent_progress','id':ident,'step':t+1,'M':M})
    massp=float(p.sum());massq=float(q.sum());dp=min(1.,max(0.,1-massp));dq=min(1.,max(0.,1-massq));pn=p/massp;qn=q/massq;tv=float(.5*np.abs(pn-qn).sum())
    eps={str(delta):epsilon_upper_for_delta_with_tv_defect(pn,qn,delta,dp,dq) for delta in (1e-3,1e-5)}
    row={'id':ident,'line':'P1','grid':'BASE','M':M,'k':k,'b':1,'status':'OK','evidence_label':'DIRECT_FULL_GRID','tv_grid':tv,'tv_upper':min(1.,tv+dp+dq),'epsilon_upper':eps,'present_mass_defect':dp,'absent_mass_defect':dq,'escaped_present':ep,'escaped_absent':eq,'present_latent_mass_error':lp}
    np.savez_compressed(OUT/f'{ident}.npz',present=pn,absent=qn)
    path.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');log({'event':'krobust_complete','id':ident,'k':k,'M':M,'tv':tv});return row

def main():
    jobs=[(k,M) for k in (1,4) for M in (1024,2048,4096)]
    (OUT/'EXPECTED_JOBS.json').write_text(json.dumps([{'k':k,'M':M} for k,M in jobs],indent=2)+'\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(run,jobs))
    (OUT/'SUMMARY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    print(f'K_ROBUST_COMPLETE {len(rows)}/{len(jobs)}',flush=True)
if __name__=='__main__': main()
