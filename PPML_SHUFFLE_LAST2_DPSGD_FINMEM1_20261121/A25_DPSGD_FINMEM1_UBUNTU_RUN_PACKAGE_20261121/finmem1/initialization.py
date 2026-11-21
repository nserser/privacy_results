from __future__ import annotations
import concurrent.futures,glob,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'input/CORE_SOURCE/src';sys.path.insert(0,str(SRC))
from scalecert.cc_solver import Grid1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D,dpsgd_noise_std
from scalecert.occupancy import next_batch_probs
OUT=ROOT/'final_state/initialization';OUT.mkdir(parents=True,exist_ok=True);LEDGER=ROOT/'FINMEM1_JOB_LEDGER.jsonl'

def log(x):
 with LEDGER.open('a') as f:f.write(json.dumps({'time_utc_epoch':time.time(),**x},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def evolve(q0,K0,K1,l0,l1,M,ident,tag):
 st={2:q0.copy()};esc=0.;init=float(q0.sum())
 for t in range(M):
  nxt={}
  for r,v in st.items():
   for c,pr in next_batch_probs(r,M-t,1):
    K,lost=(K1,l1) if c else (K0,l0);out=np.asarray(K@v).ravel();esc+=pr*float(np.dot(lost,v));out*=pr;key=r-c
    if key in nxt:nxt[key]+=out
    else:nxt[key]=out.copy()
  st=nxt
  if (t+1)%1024==0 or t+1==M:log({'event':'init_progress','id':ident,'tag':tag,'step':t+1,'M':M})
 p=st[0];return p,esc,init-float(p.sum())-esc

def make_init(kind,x,m,A):
 if kind=='invariant':return None
 if kind=='continuous_stationary':
  u=.5*m*x*x+A*np.cos(x);v=np.exp(-(u-u.min()))
 elif kind in ('well_plus','well_minus'):
  root=brentq(lambda t:m*t-A*np.sin(t),0.1,1.1)
  center=root if kind=='well_plus' else -root
  v=np.exp(-.5*((x-center)/.1)**2)
 else:raise ValueError(kind)
 return v/v.sum()

def run(job):
 M,kind=job;paths=glob.glob(str(ROOT/'input/MEMFRONT2_REVIEW/RAW_SPECTRAL/*.json'));src=None
 for f in paths:
  d=json.load(open(f));c=d.get('config',{})
  if c.get('line')=='P1' and c.get('grid_label')=='BASE':src=Path(f[:-5]);break
 d=json.load(open(str(src)+'.json'));cfg=d['config'];npz=np.load(str(src)+'.npz');ident=hashlib.sha256(json.dumps({'line':'P1','grid':'BASE','M':M,'initialization':kind,'k':2},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
 dst=OUT/f'{ident}.json'
 if dst.exists():return json.loads(dst.read_text())
 xgrid=Grid1D.make(cfg['x_max'],cfg['nx']);x=xgrid.x
 from scalecert.discrete_kernel import DiscreteGaussianKernel1D,dpsgd_noise_std
 ker=DiscreteGaussianKernel1D(xgrid,cfg['m'],cfg['A'],cfg['B'],cfg['eta'],dpsgd_noise_std(cfg['eta'],cfg['B'],cfg['z']),drop_tol=0.,tail_sigma=cfg['tail_sigma']);K0,l0,_,_=ker.kernel(0);K1,l1,_,_=ker.kernel(1)
 q0=npz['q'] if kind=='invariant' else make_init(kind,x,cfg['m'],cfg['A'])
 log({'event':'init_start','id':ident,'M':M,'initialization':kind})
 p,ep,lp=evolve(q0,K0,K1,l0,l1,M,ident,'present');q=q0.copy();eq=0.
 for t in range(M):eq+=float(np.dot(l0,q));q=np.asarray(K0@q).ravel()
 mp=float(p.sum());mq=float(q.sum());dp=max(0.,1-mp);dq=max(0.,1-mq);pn=p/mp;qn=q/mq;tv=float(.5*np.abs(pn-qn).sum())
 from scalecert.privacy import epsilon_upper_for_delta_with_tv_defect
 eps={str(delta):epsilon_upper_for_delta_with_tv_defect(pn,qn,delta,dp,dq) for delta in (1e-3,1e-5)}
 row={'id':ident,'line':'P1','grid':'BASE','M':M,'k':2,'initialization':kind,'evidence_label':'DIRECT_FULL_GRID','tv_grid':tv,'tv_upper':tv+dp+dq,'epsilon_upper':eps,'present_defect':dp,'absent_defect':dq,'present_escaped':ep,'absent_escaped':eq,'status':'OK'}
 np.savez_compressed(OUT/f'{ident}.npz',present=pn,absent=qn);dst.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');log({'event':'init_complete','id':ident,'M':M,'initialization':kind,'tv':tv});return row

def main():
 jobs=[(M,k) for M in (1024,4096,16384) for k in ('continuous_stationary','well_plus','well_minus')]
 (OUT/'EXPECTED_JOBS.json').write_text(json.dumps([{'M':M,'initialization':k} for M,k in jobs],indent=2)+'\n')
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(run,jobs))
 (OUT/'SUMMARY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');print(f'INIT_COMPLETE {len(rows)}/{len(jobs)}',flush=True)
if __name__=='__main__':main()
