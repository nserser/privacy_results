from __future__ import annotations
import concurrent.futures,json,os,time,glob,math,hashlib
from pathlib import Path
import numpy as np
from scalecert.cc_solver import Grid1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D,dpsgd_noise_std
from scalecert.occupancy import next_batch_probs
from scalecert.hidden_support import evolve_hidden_epoch
from scalecert.privacy import chi2_mass,privacy_profile_with_tv_defect,epsilon_upper_for_delta_with_tv_defect,epsilon_for_delta
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'input/MEMFRONT2_REVIEW/RAW_SPECTRAL'; OUT=ROOT/'final_state/jobs';OUT.mkdir(parents=True,exist_ok=True); LEDGER=ROOT/'FINMEM1_JOB_LEDGER.jsonl'
M_P1=[256,512,1024,2048,4096,8192,16384]
M_P2=[4096,8192,16384,32768,65536,131072,262144]
CONFIGS={}
for f in glob.glob(str(RAW/'*.json')):
 d=json.load(open(f))
 if not isinstance(d,dict) or 'config' not in d:continue
 c=d['config'];CONFIGS[(c['line'],c['grid_label'])]=(c,d,Path(f[:-5]+'.npz'))

def log(x):
 with LEDGER.open('a') as f:f.write(json.dumps({'time_utc_epoch':time.time(),**x},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def progress_evolve(q0,K0,K1,lost0,lost1,M,k,ident,tag,interval=1024):
 states={int(k):np.asarray(q0,float).copy()};escaped=0.; initial=float(np.sum(q0));
 for t in range(M):
  S=M-t;nxt={}
  for r,v in states.items():
   for c,pr in next_batch_probs(r,S,1):
    K,lost=(K1,lost1) if c else (K0,lost0)
    out=np.asarray(K@v).ravel(); e=float(np.dot(lost,v));escaped+=pr*e;out*=pr
    key=r-c
    if key in nxt:nxt[key]+=out
    else:nxt[key]=out.copy()
  states=nxt
  if (t+1)%interval==0 or t+1==M:log({'event':'final_state_progress','id':ident,'tag':tag,'step':t+1,'M':M,'state_count':len(states),'mass':float(sum(v.sum() for v in states.values())),'escaped_cumulative':escaped})
 if set(states)!={0}:raise RuntimeError(f'terminal latent states={list(states)}')
 final=states[0];le=initial-float(final.sum())-escaped
 return final,escaped,le

def absent_evolve(q0,K0,lost0,M,ident,tag,interval=4096):
 q=np.asarray(q0,float).copy(); escaped=0.
 for t in range(M):
  escaped+=float(np.dot(lost0,q));q=np.asarray(K0@q).ravel()
  if (t+1)%interval==0 or t+1==M:log({'event':'absent_progress','id':ident,'tag':tag,'step':t+1,'M':M,'mass':float(q.sum()),'escaped_cumulative':escaped})
 le=float(q0.sum())-float(q.sum())-escaped
 return q,escaped,le

def run(job):
 line,grid,M=job; c0,d,npzpath=CONFIGS[(line,grid)];cfg=dict(c0);cfg['M']=M;cfg['k']=2;cfg['b']=1; ident=hashlib.sha256(json.dumps({'line':line,'grid':grid,'M':M,'k':2,'b':1},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20];dst=OUT/f'{ident}.json';
 if dst.exists():return json.load(open(dst))
 os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
 start=time.time(); log({'event':'final_state_start','id':ident,'line':line,'grid':grid,'M':M,'k':2,'workers_pool':True})
 x_max=float(cfg['x_max']);nx=int(cfg['nx']);grid1=Grid1D.make(x_max,nx);eta=float(cfg['eta']);B=float(cfg['B']);z=float(cfg['z']);ker=DiscreteGaussianKernel1D(grid1,float(cfg['m']),float(cfg['A']),B,eta,dpsgd_noise_std(eta,B,z),drop_tol=0.,tail_sigma=float(cfg['tail_sigma']))
 K0,lost0,_,_=ker.kernel(0);K1,lost1,_,_=ker.kernel(1);q0=np.load(npzpath)['q']
 # Recurrence wrapper is checked against frozen exact hidden-support implementation in smoke test.
 p,escp,lep=progress_evolve(q0,K0,K1,lost0,lost1,M,2,ident,'present')
 q,escq,leq=absent_evolve(q0,K0,lost0,M,ident,'absent')
 massp=float(p.sum());massq=float(q.sum());defp=min(1.,max(0.,1.-massp));defq=min(1.,max(0.,1.-massq))
 if massp<=0 or massq<=0:raise RuntimeError('nonpositive survived mass')
 pn=p/massp;qn=q/massq
 tv=float(.5*np.abs(pn-qn).sum());tv_upper=min(1.,tv+defp+defq)
 try: cf=chi2_mass(pn,qn)
 except Exception:cf=float('inf')
 try: cr=chi2_mass(qn,pn)
 except Exception:cr=float('inf')
 eps={};profile=[]
 for delta in [1e-3,1e-5]:
  try:low=epsilon_for_delta(pn,qn,delta)
  except Exception:low=float('inf')
  try:up=epsilon_upper_for_delta_with_tv_defect(pn,qn,delta,defp,defq)
  except Exception:up=float('inf')
  eps[str(delta)]={'epsilon_conditional_grid':low,'epsilon_final_upper_with_mass_defect':up}
 # sparse eps grid plus exact targets; conservative bounds at targets
 profile=privacy_profile_with_tv_defect(pn,qn,[0.,.1,.25,.5,1.,2.,4.],defp,defq)
 meta={'id':ident,'line':line,'grid':grid,'M':M,'k':2,'b':1,'evidence_label':'DIRECT_FULL_GRID','status':'OK','tv_symmetric_grid':tv,'tv_symmetric_upper_mass_defect':tv_upper,'chi2_forward':cf,'chi2_reverse':cr,'epsilon':eps,'privacy_profile_grid':profile,'present_survived_mass':massp,'absent_survived_mass':massq,'present_escaped_mass':escp,'absent_escaped_mass':escq,'present_latent_mass_error':lep,'absent_latent_mass_error':leq,'present_total_mass_defect':defp,'absent_total_mass_defect':defq,'elapsed_s':time.time()-start,'config':cfg}
 np.savez_compressed(OUT/f'{ident}.npz',present_raw=p,absent_raw=q,present_conditional=pn,absent_conditional=qn)
 dst.write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');log({'event':'final_state_complete','id':ident,'line':line,'grid':grid,'M':M,'tv':tv,'tv_upper':tv_upper,'eps1e3':eps['0.001']['epsilon_final_upper_with_mass_defect'],'eps1e5':eps['1e-05']['epsilon_final_upper_with_mass_defect'],'elapsed_s':meta['elapsed_s']});return meta

def smoke():
 line='P1';grid='BASE';M=24;c,d,npzpath=CONFIGS[(line,grid)];x=Grid1D.make(c['x_max'],c['nx']);ker=DiscreteGaussianKernel1D(x,c['m'],c['A'],c['B'],c['eta'],dpsgd_noise_std(c['eta'],c['B'],c['z']),drop_tol=0.,tail_sigma=c['tail_sigma']);K0,l0,_,_=ker.kernel(0);K1,l1,_,_=ker.kernel(1);q=np.load(npzpath)['q']
 fast,ef,lf=progress_evolve(q,K0,K1,l0,l1,M,2,'SMOKE','fast',interval=1000);ref=evolve_hidden_epoch(q,ker.transition,M,1,2)
 err=float(np.max(np.abs(fast-ref.mass))); masserr=abs(ef-ref.escaped_mass)+abs(lf-ref.latent_mass_error)
 return {'M':M,'max_abs_mass_diff':err,'escaped_latent_diff':masserr,'pass':err<=1e-14 and masserr<=1e-14}

def main():
 s=smoke();(ROOT/'final_state/RECURRENCE_SMOKE.json').write_text(json.dumps(s,indent=2)+'\n');print('smoke',s)
 if not s['pass']:raise SystemExit('progress recurrence differs from frozen hidden-support implementation')
 jobs=[]
 for M in M_P1:jobs.append(('P1','BASE',M))
 for M in M_P2:jobs.append(('P2','BASE',M))
 # convergence anchors in all requested refinements; overlap jobs deduplicate by exact id/config.
 for line,anchors in [('P1',[1024,4096,16384]),('P2',[8192,32768,65536])]:
  for grid in ['GRID','GRID2','DOMAIN']:
   for M in anchors:jobs.append((line,grid,M))
 dedup=list(dict.fromkeys(jobs));(ROOT/'final_state/EXPECTED_JOBS.json').write_text(json.dumps([{'line':a,'grid':b,'M':m,'k':2,'b':1} for a,b,m in dedup],indent=2)+'\n')
 n=min(6,os.cpu_count() or 1);log({'event':'final_state_batch_start','jobs':len(dedup),'workers':n,'gpu':False})
 with concurrent.futures.ProcessPoolExecutor(max_workers=n) as ex:out=list(ex.map(run,dedup))
 (ROOT/'final_state/SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'jobs':len(out),'workers':n,'completed':sum(x['status']=='OK' for x in out)},indent=2))
if __name__=='__main__':main()
