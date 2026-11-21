from __future__ import annotations
import concurrent.futures,json,os,time,hashlib
from pathlib import Path
import numpy as np
from .spectral import compute_spectrum,direct_response,spectral_reconstruct
BASE=Path(__file__).resolve().parent; OUT=BASE/'results/spectral'; PROG=BASE/'MEMFRONT2_JOB_LEDGER.jsonl'
def log(d):
 with PROG.open('a') as f:f.write(json.dumps(d,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def job(c):
 os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
 ident=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]; t=time.time();log({'event':'grid2_start','id':ident,'config':c})
 s=compute_spectrum(c,c['nmodes'],c['tol'],350000)
 def progress(d):log({'event':'grid2_direct_checkpoint','id':ident,**d})
 d=direct_response(s,c['max_r'],checkpoint_every=2048,progress=progress)
 rs=np.arange(0,c['max_r']+1,16,dtype=int); pred=spectral_reconstruct(s,rs); obs=d['tv_grid'][rs]
 held=rs>= (2048 if c['line']=='P1' else 16384)
 err=np.abs(pred[held]-obs[held])/np.maximum(obs[held],1e-15)
 meta={'id':ident,'config':c,'status':'RAW_COMPLETE','grid':s['grid'],'lambda_perron':s['lambda_perron_real'],'quasi_stationary_residual_l1':s['quasi_stationary_residual_l1'],'perron_escape_per_step':s['perron_escape_per_step'],'modes':s['modes'],'heldout_interval':[int(rs[held][0]),int(rs[held][-1])],'heldout_max_pointwise_relative_error':float(err.max()),'heldout_relative_l1_error':float(np.abs(pred[held]-obs[held]).sum()/obs[held].sum()),'elapsed_s':time.time()-t}
 np.savez_compressed(OUT/(ident+'.npz'),r=d['tv_grid'],tv_upper=d['tv_upper'],escape_present=d['escape_present'],escape_reference=d['escape_reference'],q=s['q'],eigenvalues=s['eigenvalues'],right_vectors=s['right_vectors'],left_vectors=s['left_vectors'],spectral_at_stride=pred,spectral_r=rs)
 (OUT/(ident+'.json')).write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');log({'event':'grid2_complete','id':ident,'elapsed_s':meta['elapsed_s'],'heldout_max_relative_error':meta['heldout_max_pointwise_relative_error']});return meta
def main():
 cs=[]
 for line,z,maxr in [('P1',2.,4096),('P2',4.,65536)]:cs.append({'line':line,'eta':.1,'z':z,'m':.5,'A':.55,'B':.05,'x_max':6.,'nx':4801,'tail_sigma':40.,'max_r':maxr,'nmodes':32,'tol':1e-11,'grid_label':'GRID2'})
 (BASE/'GRID2_EXPECTED_JOBS.json').write_text(json.dumps(cs,indent=2)+'\n')
 n=min(2,os.cpu_count() or 1);log({'event':'grid2_batch_start','jobs':len(cs),'workers':n})
 with concurrent.futures.ProcessPoolExecutor(max_workers=n) as ex:
  results=list(ex.map(job,cs))
 (BASE/'GRID2_SUMMARY.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
