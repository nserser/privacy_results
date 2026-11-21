from __future__ import annotations
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
import numpy as np
from .spectral import compute_spectrum,direct_response,spectral_reconstruct
BASE=Path(__file__).resolve().parent;OUT=BASE/'results/secondary';PROG=BASE/'MEMFRONT2_JOB_LEDGER.jsonl'
def log(x):
 with PROG.open('a') as f:f.write(json.dumps(x,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def cfgs():
 return [
 {'line':'R1','eta':.1,'z':8.}, {'line':'R2','eta':.05,'z':2.}, {'line':'R3','eta':.05,'z':4.},
 {'line':'R4','eta':.2,'z':2.}, {'line':'R5','eta':.2,'z':4.}]
def run(c):
 os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
 c={**c,'m':.5,'A':.55,'B':.05,'x_max':6.,'nx':1201,'tail_sigma':40.,'max_r':65536,'nmodes':32,'tol':1e-11,'grid_label':'BASE'}
 ident=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20];OUT.mkdir(parents=True,exist_ok=True);t=time.time();log({'event':'secondary_spectral_start','id':ident,'config':c})
 s=compute_spectrum(c,c['nmodes'],c['tol'],250000)
 def progress(d):log({'event':'secondary_direct_checkpoint','id':ident,**d})
 d=direct_response(s,c['max_r'],2048,progress)
 rs=np.arange(0,c['max_r']+1,16,dtype=int);pred=spectral_reconstruct(s,rs);obs=d['tv_grid'][rs];held=rs>=16384;er=np.abs(pred[held]-obs[held])/np.maximum(obs[held],1e-15)
 meta={'id':ident,'config':c,'status':'RAW_COMPLETE','grid':s['grid'],'lambda_perron':s['lambda_perron_real'],'quasi_stationary_residual_l1':s['quasi_stationary_residual_l1'],'perron_escape_per_step':s['perron_escape_per_step'],'modes':s['modes'],'direct_points':len(d['tv_grid']),'heldout_interval':[int(rs[held][0]),int(rs[held][-1])],'heldout_max_pointwise_relative_error':float(er.max()),'heldout_relative_l1_error':float(np.abs(pred[held]-obs[held]).sum()/obs[held].sum()),'elapsed_s':time.time()-t}
 np.savez_compressed(OUT/(ident+'.npz'),tv_grid=d['tv_grid'],tv_upper=d['tv_upper'],escape_present=d['escape_present'],escape_reference=d['escape_reference'],q=s['q'],eigenvalues=s['eigenvalues'],right_vectors=s['right_vectors'],left_vectors=s['left_vectors'],spectral_at_stride=pred,spectral_r=rs)
 (OUT/(ident+'.json')).write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');log({'event':'secondary_spectral_complete','id':ident,'line':c['line'],'elapsed_s':meta['elapsed_s'],'heldout_max_relative_error':meta['heldout_max_pointwise_relative_error']});return meta
def main():
 rows=cfgs();(BASE/'SECONDARY_EXPECTED_JOBS.json').write_text(json.dumps(rows,indent=2)+'\n');n=min(5,os.cpu_count() or 1);log({'event':'secondary_batch_start','jobs':len(rows),'workers':n})
 with concurrent.futures.ProcessPoolExecutor(max_workers=n) as ex:res=list(ex.map(run,rows))
 (BASE/'SECONDARY_SUMMARY.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
