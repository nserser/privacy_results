from __future__ import annotations
import concurrent.futures,json,os,time,glob
from pathlib import Path
import numpy as np
from scalecert.cc_solver import Grid1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D,dpsgd_noise_std
from spectral import direct_response
BASE=Path(__file__).resolve().parents[1]
INPUT=BASE/'input/MEMFRONT2_REVIEW/RAW_SPECTRAL'
OUT=BASE/'memory/direct_extended';OUT.mkdir(parents=True,exist_ok=True)
LEDGER=BASE/'FINMEM1_JOB_LEDGER.jsonl'
def log(e):
 with LEDGER.open('a') as f:f.write(json.dumps({'time':time.time(),**e},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def configs():
 res=[]
 for f in glob.glob(str(INPUT/'*.json')):
  d=json.load(open(f));c=d['config']
  if c['line']=='P1':res.append((d,f[:-5]+'.npz'))
 return res
def run(arg):
 d,npzpath=arg;c={**d['config'],'max_r':16384};ident=d['id'];t=time.time()
 if (OUT/f'{ident}.npz').exists():return {'id':ident,'status':'EXISTS'}
 os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
 grid=Grid1D.make(c['x_max'],c['nx']);ker=DiscreteGaussianKernel1D(grid,c['m'],c['A'],c['B'],c['eta'],dpsgd_noise_std(c['eta'],c['B'],c['z']),drop_tol=0.,tail_sigma=c['tail_sigma'])
 K,lost,_,_=ker.kernel(0);K1=ker.kernel(1)[0]; old=np.load(npzpath);q=old['q']
 spec={'kernel':K,'kernel1':K1,'q':q,'kernel_obj':ker,'quasi_stationary_residual_l1':d['quasi_stationary_residual_l1']}
 def prog(v):log({'event':'memory_extended_checkpoint','id':ident,'r':v['r'],'max_r':c['max_r'],'tv_upper':v['tv_upper'],'escape_present':v['escape_present'],'escape_reference':v['escape_reference']})
 direct=direct_response(spec,c['max_r'],2048,prog)
 # Exact consistency check over the overlap with immutable MEMFRONT2 trajectory.
 n=min(len(old['r']),len(direct['tv_grid']));overlap=float(np.max(np.abs(old['r'][:n]-direct['tv_grid'][:n])))
 np.savez_compressed(OUT/f'{ident}.npz',r=direct['tv_grid'],tv_grid=direct['tv_grid'],tv_upper=direct['tv_upper'],escape_present=direct['escape_present'],escape_reference=direct['escape_reference'],q=q)
 meta={'id':ident,'line':c['line'],'grid':c['grid_label'],'eta':c['eta'],'z':c['z'],'x_max':c['x_max'],'nx':c['nx'],'max_r':c['max_r'],'overlap_max_abs_difference':overlap,'source_memfront2_id':ident,'elapsed_s':time.time()-t,'status':'PASS' if overlap<=1e-14 else 'OVERLAP_MISMATCH'}
 (OUT/f'{ident}.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');log({'event':'memory_extended_complete',**meta});return meta
def main():
 jobs=configs();log({'event':'memory_extended_start','jobs':len(jobs),'workers':min(4,os.cpu_count() or 1),'max_r':16384})
 with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as ex:out=list(ex.map(run,jobs))
 (OUT/'SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
