from __future__ import annotations
import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import numpy as np
from .spectral import compute_spectrum,direct_response,spectral_reconstruct

BASE=Path(__file__).resolve().parent
OUT=BASE/"results/spectral"
PROGRESS=BASE/"MEMFRONT2_JOB_LEDGER.jsonl"

def log(x):
    with PROGRESS.open('a') as f: f.write(json.dumps(x,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
def cfgs():
    rows=[]
    for line,z,maxr in [('P1',2.,4096),('P2',4.,65536)]:
        for grid,nx,xmax in [('BASE',1201,6.),('GRID',2401,6.),('DOMAIN',1601,8.)]:
            rows.append({'line':line,'eta':.1,'z':z,'m':.5,'A':.55,'B':.05,'x_max':xmax,'nx':nx,'tail_sigma':40.,'max_r':maxr,'nmodes':32,'tol':1e-11,'grid_label':grid})
    return rows
def jid(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
def run(c):
    os.environ.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    OUT.mkdir(parents=True,exist_ok=True); ident=jid(c); dest=OUT/(ident+'.npz'); meta=OUT/(ident+'.json')
    if meta.exists(): return json.loads(meta.read_text())
    t=time.time(); log({'event':'spectral_start','id':ident,'config':c})
    s=compute_spectrum(c,c['nmodes'],c['tol'],250000)
    log({'event':'spectrum_solved','id':ident,'lambda0':s['lambda_perron_real'],'modes':len(s['modes']),'q_residual':s['quasi_stationary_residual_l1'],'escape':s['perron_escape_per_step']})
    def prog(d):log({'event':'direct_memory_checkpoint','id':ident,**d})
    direct=direct_response(s,c['max_r'],checkpoint_every=2048,progress=prog)
    # Reconstruct at 16-step spacing on the response tail; use first half for
    # mode-count selection and the disjoint second half for validation.
    stride=16; rs=np.arange(0,c['max_r']+1,stride,dtype=int)
    pred=spectral_reconstruct(s,rs)
    obs=direct['tv_grid'][rs]
    train=rs<=c['max_r']//2; valid=~train
    point_err=np.abs(pred-obs)/np.maximum(obs,1e-15)
    tail_start=2048 if c['line']=='P1' else 16384
    held=(rs>=tail_start)&valid
    held_rel=point_err[held]
    result={'id':ident,'config':c,'status':'RAW_COMPLETE','grid':s['grid'],'lambda_perron':s['lambda_perron_real'],
      'quasi_stationary_residual_l1':s['quasi_stationary_residual_l1'],'perron_escape_per_step':s['perron_escape_per_step'],
      'modes':s['modes'],'direct_points':len(direct['tv_grid']),'spectral_stride':stride,
      'training_interval':[int(rs[train][0]),int(rs[train][-1])], 'heldout_interval':[int(rs[held][0]),int(rs[held][-1])] if np.any(held) else None,
      'heldout_max_pointwise_relative_error':float(np.max(held_rel)) if len(held_rel) else None,
      'heldout_median_pointwise_relative_error':float(np.median(held_rel)) if len(held_rel) else None,
      'heldout_relative_l1_error':float(np.sum(np.abs(pred[held]-obs[held]))/max(np.sum(obs[held]),1e-300)) if np.any(held) else None,
      'heldout_explained_fraction_l1':float(1-np.sum(np.abs(pred[held]-obs[held]))/max(np.sum(obs[held]),1e-300)) if np.any(held) else None,
      'projection_condition_max':float(max(m['biorthogonal_condition_indicator'] for m in s['modes'])),
      'elapsed_s':time.time()-t}
    np.savez_compressed(dest,r=direct['tv_grid'],tv_upper=direct['tv_upper'],escape_present=direct['escape_present'],escape_reference=direct['escape_reference'],q=s['q'],eigenvalues=s['eigenvalues'],right_vectors=s['right_vectors'],left_vectors=s['left_vectors'],spectral_at_stride=pred,spectral_r=rs)
    meta.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    log({'event':'spectral_job_complete','id':ident,'status':result['status'],'elapsed_s':result['elapsed_s'],'heldout_max_relative_error':result['heldout_max_pointwise_relative_error']})
    return result
def main():
    rows=cfgs(); (BASE/'SPECTRAL_EXPECTED_JOB_MANIFEST.json').write_text(json.dumps({'jobs':[{'id':jid(c),'config':c} for c in rows]},indent=2,sort_keys=True)+'\n')
    n=min(6,os.cpu_count() or 1); log({'event':'spectral_batch_start','jobs':len(rows),'workers':n})
    outs=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as ex:
        fs=[ex.submit(run,c) for c in rows]
        for f in concurrent.futures.as_completed(fs): outs.append(f.result()); log({'event':'spectral_batch_progress','completed':len(outs),'total':len(rows),'id':outs[-1]['id']})
    (BASE/'SPECTRAL_BATCH_SUMMARY.json').write_text(json.dumps(sorted(outs,key=lambda x:(x['config']['line'],x['config']['grid_label'])),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
