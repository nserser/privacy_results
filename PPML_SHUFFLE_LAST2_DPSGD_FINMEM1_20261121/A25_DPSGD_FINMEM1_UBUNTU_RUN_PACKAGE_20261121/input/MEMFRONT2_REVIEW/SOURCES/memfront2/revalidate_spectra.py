from __future__ import annotations
import json,glob
from pathlib import Path
import numpy as np
from .spectral import make_kernel,spectral_reconstruct
BASE=Path(__file__).resolve().parent
def main():
    for p in glob.glob(str(BASE/'results/**/*.json'),recursive=True):
        if p.endswith('summary.json') or p.endswith('SUMMARY.json'):continue
        try:d=json.loads(Path(p).read_text())
        except Exception:continue
        if 'modes' not in d or 'config' not in d or 'heldout_interval' not in d:continue
        npz=np.load(p[:-5]+'.npz');grid,ker=make_kernel(d['config']); K0=ker.kernel(0)[0];K1=ker.kernel(1)[0]
        s={'config':d['config'],'kernel':K0,'kernel1':K1,'q':npz['q'],'eigenvalues':npz['eigenvalues'],'right_vectors':npz['right_vectors'],'left_vectors':npz['left_vectors'],'lambda_perron_real':d['lambda_perron']}
        rs=npz['spectral_r'];pred=spectral_reconstruct(s,rs);obs=npz['r'] if 'r' in npz.files else npz['tv_grid'];obs=obs[rs]
        start,end=d['heldout_interval'];held=(rs>=start)&(rs<=end);err=np.abs(pred[held]-obs[held])/np.maximum(obs[held],1e-15)
        d['heldout_max_pointwise_relative_error_perron_corrected']=float(err.max())
        d['heldout_relative_l1_error_perron_corrected']=float(np.abs(pred[held]-obs[held]).sum()/max(obs[held].sum(),1e-300))
        d['spectral_perron_policy']='clamp invariant eigenvalue to 1; remove numerical Perron projection; separate exact Gaussian escape envelope'
        np.savez_compressed(p[:-5]+'.npz',**{k:npz[k] for k in npz.files if k not in ('spectral_at_stride',)},spectral_at_stride=pred)
        Path(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    print('Recomputed stored direct-vs-spectral validation with numerical Perron drift removed.')
if __name__=='__main__':main()
