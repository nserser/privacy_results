import sys,time,json
from pathlib import Path
import numpy as np
PKG=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_MEMFRONT2_INVESTIGATION_20261121/privacy_results/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_20261121/A25_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121')
sys.path.insert(0,str(PKG/'input/PLD_accounting_git'))
from PLD_accounting.geometric_convolution import _numpy_geometric_kernel,_pairwise_sum_grid_offsets,_geometric_output_extent,_bin_offsets_and_weights
from PLD_accounting.types import BoundType,ProjectionMethod
from deterministic_geometric_kernel import deterministic_geometric_kernel

def run(n,method):
 x=np.linspace(-5,5,n);a=np.exp(-.5*(x/1.2)**2);b=np.exp(-.5*((x-.15)/1.1)**2);a/=a.sum();b/=b.sum();step=.005
 off=_pairwise_sum_grid_offsets(origin_1=.1,origin_2=.12,log_ratio=step,num_bins=n,smallest_sum_offset=0.,diagonal_offset=0.)
 mi,on=_geometric_output_extent(grid_offsets=off,size_1=n,size_2=n,diagonal_shifts=None,bound_type=BoundType.DOMINATES)
 bins,w=_bin_offsets_and_weights(grid_offsets=off,minimum_offset=mi,log_ratio=step,bound_type=BoundType.DOMINATES,projection_method=method);diag=np.full(n,bins[0,0],dtype=np.int64)
 kw=dict(pmf_base=a,pmf_scaled=b,bin_offsets=bins,weights=w,diagonal_bins=diag,output_size=on)
 t=time.perf_counter();ref=_numpy_geometric_kernel(**kw);cpu=time.perf_counter()-t
 deterministic_geometric_kernel(**kw)
 t=time.perf_counter();got=deterministic_geometric_kernel(**kw);gpu=time.perf_counter()-t
 return {'n':n,'method':method.value,'cpu_s':cpu,'gpu_s':gpu,'speedup':cpu/gpu,'l1':float(np.abs(ref-got).sum()),'max_abs':float(np.max(np.abs(ref-got))),'mass_delta':float(got.sum()-ref.sum()),'nonnegative':bool(got.min()>=0),'weight_nonzero':int(np.count_nonzero(w))}
if __name__=='__main__':
 rows=[run(n,m) for n in (4096,8192,16869) for m in (ProjectionMethod.STOCH_DOM,)]
 print(json.dumps({'results':rows},indent=2))
