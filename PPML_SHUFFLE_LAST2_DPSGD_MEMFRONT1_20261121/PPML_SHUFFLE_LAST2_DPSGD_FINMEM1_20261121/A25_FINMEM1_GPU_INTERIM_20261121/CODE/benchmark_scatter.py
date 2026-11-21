import os, sys, time, json
import numpy as np
import cupy as cp
from pathlib import Path
PKG=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_MEMFRONT2_INVESTIGATION_20261121/privacy_results/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_20261121/A25_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121')
sys.path.insert(0,str(PKG/'input/PLD_accounting_git'))
from PLD_accounting.geometric_convolution import (_numpy_geometric_kernel,_pairwise_sum_grid_offsets,_geometric_output_extent,_bin_offsets_and_weights)
from PLD_accounting.types import BoundType,ProjectionMethod
CUDA=r'''extern "C" __global__ void pair_scatter(
 const double* a,const double* b,const long long* offsets,const double* weights,
 const long long* diagonal,long long n,long long outn,double* out) {
 long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 long long nn=n*n; if(idx>=nn) return;
 long long i=idx/n, j=idx-i*n;
 long long row,d,low; double mass;
 if(i==j){
   mass=a[i]*b[i];
   if(weights[0]==0.0){ atomicAdd(out+i+diagonal[i],mass); return; }
   row=0; d=0; low=i;
 } else if(i<j){
   row=0; d=j-i; low=i; mass=a[i]*b[j];
 } else {
   row=1; d=i-j; low=j; mass=a[i]*b[j];
 }
 long long at=low+offsets[row*n+d]; double w=weights[row*n+d];
 atomicAdd(out+at,mass*(1.0-w));
 if(w>0.0) atomicAdd(out+at+1,mass*w);
}
'''
kern=cp.RawKernel(CUDA,'pair_scatter',options=('--std=c++11',))
def run(n):
 rng=np.random.default_rng(20261121+n)
 # Smooth, nonnegative synthetic PLDs with finite tails and occasional exact zeros.
 x=np.linspace(-5,5,n); a=np.exp(-0.5*(x/1.2)**2); b=np.exp(-0.5*((x-.15)/1.1)**2)
 a[a<1e-14]=0; b[b<1e-14]=0
 a=a/a.sum(); b=b/b.sum()
 step=0.005; off=_pairwise_sum_grid_offsets(origin_1=.1,origin_2=.12,log_ratio=step,num_bins=n,smallest_sum_offset=0.,diagonal_offset=0.)
 minoff,outn=_geometric_output_extent(grid_offsets=off,size_1=n,size_2=n,diagonal_shifts=None,bound_type=BoundType.DOMINATES)
 bins,weights=_bin_offsets_and_weights(grid_offsets=off,minimum_offset=minoff,log_ratio=step,bound_type=BoundType.DOMINATES,projection_method=ProjectionMethod.STOCH_DOM)
 diagonal=np.full(n,bins[0,0],dtype=np.int64)
 t=time.perf_counter(); ref=_numpy_geometric_kernel(pmf_base=a,pmf_scaled=b,bin_offsets=bins,weights=weights,diagonal_bins=diagonal,output_size=outn); cpu=time.perf_counter()-t
 transfer_start=time.perf_counter()
 da,db,do,dw,dd=[cp.asarray(q) for q in (a,b,bins.astype(np.int64),weights.astype(np.float64),diagonal)]
 out=cp.zeros(outn,dtype=cp.float64); threads=256; blocks=(n*n+threads-1)//threads
 cp.cuda.Stream.null.synchronize(); transfer_s=time.perf_counter()-transfer_start
 # warm compilation
 kern((blocks,),(threads,),(da,db,do,dw,dd,np.int64(n),np.int64(outn),out)); cp.cuda.Stream.null.synchronize()
 out.fill(0); start=cp.cuda.Event();end=cp.cuda.Event();start.record()
 kern((blocks,),(threads,),(da,db,do,dw,dd,np.int64(n),np.int64(outn),out));end.record();end.synchronize(); gpu_ms=cp.cuda.get_elapsed_time(start,end)
 copy_start=time.perf_counter(); got=cp.asnumpy(out); copy_s=time.perf_counter()-copy_start; l1=float(np.abs(ref-got).sum()); maxabs=float(np.max(np.abs(ref-got))); mass_cpu=float(ref.sum());mass_gpu=float(got.sum())
 return {'n':n,'output_n':int(outn),'cpu_numpy_seconds':cpu,'gpu_kernel_ms':gpu_ms,'h2d_seconds':transfer_s,'d2h_seconds':copy_s,'gpu_end_to_end_seconds':transfer_s+gpu_ms/1000+copy_s,'speedup_end_to_end':cpu/(transfer_s+gpu_ms/1000+copy_s),'speedup_kernel_only':cpu/(gpu_ms/1000),'l1_abs':l1,'l1_relative_to_mass':l1/mass_cpu,'max_bin_abs':maxabs,'mass_cpu':mass_cpu,'mass_gpu':mass_gpu,'mass_diff':mass_gpu-mass_cpu,'nonnegative':bool(got.min()>=0)}
if __name__=='__main__':
 result={'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'cupy':cp.__version__,'rows':[run(n) for n in (256,512,1024,2048,4096)]}
 print(json.dumps(result,indent=2))
