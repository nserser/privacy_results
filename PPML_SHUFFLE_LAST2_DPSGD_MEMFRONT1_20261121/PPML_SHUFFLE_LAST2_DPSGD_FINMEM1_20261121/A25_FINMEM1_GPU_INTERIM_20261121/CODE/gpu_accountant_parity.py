import os,sys,time,json
from pathlib import Path
import numpy as np
import cupy as cp
CORE=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121/input/CORE_SOURCE')
PKG=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_MEMFRONT2_INVESTIGATION_20261121/privacy_results/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_20261121/A25_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121')
sys.path.insert(0,str(CORE/'src'));sys.path.insert(0,str(PKG/'input/PLD_accounting_git'))
import PLD_accounting.geometric_convolution as geom
from scalecert.baseline_fs import gaussian_allocation_epsilon
CUDA=r'''extern "C" __global__ void pair_scatter(const double* a,const double* b,const long long* offsets,const double* weights,const long long* diagonal,long long n,long long outn,double* out) {
 long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x; if(idx>=n*n) return;
 long long i=idx/n,j=idx-i*n,row,d,low; double mass;
 if(i==j){mass=a[i]*b[i];if(weights[0]==0.0){atomicAdd(out+i+diagonal[i],mass);return;}row=0;d=0;low=i;}
 else if(i<j){row=0;d=j-i;low=i;mass=a[i]*b[j];}
 else {row=1;d=i-j;low=j;mass=a[i]*b[j];}
 long long at=low+offsets[row*n+d];double w=weights[row*n+d];
 atomicAdd(out+at,mass*(1.0-w));if(w>0.0)atomicAdd(out+at+1,mass*w);
}
'''
kern=cp.RawKernel(CUDA,'pair_scatter',options=('--std=c++11',))
orig=geom._geometric_kernel
state={'calls':0,'parity_checked':0,'max_l1':0.0,'max_mass_error':0.0,'max_abs':0.0,'gpu_kernel_s':0.0,'copy_s':0.0,'input_bins':[]}
def gpu_kernel(*,pmf_base,pmf_scaled,bin_offsets,weights,diagonal_bins,output_size):
 a=np.ascontiguousarray(pmf_base,dtype=np.float64);b=np.ascontiguousarray(pmf_scaled,dtype=np.float64)
 bins=np.ascontiguousarray(bin_offsets,dtype=np.int64);w=np.ascontiguousarray(weights,dtype=np.float64);diag=np.ascontiguousarray(diagonal_bins,dtype=np.int64)
 t=time.perf_counter();da,db,dbins,dw,dd=[cp.asarray(q) for q in (a,b,bins,w,diag)];out=cp.zeros(output_size,dtype=cp.float64);cp.cuda.Stream.null.synchronize();state['copy_s']+=time.perf_counter()-t
 n=a.size;state['input_bins'].append(int(n));threads=256;blocks=(n*n+threads-1)//threads;start=cp.cuda.Event();end=cp.cuda.Event();start.record();kern((blocks,),(threads,),(da,db,dbins,dw,dd,np.int64(n),np.int64(output_size),out));end.record();end.synchronize();state['gpu_kernel_s']+=cp.cuda.get_elapsed_time(start,end)/1000
 result=cp.asnumpy(out);state['calls']+=1
 # Verify first two representative merge outputs with unchanged CPU reference.
 if state['parity_checked']<2:
  ref=geom._numpy_geometric_kernel(pmf_base=a,pmf_scaled=b,bin_offsets=bins,weights=w,diagonal_bins=diag,output_size=output_size)
  l1=float(np.abs(result-ref).sum());mass=abs(float(result.sum()-ref.sum()));mx=float(np.max(np.abs(result-ref)))
  state['max_l1']=max(state['max_l1'],l1);state['max_mass_error']=max(state['max_mass_error'],mass);state['max_abs']=max(state['max_abs'],mx);state['parity_checked']+=1
 return result

def one(M,z,delta):
 saved=geom._geometric_kernel;geom._geometric_kernel=gpu_kernel
 t=time.perf_counter();gpu=gaussian_allocation_epsilon(delta=delta,sigma=z,num_steps=M,num_selected=2,num_epochs=1);gpu_s=time.perf_counter()-t
 t=time.perf_counter();gpu2=gaussian_allocation_epsilon(delta=delta,sigma=z,num_steps=M,num_selected=2,num_epochs=1);gpu2_s=time.perf_counter()-t
 geom._geometric_kernel=saved
 keys={(256,2.0,1e-3):'f22e16eeea134c7cce32',(4096,2.0,1e-5):'2bda0b35d2232a657212',(4096,4.0,1e-5):'cf7ead0fa9f468ccb5e8'}
 cpu_path=PKG/'final_state/fs'/f"{keys[(M,z,delta)]}.json"
 cpu=json.loads(cpu_path.read_text())
 return {'M':M,'z':z,'k':2,'delta':delta,'cpu_seconds':None,'gpu_seconds':gpu_s,'speedup':None,'cpu_cached_result':cpu,'gpu':gpu,'gpu_repeat':gpu2,'gpu_repeat_seconds':gpu2_s,'repeat_upper_diff':gpu2['epsilon_upper']-gpu['epsilon_upper'],'repeat_lower_diff':gpu2['epsilon_lower']-gpu['epsilon_lower'],'epsilon_upper_diff':gpu['epsilon_upper']-cpu['epsilon_upper'],'epsilon_lower_diff':gpu['epsilon_lower']-cpu['epsilon_lower']}
if __name__=='__main__':
 rows=[]
 for args in [(4096,2.0,1e-5),(4096,4.0,1e-5)]: rows.append(one(*args))
 print(json.dumps({'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'cupy':cp.__version__,'rows':rows,'kernel_checks':state},indent=2,sort_keys=True))
