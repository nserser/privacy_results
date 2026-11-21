import sys,time,json
from pathlib import Path
import numpy as np
PKG=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_MEMFRONT2_INVESTIGATION_20261121/privacy_results/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_20261121/A25_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121')
CORE=PKG/'input/CORE_SOURCE';sys.path.insert(0,str(CORE/'src'));sys.path.insert(0,str(PKG/'input/PLD_accounting_git'));sys.path.insert(0,'/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/gpu_investigation')
import PLD_accounting.geometric_convolution as geom
from PLD_accounting.geometric_convolution import _numpy_geometric_kernel
from scalecert.baseline_fs import gaussian_allocation_epsilon
from deterministic_geometric_kernel import deterministic_geometric_kernel
state={'calls':0,'parity_checks':0,'max_l1':0.0,'max_abs':0.0,'max_mass_delta':0.0,'gpu_seconds':0.0,'mismatch_calls':[]}
def gpu_kernel(**kw):
 t=time.perf_counter();got=deterministic_geometric_kernel(**kw);state['gpu_seconds']+=time.perf_counter()-t;state['calls']+=1
 ref=_numpy_geometric_kernel(**kw);diff=np.abs(got-ref);l1=float(diff.sum());mx=float(diff.max());state['max_l1']=max(state['max_l1'],l1);state['max_abs']=max(state['max_abs'],mx);state['max_mass_delta']=max(state['max_mass_delta'],abs(float(got.sum()-ref.sum())));state['parity_checks']+=1
 if not np.array_equal(got,ref):state['mismatch_calls'].append({'call':state['calls'],'n':int(len(kw['pmf_base'])),'output_n':int(kw['output_size']),'l1':l1,'max_abs':mx,'weight_nonzero':int(np.count_nonzero(kw['weights'])),'diag_unique':int(np.unique(kw['diagonal_bins']).size)})
 return got
KEYS={(256,2.0,1e-3):'f22e16eeea134c7cce32',(4096,2.0,1e-5):'2bda0b35d2232a657212',(4096,4.0,1e-5):'cf7ead0fa9f468ccb5e8'}
def one(M,z,delta):
 expected=json.loads((PKG/'final_state/fs'/f'{KEYS[(M,z,delta)]}.json').read_text())
 old=geom._geometric_kernel;geom._geometric_kernel=gpu_kernel
 samples=[]
 for _ in range(2):
  t=time.perf_counter();ans=gaussian_allocation_epsilon(delta=delta,sigma=z,num_steps=M,num_selected=2,num_epochs=1);wall=time.perf_counter()-t
  samples.append({'result':ans,'wall_s':wall})
 geom._geometric_kernel=old
 diffs=[{'upper':q['result']['epsilon_upper']-expected['epsilon_upper'],'lower':q['result']['epsilon_lower']-expected['epsilon_lower']} for q in samples]
 repeat={'upper':samples[1]['result']['epsilon_upper']-samples[0]['result']['epsilon_upper'],'lower':samples[1]['result']['epsilon_lower']-samples[0]['result']['epsilon_lower']}
 return {'M':M,'z':z,'k':2,'delta':delta,'cpu_cached':{'lower':expected['epsilon_lower'],'upper':expected['epsilon_upper']},'gpu_samples':samples,'gpu_minus_cpu':diffs,'gpu_repeat_delta':repeat}
if __name__=='__main__':
 import cupy as cp
 rows=[one(4096,2.0,1e-5)]
 print(json.dumps({'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'cupy':cp.__version__,'rows':rows,'kernel_audit':state},indent=2,sort_keys=True))
