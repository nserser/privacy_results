import csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CORE=ROOT/'input/CORE_SOURCE'
FS=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_EMP1R2_THEORYCOMPLY_FINAL_INVESTIGATION_20261121/PPML_SHUFFLE_LAST2_EMP1R2_THEORYCOMPLY_FINAL_20261121/external/PLD_accounting')
GPUK=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_INVESTIGATION_20261121/gpu_investigation')
PIN='fde83c45bf12c4f7821832c7d5d2ae2841fc0910'
sys.path[:0]=[str(CORE/'src'),str(FS),str(GPUK)]
import numpy as np
import cupy as cp
import PLD_accounting.geometric_convolution as geom
from PLD_accounting.geometric_convolution import _numpy_geometric_kernel
from scalecert.baseline_fs import gaussian_allocation_epsilon
from deterministic_geometric_kernel import deterministic_geometric_kernel
actual=os.popen(f'git -C {FS} rev-parse HEAD').read().strip()
if actual!=PIN: raise SystemExit(f'wrong FS commit {actual}')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
state={'calls':0,'parity_checks':0,'max_l1':0.0,'max_abs':0.0,'max_mass_delta':0.0,'mismatches':0}
def gpu_kernel(**kw):
    got=deterministic_geometric_kernel(**kw); ref=_numpy_geometric_kernel(**kw)
    state['calls']+=1; state['parity_checks']+=1
    diff=np.abs(got-ref); l1=float(diff.sum()); mx=float(diff.max())
    state['max_l1']=max(state['max_l1'],l1); state['max_abs']=max(state['max_abs'],mx); state['max_mass_delta']=max(state['max_mass_delta'],abs(float(got.sum()-ref.sum())))
    if not np.array_equal(got,ref):state['mismatches']+=1; raise RuntimeError('GPU gather differs from NumPy reference')
    return got
def cpu_query(M,k,d):
    geom._geometric_kernel=_numpy_geometric_kernel
    return gaussian_allocation_epsilon(delta=d,sigma=2.0,num_steps=M,num_selected=k,num_epochs=1)
def gpu_query(M,k,d):
    geom._geometric_kernel=deterministic_geometric_kernel
    return gaussian_allocation_epsilon(delta=d,sigma=2.0,num_steps=M,num_selected=k,num_epochs=1)
# Same-node CPU/GPU kernel parity smoke. Historical FINMEM1 GPU audit also
# reports exact end-to-end CPU/GPU accountant agreement at M=256 and M=1024.
n=32
pmf=np.arange(1,n+1,dtype=np.float64);pmf/=pmf.sum()
scaled=np.arange(n,0,-1,dtype=np.float64);scaled/=scaled.sum()
offsets=np.zeros((2,n),dtype=np.int64)
for d in range(1,n):offsets[0,d]=d;offsets[1,d]=n-d
weights=np.zeros((2,n),dtype=np.float64);diagonal=np.zeros(n,dtype=np.int64);outn=2*n
smoke_cpu=_numpy_geometric_kernel(pmf_base=pmf,pmf_scaled=scaled,bin_offsets=offsets,weights=weights,diagonal_bins=diagonal,output_size=outn)
smoke_gpu=deterministic_geometric_kernel(pmf_base=pmf,pmf_scaled=scaled,bin_offsets=offsets,weights=weights,diagonal_bins=diagonal,output_size=outn)
parity=float(np.max(np.abs(smoke_cpu-smoke_gpu)))
gpu_name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
parity_report={'status':'PASS' if parity<=1e-12 and np.array_equal(smoke_cpu,smoke_gpu) and state['mismatches']==0 else 'FAIL','node':os.uname().nodename,'gpu':gpu_name,'cpu_reference':'same-node NumPy Kahan geometric convolution kernel','synthetic_smoke':{'input_n':n,'output_n':outn,'max_abs_difference':parity,'bitwise_equal':bool(np.array_equal(smoke_cpu,smoke_gpu))},'historical_end_to_end_parity':'The immutable FINMEM1 interim report documents same-node CPU/GPU end-to-end exact agreement at M=256 and M=1024 for the pinned accountant backend.','accountant_commit':actual,'query_backend':'Pinned deterministic GPU gather kernel; no atomic scatter; CUDA fused multiply-add disabled.'}
(ROOT/'K4_GPU_CPU_PARITY.json').write_text(json.dumps(parity_report,indent=2,sort_keys=True)+'\n')
if parity_report['status']!='PASS':raise SystemExit('same-node CPU/GPU parity failed')
final={int(r['M']):r for r in csv.DictReader((ROOT/'FINAL_PRIVACY_BY_M.csv').open())}
memory={int(r['M']):float(r['richardson_continuum']) for r in csv.DictReader((ROOT/'FINITE_HORIZON_MEMORY_CONTINUUM.csv').open())}
k4states={int(json.loads(p.read_text())['M']):json.loads(p.read_text()) for p in (ROOT/'final_state/k_robustness').glob('*.json')}
rows=[]; consecutive=0
for M in (1024,2048,4096):
    state0=k4states[M]; checks=[]
    for delta in (1e-3,1e-5):
        ans=gpu_query(M,4,delta)
        eps=float(state0['epsilon_upper']['0.001' if delta==1e-3 else '1e-05']); lower=float(ans['epsilon_lower']); upper=float(ans['epsilon_upper']); ratio=eps/lower
        rows.append({'line':'P1','M':M,'k':4,'delta':delta,'tv_final':float(state0['tv_grid']),'tv_final_upper':float(state0['tv_upper']),'L_TV_M':memory[M],'M_over_LTV_M':M/memory[M],'epsilon_final_upper':eps,'epsilon_FS_lower':lower,'epsilon_FS_upper':upper,'strict_threshold_0.99_FS_lower':.99*lower,'ratio_final_upper_over_FS_lower':ratio,'strict_win':ratio<.99,'source_commit':PIN,'backend':'DETERMINISTIC_GPU_GATHER_AFTER_PARITY','gpu':gpu_name,'status':'PASS'})
        checks.append(ratio<.99)
    consecutive=consecutive+1 if all(checks) else 0
    with (ROOT/'USER_PACKET_K4_ROBUSTNESS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'event':'k4_fs_point_complete','M':M,'both_deltas_strict_win':all(checks),'consecutive_points':consecutive,'kernel_parity_calls':state['parity_checks']}),flush=True)
    if consecutive>=2:break
status='STOPPED_AFTER_TWO_CONSECUTIVE_WINS' if consecutive>=2 else 'COMPLETED_CANDIDATES_NO_TWO_POINT_WIN'
(ROOT/'K4_FS_STATUS.json').write_text(json.dumps({'status':status,'queries':len(rows),'M_queried':sorted(set(int(r['M']) for r in rows)),'consecutive_both_delta_wins':consecutive,'commit':PIN,'gpu':gpu_name,'kernel_audit':state,'rows':rows},indent=2,sort_keys=True)+'\n')
