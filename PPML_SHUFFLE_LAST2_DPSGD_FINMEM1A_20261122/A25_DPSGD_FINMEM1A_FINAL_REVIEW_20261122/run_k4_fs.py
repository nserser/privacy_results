import csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CORE=ROOT/'input/CORE_SOURCE/src'
sys.path.insert(0,str(CORE))
from scalecert.baseline_fs import gaussian_allocation_epsilon
FSREPO=Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_EMP1R2_THEORYCOMPLY_FINAL_INVESTIGATION_20261121/PPML_SHUFFLE_LAST2_EMP1R2_THEORYCOMPLY_FINAL_20261121/external/PLD_accounting')
PIN='fde83c45bf12c4f7821832c7d5d2ae2841fc0910'
actual=os.popen(f'git -C {FSREPO} rev-parse HEAD').read().strip()
if actual!=PIN: raise SystemExit(f'FS commit mismatch {actual}')
sys.path.insert(0,str(FSREPO))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
final={int(r['M']):r for r in csv.DictReader((ROOT/'FINAL_PRIVACY_BY_M.csv').open())}
memory={int(r['M']):float(r['richardson_continuum']) for r in csv.DictReader((ROOT/'FINITE_HORIZON_MEMORY_CONTINUUM.csv').open())}
k4={int(json.loads(p.read_text())['M']):json.loads(p.read_text()) for p in (ROOT/'final_state/k_robustness').glob('*.json')}
rows=[]; consec=0
for M in (1024,2048,4096):
    if M not in k4: raise SystemExit(f'missing k4 final-state M={M}')
    state=k4[M]; lower_by_delta={}
    for delta in (1e-3,1e-5):
        ans=gaussian_allocation_epsilon(delta=delta,sigma=2.0,num_steps=M,num_selected=4,num_epochs=1)
        lower_by_delta[delta]=float(ans['epsilon_lower'])
        eps=float(state['epsilon_upper']['0.001' if delta==1e-3 else '1e-05'])
        ratio=eps/lower_by_delta[delta]
        row={'line':'P1','M':M,'k':4,'delta':delta,'tv_final':float(state['tv_grid']),'tv_final_upper':float(state['tv_upper']),'L_TV_M':memory[M],'M_over_LTV_M':M/memory[M],'epsilon_final_upper':eps,'epsilon_FS_lower':lower_by_delta[delta],'epsilon_FS_upper':float(ans['epsilon_upper']),'strict_threshold_0.99_FS_lower':.99*lower_by_delta[delta],'ratio_final_upper_over_FS_lower':ratio,'strict_win':ratio<.99,'source_commit':PIN,'backend':'CPU_PINNED_ACCOUNTANT','status':'PASS'}
        rows.append(row)
    if all(r['strict_win'] for r in rows if r['M']==M): consec+=1
    else: consec=0
    print(json.dumps({'event':'k4_fs_point','M':M,'both_deltas_strict_win':consec>0,'consecutive_win_points':consec}),flush=True)
    if consec>=2: break
with (ROOT/'USER_PACKET_K4_ROBUSTNESS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'K4_FS_STATUS.json').write_text(json.dumps({'status':'STOPPED_AFTER_TWO_CONSECUTIVE_WINS' if consec>=2 else 'COMPLETED_CANDIDATES_NO_TWO_POINT_WIN','queries':len(rows),'M_queried':sorted(set(int(r['M']) for r in rows)),'consecutive_both_delta_wins':consec,'commit':PIN,'rows':rows},indent=2,sort_keys=True)+'\n')
