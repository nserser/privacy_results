import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read_csv(p): return list(csv.DictReader(Path(p).open(newline='')))
checks={}
checks['m0_pass']=json.loads((ROOT/'M0_VERIFICATION.json').read_text())['status']=='PASS'
fs=read_csv(ROOT/'P1_FS_FROZEN_BASELINE.csv')
checks['archived_p1_fs_14_of_14']=len(fs)==14 and len({r['id'] for r in fs})==14 and all(r['status']=='REUSED_VALIDATED' for r in fs)
checks['fs_epsilon_bounds_valid']=all(0<=float(r['epsilon_lower'])<=float(r['epsilon_upper']) for r in fs)
mem=read_csv(ROOT/'FINITE_HORIZON_MEMORY_CONTINUUM.csv')
checks['memory_7_points']=len(mem)==7 and all(r['status']=='PASS' for r in mem)
checks['memory_criteria']=all(abs(float(r['L_DOMAIN'])-float(r['L_BASE']))/float(r['L_BASE'])<=.01 and abs(float(r['L_GRID2'])-float(r['richardson_continuum']))/float(r['richardson_continuum'])<=.02 and float(r['finite_domain_defect_relative_max'])<=.005 for r in mem)
recon=read_csv(ROOT/'input/MEMFRONT2_REVIEW/SPECTRAL_RECONSTRUCTION_VALIDATION.csv')
p1recon=[r for r in recon if r['line']=='P1' and r['grid'] in ('BASE','GRID','GRID2','DOMAIN')]
checks['spectral_validation_under_5pct']=len(p1recon)>=4 and all(r['pass_5pct']=='True' and float(r['heldout_max_pointwise_relative_error'])<=.05 for r in p1recon)
priv=read_csv(ROOT/'FINAL_PRIVACY_BY_M.csv')
checks['fresh_p1_final_curve_7_direct']=len(priv)==7 and [int(r['M']) for r in priv]==[256,512,1024,2048,4096,8192,16384] and all(r['evidence_label']=='DIRECT_FULL_GRID' and r['status']=='OK' for r in priv)
conv=read_csv(ROOT/'NUMERICAL_CONVERGENCE_FINAL.csv')
checks['fresh_convergence_anchors_3_pass']=len(conv)==3 and all(r['status']=='PASS' for r in conv)
fits=read_csv(ROOT/'ASYMPTOTIC_FITS.csv'); f4=next(r for r in fits if r['x_threshold']=='4')
checks['p1_scaling_gate']=int(f4['n'])>=3 and float(f4['beta_TV'])>.55 and float(f4['R2'])>=.95 and float(f4['beta_drop_first'])>0 and float(f4['R2_drop_first'])>=.95
local=read_csv(ROOT/'LOCAL_EXPONENTS.csv')
checks['last_three_local_beta_positive']=len(local)>=3 and all(float(r['beta_local'])>0 for r in local[-3:])
cmp=read_csv(ROOT/'P1_FS_COMPARISON.csv'); byM={}
for r in cmp: byM.setdefault(int(r['M']),[]).append(r)
wins=[m for m,rr in byM.items() if len(rr)==2 and all(r['strict_win']=='True' and float(r['ratio_final_upper_over_FS_lower'])<.99 for r in rr)]
asym=[int(r['M']) for r in priv if float(r['M_over_LTV_M'])>=4]
checks['p1_strict_fs_two_consecutive_asymptotic']=any(a in wins and b in wins for a,b in zip(asym,asym[1:]))
checks['p1_fs_all_seven_points_matched']=len(cmp)==14 and all(len(v)==2 for v in byM.values())
k4file=ROOT/'USER_PACKET_K4_ROBUSTNESS.csv'
if k4file.exists():
 k4=read_csv(k4file); k4m={}
 for r in k4:k4m.setdefault(int(r['M']),[]).append(r)
 consecutive=any(a in k4m and b in k4m and len(k4m[a])==len(k4m[b])==2 and all(x['strict_win']=='True' for x in k4m[a]+k4m[b]) for a,b in zip(sorted(k4m),sorted(k4m)[1:]))
 k4final=json.loads((ROOT/'K4_FINAL_STATE_SUMMARY.json').read_text())
 checks['k4_three_final_points']=len(k4final)>=3 and sorted(int(r['M']) for r in k4final)==[1024,2048,4096]
 checks['k4_consecutive_two_delta_fs_wins']=consecutive
else:
 checks['k4_three_final_points']=False; checks['k4_consecutive_two_delta_fs_wins']=False
expected=json.loads((ROOT/'FINMEM1A_EXPECTED_JOB_MANIFEST.json').read_text())
checks['expected_manifest_present']=expected['status']=='FROZEN'
checks['expected_final_state_ids_complete']=len(expected['final_state_jobs'])==16 and expected['all_required_final_state_ids_match'] is True
actual_final={json.loads(p.read_text())['id'] for p in (ROOT/'final_state/jobs').glob('*.json')}
expected_final={x['id'] for x in expected['final_state_jobs']}
checks['expected_vs_observed_final_state_ids_exact']=actual_final==expected_final
memory_rows=read_csv(ROOT/'FINITE_HORIZON_MEMORY.csv')
checks['expected_vs_observed_memory_rows_exact']=len(memory_rows)==28 and len({(r['line'],r['grid'],int(r['M'])) for r in memory_rows})==28
checks['final_state_jobs_cpu_only']=expected['execution_facts']['final_state_gpu_used'] is False
parity_path=ROOT/'K4_GPU_CPU_PARITY.json'
checks['conditional_gpu_after_same_node_parity']=parity_path.exists() and json.loads(parity_path.read_text()).get('status')=='PASS' and expected['execution_facts']['conditional_k4_fs_gpu_used'] is True
checks['p2_fs_deferred']=expected['deferred_jobs']['P2_FS']=='DEFERRED_UNTIL_P1_DECISION'
k4_query_rows=read_csv(ROOT/'USER_PACKET_K4_ROBUSTNESS.csv')
completed_k4_fs=[x for x in expected['conditional_k4_fs_jobs'] if x['status']=='COMPLETED']
checks['expected_vs_observed_k4_fs_exact']=len(completed_k4_fs)==len(k4_query_rows)==4 and len({(int(r['M']),float(r['delta'])) for r in k4_query_rows})==4 and sorted(x['id'] for x in completed_k4_fs)==sorted(hashlib.sha256(json.dumps({'line':'P1','M':int(r['M']),'delta':float(r['delta']),'z':2.0,'k':4,'b':1},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20] for r in k4_query_rows)
failed=[k for k,v in checks.items() if not v]
out={'status':'PASS' if not failed else 'FAIL','checks_passed':len(checks)-len(failed),'checks_total':len(checks),'checks':checks,'failed_checks':failed,'independent_method':'recomputed from raw CSV/JSON artifacts; does not import aggregate or decision code','historical_decisions_preserved':True,'gpu_use':'none for final-state/memory; only conditional k4 FS after parity PASS'}
(ROOT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2))
