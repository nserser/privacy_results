"""Independent, output-only audit; does not import the MEMFRONT2 decision builder."""
import csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[2]
checks=[]
def add(name,ok,detail): checks.append({'check':name,'pass':bool(ok),'detail':detail})
dec=json.loads((P/'MEMFRONT2_DECISION.json').read_text())
m0=json.loads((P/'M0_VERIFICATION.json').read_text())
with (P/'MEMORY_SUMMARY_FINAL.csv').open() as f: rows=list(csv.DictReader(f))
lookup={(r['line'],r['grid']):r for r in rows}
for line in ['P1','P2']:
 req=[lookup[(line,g)] for g in ['BASE','GRID','DOMAIN']]
 ltv=[float(x['LTV_completed_estimate']) for x in req]
 tau=[float(x['tau_slow']) for x in req]
 lspread=(max(ltv)-min(ltv))/float(lookup[(line,'BASE')]['LTV_completed_estimate'])
 tspread=(max(tau)-min(tau))/float(lookup[(line,'BASE')]['tau_slow'])
 fail=(lspread>.05 if line=='P1' else tspread>.05)
 add(line+'_closure_gate_recomputed_fail',fail,f'LTV spread={lspread:.8g}; slow-tau spread={tspread:.8g}')
 add(line+'_heldout_reconstruction',all(float(x['heldout_max_relative_error'])<=.05 for x in req), 'all three required grids <=5%')
 reqj=dec['memory_closure'][line]['required_grids']
 add(line+'_slow_mode_residual',all(float(x['max_retained_eigenpair_residual'])>=float(x['slow_mode_eigenpair_residual']) for x in reqj.values()),'decision records full and slow-mode residuals separately')
add('M0_pass',m0.get('status')=='PASS','frozen checks and worker-equivalence verified')
add('GPU_not_used',m0.get('gpu_used') is False,'CPU-only execution')
add('historical_statuses_unchanged',dec['historical_decisions_preserved']=={'SCALECERT':'STOP_REALISTIC_DP_SGD','MEMFRONT1':'REPAIR_MEMFRONT'},'historical statuses preserved')
add('downstream_not_run',dec.get('final_state_scaling')=='NOT_RUN_BLOCKED_BY_SPECTRAL_CLOSURE' and dec.get('feldman_shenfeld')=='NOT_RUN','fail-closed stages remain unexecuted')
add('terminal_decision_correct',dec.get('decision')=='REPAIR_MEMFRONT2','closure failure maps to REPAIR')
with (P/'MEMFRONT2_EXPECTED_JOB_MANIFEST.json').open() as f: manifest=json.load(f)
completed=[x for x in manifest['jobs'] if x['status']=='COMPLETED']
blocked=[x for x in manifest['jobs'] if x['status']=='BLOCKED_BY_SPECTRAL_CLOSURE']
add('job_manifest_consistent',len(completed)==13 and len(blocked)==5,'13 spectral jobs completed; five downstream stages blocked')
# checksum the complete review contents except the checksum manifest itself
hash_path=P/'SHA256SUMS.txt'
entries=[]
for line in hash_path.read_text().splitlines():
    h,rel=line.split('  ',1); f=P/rel
    if rel in ('SHA256SUMS.txt','INDEPENDENT_AUDIT.json'): continue
    entries.append(f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==h)
add('package_hashes_current',all(entries) and bool(entries),'all listed files match SHA-256 manifest')
out={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL_INDEPENDENT_AUDIT','checks_passed':sum(c['pass'] for c in checks),'checks_total':len(checks),'checks':checks,'scientific_outcome':'REPAIR_MEMFRONT2; no final-state result authorized'}
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2))
