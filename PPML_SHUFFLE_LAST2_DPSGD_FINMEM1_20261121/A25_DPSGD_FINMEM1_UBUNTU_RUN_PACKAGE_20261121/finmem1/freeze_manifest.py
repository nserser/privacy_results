import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':')).encode()
def ident(o):return hashlib.sha256(canon(o)).hexdigest()[:20]
mem=[]
for line,Ms in [('P1',[256,512,1024,2048,4096,8192,16384]),('P2',[4096,8192,16384,32768,65536,131072,262144])]:
 for M in Ms:
  for grid in ('BASE','GRID','GRID2','DOMAIN'):mem.append({'id':ident({'line':line,'grid':grid,'M':M,'stage':'FINITE_MEMORY'}),'stage':'FINITE_MEMORY','line':line,'grid':grid,'M':M})
final=[]
for q in json.loads((R/'final_state/EXPECTED_JOBS.json').read_text()):
 x={'line':q['line'],'grid':q['grid'],'M':q['M'],'k':q['k'],'b':q['b']};final.append({'id':ident(x),'stage':'FINAL_STATE',**x})
fs=[]
for line,z,Ms in [('P1',2.0,[256,512,1024,2048,4096,8192,16384]),('P2',4.0,[4096,8192,16384,32768,65536,131072,262144])]:
 for M in Ms:
  for d in (1e-3,1e-5):
   x={'line':line,'M':M,'delta':d,'z':z,'k':2,'b':1};fs.append({'id':ident(x),'stage':'FS','evidence_label':'EXTERNAL_ACCOUNTANT',**x})
for k in (1,4):
 for M in (1024,2048,4096):
  for d in (1e-3,1e-5):
   x={'line':'P1','M':M,'delta':d,'z':2.0,'k':k,'b':1};fs.append({'id':ident(x),'stage':'FS','evidence_label':'EXTERNAL_ACCOUNTANT',**x})
krob=[{'id':ident({'line':'P1','grid':'BASE','M':M,'k':k,'b':1}),'stage':'K_ROBUSTNESS','line':'P1','grid':'BASE','M':M,'k':k,'b':1} for k in (1,4) for M in (1024,2048,4096)]
ini=[{'id':ident({'line':'P1','grid':'BASE','M':M,'initialization':q,'k':2}),'stage':'INITIALIZATION','line':'P1','grid':'BASE','M':M,'initialization':q,'k':2} for M in (1024,4096,16384) for q in ('continuous_stationary','well_plus','well_minus')]
out={'milestone':'PPML-SHUFFLE-LAST2-DPSGD-FINMEM1','frozen_spec_date':'2026-11-21','gpu_requested':False,'job_id_rule':'first 20 hex chars of SHA256(canonical sorted JSON configuration)','expected_counts':{'finite_memory':len(mem),'final_state':len(final),'feldman_shenfeld':len(fs),'k_robustness':len(krob),'initialization':len(ini)},'jobs':mem+final+fs+krob+ini}
(R/'FINMEM1_EXPECTED_JOB_MANIFEST.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out['expected_counts'],sort_keys=True))
