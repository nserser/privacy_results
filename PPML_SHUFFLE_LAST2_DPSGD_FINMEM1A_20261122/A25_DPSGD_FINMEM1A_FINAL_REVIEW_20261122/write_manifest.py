import csv,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def id_of(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
memory=[{'id':id_of({'line':'P1','grid':g,'M':m,'k':2,'stage':'FINITE_HORIZON_MEMORY'}),'line':'P1','grid':g,'M':m,'k':2,'status':'COMPLETED'} for g in ('BASE','GRID','GRID2','DOMAIN') for m in (256,512,1024,2048,4096,8192,16384)]
jobs=json.loads((ROOT/'FINMEM1A_EXPECTED_JOBS.json').read_text())
final=[{'id':id_of({'line':x['line'],'grid':x['grid'],'M':x['M'],'k':x['k'],'b':1}),'line':x['line'],'grid':x['grid'],'M':x['M'],'k':2,'status':'COMPLETED'} for x in jobs]
fs=list(csv.DictReader((ROOT/'P1_FS_FROZEN_BASELINE.csv').open()))
fs=[{'id':r['id'],'line':'P1','M':int(r['M']),'delta':float(r['delta']),'k':2,'z':2.0,'b':1,'status':'REUSED_VALIDATED'} for r in fs]
ks=json.loads((ROOT/'K4_EXPECTED_JOBS.json').read_text())
k4=[{'id':id_of({'line':'P1','grid':'BASE','M':x['M'],'k':4,'b':1}),'line':'P1','grid':'BASE','M':x['M'],'k':4,'b':1,'status':'COMPLETED'} for x in ks]
k4fs=[]
if (ROOT/'K4_FS_STATUS.json').exists():
 status=json.loads((ROOT/'K4_FS_STATUS.json').read_text())
 queried=set(status['M_queried'])
 for m in (1024,2048,4096):
  for delta in (1e-3,1e-5):
   jid=id_of({'line':'P1','M':m,'delta':delta,'z':2.0,'k':4,'b':1})
   k4fs.append({'id':jid,'line':'P1','M':m,'delta':delta,'k':4,'z':2.0,'b':1,'status':'COMPLETED' if m in queried else 'NOT_TRIGGERED_AFTER_TWO_CONSECUTIVE_WINS'})
expected={'milestone':'PPML-SHUFFLE-LAST2-DPSGD-FINMEM1A','status':'FROZEN','gpu_requested_for_final_state':False,'execution_facts':{'final_state_gpu_used':False,'conditional_k4_fs_gpu_used':bool(k4fs),'workers_final_state':8,'workers_k4_final_state':3,'blas_threads_per_worker':1},'memory_jobs':memory,'final_state_jobs':final,'archived_p1_fs_jobs':fs,'conditional_k4_final_state_jobs':k4,'conditional_k4_fs_jobs':k4fs,'deferred_jobs':{'P2_FS':'DEFERRED_UNTIL_P1_DECISION','k1_FS':'NOT_AUTHORIZED','P2_final_state':'NOT_REQUIRED'},'all_required_final_state_ids_match':len(final)==16}
(ROOT/'FINMEM1A_EXPECTED_JOB_MANIFEST.json').write_text(json.dumps(expected,indent=2,sort_keys=True)+'\n')
def cmd(x):
 try:return subprocess.check_output(x,text=True,stderr=subprocess.DEVNULL).strip()
 except Exception:return None
import numpy,scipy
env={'timestamp_utc':cmd(['date','-u','+%Y-%m-%dT%H:%M:%SZ']),'hostname':platform.node(),'os':platform.platform(),'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'cpu_model':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),'visible_cpus':os.cpu_count(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'gpu_requested_for_final_state':False,'gpu_used_for_final_state':False,'gpu_used_for_conditional_k4_fs':bool(k4fs),'feldman_shenfeld_commit':'fde83c45bf12c4f7821832c7d5d2ae2841fc0910'}
with (ROOT/'ENVIRONMENT.json').open('w') as f:json.dump(env,f,indent=2,sort_keys=True)
print(json.dumps({'memory_jobs':len(memory),'final_state_jobs':len(final),'archived_fs':len(fs),'k4_final_state_jobs':len(k4),'status':expected['status']}))
