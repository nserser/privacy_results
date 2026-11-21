from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1]; JOBS=ROOT/'final_state/jobs';MEM=ROOT/'memory/FINITE_HORIZON_MEMORY_CONTINUUM.csv';FS=ROOT/'final_state/fs'

def readcsv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
 memory={(r['line'],int(r['M'])):r for r in readcsv(MEM)}
 rows=[]
 for p in JOBS.glob('*.json'):
  d=json.loads(p.read_text())
  key=(d['line'],int(d['M']))
  if d['grid']!='BASE' or key not in memory:continue
  mr=memory[key];L=float(mr['richardson_continuum']);tv=float(d['tv_symmetric_grid']);upper=float(d['tv_symmetric_upper_mass_defect'])
  rows.append({'line':d['line'],'grid':d['grid'],'M':int(d['M']),'k':2,'L_TV_M':L,'M_over_LTV_M':int(d['M'])/L,'TV_final':tv,'TV_final_upper':upper,'epsilon_delta1e-3_upper':d['epsilon']['0.001']['epsilon_final_upper_with_mass_defect'],'epsilon_delta1e-5_upper':d['epsilon']['1e-05']['epsilon_final_upper_with_mass_defect'],'mass_defect_present':d['present_total_mass_defect'],'mass_defect_absent':d['absent_total_mass_defect'],'evidence_label':'DIRECT_FULL_GRID','status':d['status']})
 rows.sort(key=lambda r:(r['line'],r['M']))
 fields=list(rows[0]) if rows else ['line','grid','M']
 with (ROOT/'FINAL_PRIVACY_BY_M.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 local=[];fits=[];cross=[]
 for line in ('P1','P2'):
  rr=[r for r in rows if r['line']==line]
  for a,b in zip(rr,rr[1:]):
   beta=-math.log(b['TV_final']/a['TV_final'])/math.log(b['M']/a['M'])
   local.append({'line':line,'M_left':a['M'],'M_right':b['M'],'x_left':a['M_over_LTV_M'],'x_right':b['M_over_LTV_M'],'beta_local':beta,'evidence_label':'DIRECT_FULL_GRID'})
  x=np.array([r['M_over_LTV_M'] for r in rr]);m=np.array([r['M'] for r in rr]);tv=np.array([r['TV_final'] for r in rr])
  for threshold in (4,8):
   mask=x>=threshold;n=int(mask.sum());beta=r2=float('nan');lo=hi=float('nan');window='';bdrop=r2drop=float('nan')
   if n>=3:
    xx=np.log(m[mask]); yy=np.log(tv[mask]); slope,intercept=np.polyfit(xx,yy,1);fit=intercept+slope*xx;ssr=float(np.sum((yy-fit)**2));sst=float(np.sum((yy-yy.mean())**2));r2=1-ssr/sst if sst else 1.
    se=math.sqrt(ssr/(n-2)/float(np.sum((xx-xx.mean())**2)));crit=float(student_t.ppf(.975,n-2));beta=-slope;lo=beta-crit*se;hi=beta+crit*se;window=f"{int(m[mask][0])}-{int(m[mask][-1])}"
    if n>=4:
     xd=xx[1:];yd=yy[1:];sd,idc=np.polyfit(xd,yd,1);ydfit=idc+sd*xd;bdrop=-sd;r2drop=1-float(np.sum((yd-ydfit)**2))/float(np.sum((yd-yd.mean())**2))
   fits.append({'line':line,'x_threshold':threshold,'beta_TV':beta,'beta_CI95_low':lo,'beta_CI95_high':hi,'R2':r2,'n':n,'fit_window_M':window,'beta_drop_first':bdrop,'R2_drop_first':r2drop,'evidence_label':'DIRECT_FULL_GRID','status':'PASS' if n>=3 else 'INSUFFICIENT_POINTS'})
  if rr:
   peak=max(rr,key=lambda r:r['TV_final']); decay=None
   for i in range(max(0,len(rr)-3)):
    # Three consecutive positive local exponents means four observed points.
    vals=[x['beta_local'] for x in local if x['line']==line and x['M_left']>=rr[i]['M']]
    if len(vals)>=3 and all(v>0 for v in vals[:3]):decay=rr[i];break
   cross.append({'line':line,'M_peak':peak['M'],'x_peak':peak['M_over_LTV_M'],'M_decay':decay['M'] if decay else None,'x_decay':decay['M_over_LTV_M'] if decay else None,'max_M':rr[-1]['M'],'max_x':rr[-1]['M_over_LTV_M']})
 def write_csv(name,data):
  if not data:return
  with (ROOT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 write_csv('LOCAL_EXPONENTS.csv',local);write_csv('ASYMPTOTIC_FITS.csv',fits);write_csv('CROSSOVER_SUMMARY.csv',cross)
 fsrows=[]
 allgrid={(r['line'],int(r['M'])):r for r in readcsv(ROOT/'FINAL_PRIVACY_NUMERICAL_BOUNDS.csv')}
 for p in FS.glob('*.json'):
  if p.name in ('SUMMARY.json','EXPECTED_JOBS.json'):continue
  d=json.loads(p.read_text()); key=(d['line'],int(d['M']))
  if int(d.get('k',2))==2: fin=next((r for r in rows if (r['line'],r['M'])==key),None)
  else:
   import hashlib
   kid=hashlib.sha256(json.dumps({'line':'P1','grid':'BASE','M':int(d['M']),'k':int(d['k']),'b':1},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
   kp=ROOT/'final_state/k_robustness'/f"{kid}.json"
   fin=json.loads(kp.read_text()) if kp.exists() else None
  if fin:
   final_eps=(fin['epsilon_delta1e-3_upper'] if d['delta']==1e-3 else fin['epsilon_delta1e-5_upper']) if int(d.get('k',2))==2 else fin['epsilon_upper']['0.001' if d['delta']==1e-3 else '1e-05']
   if int(d.get('k',2))==2:
    br=allgrid.get(key,{})
    certified=br.get('epsilon_1e-3_upper_all_grids_max' if d['delta']==1e-3 else 'epsilon_1e-5_upper_all_grids_max','')
    if certified not in ('',None):final_eps=float(certified)
   fsrows.append({'line':d['line'],'M':d['M'],'k':d.get('k',2),'delta':d['delta'],'epsilon_final_upper':final_eps,'epsilon_FS_lower':d['epsilon_lower'],'epsilon_FS_upper':d['epsilon_upper'],'strict_win':bool(final_eps<.99*d['epsilon_lower']),'evidence_label':'EXTERNAL_ACCOUNTANT'})
 fsrows.sort(key=lambda r:(r['line'],r['M'],r['delta']));write_csv('FS_COMPARISON.csv',fsrows)
 expected=json.loads((ROOT/'final_state/EXPECTED_JOBS.json').read_text());done={json.loads(p.read_text())['id'] for p in JOBS.glob('*.json')};
 (ROOT/'FINAL_AGGREGATION_STATUS.json').write_text(json.dumps({'final_state_expected_jobs':len(expected),'final_state_completed_unique':len(done),'final_state_complete':len(expected)==len(done),'fs_queries_expected':40,'fs_queries_completed':len([p for p in FS.glob('*.json') if p.name not in ('SUMMARY.json','EXPECTED_JOBS.json')]),'fits':fits,'crossovers':cross},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'final_jobs':len(done),'expected':len(expected),'fs':len(fsrows),'fits':fits},indent=2))

if __name__=='__main__':main()
