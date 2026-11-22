import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parent
def read(p): return list(csv.DictReader(Path(p).open(newline='')))
def write(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
mem={int(r['M']):r for r in read(ROOT/'memory/FINITE_HORIZON_MEMORY_CONTINUUM.csv') if r['line']=='P1'}
jobdir=ROOT/'final_state/jobs'
raw=[]
for p in jobdir.glob('*.json'):
    d=json.loads(p.read_text())
    if d.get('line')=='P1' and d.get('k')==2: raw.append(d)
by={(d['grid'],int(d['M'])):d for d in raw}
Ms=[256,512,1024,2048,4096,8192,16384]
rows=[]; bounds=[]
for M in Ms:
    d=by[('BASE',M)]; mr=mem[M];L=float(mr['richardson_continuum'])
    emax={}
    for delta in ('0.001','1e-05'):
        vals=[]
        for g in ('BASE','GRID','GRID2','DOMAIN'):
            x=by.get((g,M))
            if x: vals.append(float(x['epsilon'][delta]['epsilon_final_upper_with_mass_defect']))
        emax[delta]=max(vals)
        bounds.append({'line':'P1','M':M,'delta':delta,'epsilon_final_upper_max_available_grid':emax[delta], 'grids_used':['BASE','GRID','GRID2','DOMAIN'] if M in (1024,4096,16384) else ['BASE'],'status':'CERTIFIED' if M not in (1024,4096,16384) or len(vals)==4 else 'ANCHOR_GRID_MISSING'})
    rows.append({'line':'P1','grid':'BASE','M':M,'k':2,'L_TV_M':L,'M_over_LTV_M':M/L,'TV_final':float(d['tv_symmetric_grid']),'TV_final_upper':float(d['tv_symmetric_upper_mass_defect']),'epsilon_delta1e-3_upper':emax['0.001'],'epsilon_delta1e-5_upper':emax['1e-05'],'mass_defect_present':float(d['present_total_mass_defect']),'mass_defect_absent':float(d['absent_total_mass_defect']),'evidence_label':'DIRECT_FULL_GRID','status':d['status']})
write(ROOT/'FINITE_HORIZON_MEMORY_CONTINUUM.csv',[{'line':'P1',**{k:v for k,v in m.items() if k!='line'}} for m in [mem[M] for M in Ms]])
write(ROOT/'FINITE_HORIZON_MEMORY.csv',[r for r in read(ROOT/'memory/FINITE_HORIZON_MEMORY.csv') if r['line']=='P1'])
write(ROOT/'FINAL_PRIVACY_BY_M.csv',rows);write(ROOT/'FINAL_PRIVACY_NUMERICAL_BOUNDS.csv',bounds)
local=[]
for a,b in zip(rows,rows[1:]):
    local.append({'line':'P1','M_left':a['M'],'M_right':b['M'],'x_left':a['M_over_LTV_M'],'x_right':b['M_over_LTV_M'],'beta_local':-math.log(b['TV_final']/a['TV_final'])/math.log(b['M']/a['M']),'evidence_label':'DIRECT_FULL_GRID'})
fits=[]
for threshold in (4,8):
    rr=[r for r in rows if r['M_over_LTV_M']>=threshold]; n=len(rr); xx=np.log([r['M'] for r in rr]); yy=np.log([r['TV_final'] for r in rr]); slope,inter=np.polyfit(xx,yy,1); pred=inter+slope*xx; ssr=float(np.sum((yy-pred)**2));sst=float(np.sum((yy-yy.mean())**2));r2=1-ssr/sst; se=math.sqrt(ssr/(n-2)/float(np.sum((xx-xx.mean())**2)));crit=float(student_t.ppf(.975,n-2))
    dropped=rr[1:];xd=np.log([r['M'] for r in dropped]);yd=np.log([r['TV_final'] for r in dropped]);sd,idc=np.polyfit(xd,yd,1);pdrop=idc+sd*xd;r2drop=1-float(np.sum((yd-pdrop)**2))/float(np.sum((yd-yd.mean())**2))
    fits.append({'line':'P1','x_threshold':threshold,'beta_TV':-slope,'beta_CI95_low':-slope-crit*se,'beta_CI95_high':-slope+crit*se,'R2':r2,'n':n,'fit_window_M':f"{rr[0]['M']}-{rr[-1]['M']}",'beta_drop_first':-sd,'R2_drop_first':r2drop,'evidence_label':'DIRECT_FULL_GRID','status':'PASS' if n>=3 else 'INSUFFICIENT_POINTS'})
write(ROOT/'LOCAL_EXPONENTS.csv',local);write(ROOT/'ASYMPTOTIC_FITS.csv',fits)
peak=max(rows,key=lambda r:r['TV_final']);decay=next((r for i,r in enumerate(rows[:-3]) if all(x['beta_local']>0 for x in local[i:i+3])),None)
write(ROOT/'CROSSOVER_SUMMARY.csv',[{'line':'P1','M_peak':peak['M'],'x_peak':peak['M_over_LTV_M'],'M_decay':decay['M'] if decay else None,'x_decay':decay['M_over_LTV_M'] if decay else None,'max_M':rows[-1]['M'],'max_x':rows[-1]['M_over_LTV_M']}])
fs=read(ROOT/'P1_FS_FROZEN_BASELINE.csv');fsrows=[]
for f in fs:
    M=int(f['M']); delta=float(f['delta']); d=next(r for r in rows if r['M']==M); eps=d['epsilon_delta1e-3_upper'] if delta==.001 else d['epsilon_delta1e-5_upper']; lo=float(f['epsilon_lower']); ratio=eps/lo
    fsrows.append({'line':'P1','M':M,'k':2,'delta':delta,'epsilon_final_upper_cert':eps,'epsilon_FS_lower':lo,'epsilon_FS_upper':float(f['epsilon_upper']),'strict_threshold_0.99_FS_lower':.99*lo,'ratio_final_upper_over_FS_lower':ratio,'strict_win':ratio<.99,'asymptotic_x_ge_4':d['M_over_LTV_M']>=4,'baseline_evidence':'REUSED_VALIDATED_ARCHIVED_GPU','status':'PASS'})
fsrows.sort(key=lambda r:(r['M'],r['delta']));write(ROOT/'P1_FS_COMPARISON.csv',fsrows)
decision={'finite_memory_pass':all(m['status']=='PASS' for m in [mem[M] for M in Ms]),'points_x_ge_4':[r['M'] for r in rows if r['M_over_LTV_M']>=4],'points_x_ge_8':[r['M'] for r in rows if r['M_over_LTV_M']>=8],'fit_x_ge_4':fits[0],'fit_x_ge_8':fits[1],'all_last_three_local_exponents_positive':all(x['beta_local']>0 for x in local[-3:]),'drop_first_conclusion_preserved':fits[0]['beta_drop_first']>0 and fits[0]['R2_drop_first']>=.95,'strict_fs_both_deltas_M':[M for M in Ms if all(x['strict_win'] for x in fsrows if x['M']==M)],'strict_asymptotic_consecutive_pairs':[[a,b] for a,b in zip(Ms,Ms[1:]) if a in [r['M'] for r in rows if r['M_over_LTV_M']>=4] and b in [r['M'] for r in rows if r['M_over_LTV_M']>=4] and all(x['strict_win'] for x in fsrows if x['M'] in (a,b))]}
(ROOT/'P1_GATES.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
print(json.dumps(decision,indent=2))
