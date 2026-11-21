import csv,json,glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];F=R/'figures';F.mkdir(exist_ok=True)
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def save(fig,name):fig.tight_layout();fig.savefig(F/f'{name}.png',dpi=170);plt.close(fig)

# Memory trajectories: parent direct finite responses plus the newly extended P1 series.
direct=read(R/'input/MEMFRONT2_REVIEW/MEMORY_RESPONSE_DIRECT.csv')
ext=[]
for p in (R/'memory/direct_extended').glob('*.npz'):
 j=p.with_suffix('.json')
 if not j.exists():continue
 d=json.loads(j.read_text());a=np.load(p);vals=a['tv_grid'] if 'tv_grid' in a else a['r']
 ext.extend((d['line'],d['grid'],i,float(v),'DIRECT_FULL_GRID') for i,v in enumerate(vals))
for line in ('P1','P2'):
 fig,ax=plt.subplots(figsize=(7,4.4))
 q=[r for r in direct if r['line']==line and r['grid']=='BASE']
 ax.loglog([int(r['r'])+1 for r in q],[float(r['TV_direct_grid'])/float(q[0]['TV_direct_grid']) for r in q],label='MEMFRONT2 direct, normalized')
 e=[r for r in ext if r[0]==line and r[1]=='BASE']
 if e:ax.loglog([r[2]+1 for r in e],[r[3]/e[0][3] for r in e],label='FINMEM1 direct extension, normalized')
 ax.axhline(1e-4,color='gray',ls=':');ax.set(xlabel='lag r + 1',ylabel='a_r / a_0',title=f'{line} one-shock response (direct evidence)');ax.legend();save(fig,f'memory_response_{line}')

mem=read(R/'memory/FINITE_HORIZON_MEMORY_CONTINUUM.csv')
final=read(R/'FINAL_PRIVACY_BY_M.csv')
for line in ('P1','P2'):
 rr=[r for r in mem if r['line']==line];fig,ax=plt.subplots(figsize=(7,4.4))
 ax.loglog([int(r['M']) for r in rr],[float(r['richardson_continuum']) for r in rr],'o-',label='finite-horizon Richardson estimate')
 inf=next((float(x['LTV_completed_point_estimate']) for x in read(R/'input/MEMFRONT2_REVIEW/MEMORY_SPECTRAL_COMPLETION.csv') if x['line']==line and x['grid']=='BASE'),None)
 if inf:ax.axhline(inf,color='tab:red',ls='--',label='formal infinite-horizon diagnostic')
 ax.set(xlabel='M',ylabel='L_TV(M)',title=f'{line} finite vs infinite-horizon memory');ax.legend();save(fig,f'LTV_M_vs_M_{line}')

fig,ax=plt.subplots(figsize=(7,4.4))
for line in ('P1','P2'):
 rr=[r for r in mem if r['line']==line];ax.semilogx([int(r['M']) for r in rr],[float(r['richardson_continuum']) for r in rr],'o-',label=line)
ax.set(xlabel='M',ylabel='L_TV(M)',title='Finite-horizon memory');ax.legend();save(fig,'LTV_M_vs_M')

fig,ax=plt.subplots(figsize=(7,4.4))
for line in ('P1','P2'):
 rr=[r for r in final if r['line']==line and r['grid']=='BASE'];ax.loglog([int(r['M']) for r in rr],[float(r['TV_final']) for r in rr],'o-',label=f'{line} DIRECT_FULL_GRID')
ax.set(xlabel='M',ylabel='final symmetric TV',title='Final privacy vs raw M');ax.legend();save(fig,'final_TV_vs_M')

fig,ax=plt.subplots(figsize=(7,4.4))
for line in ('P1','P2'):
 rr=[r for r in final if r['line']==line and r['grid']=='BASE'];ax.loglog([float(r['M_over_LTV_M']) for r in rr],[float(r['TV_final']) for r in rr],'o-',label=line)
ax.set(xlabel='x_M = M / L_TV(M)',ylabel='final symmetric TV',title='Final privacy vs finite-memory-normalized M');ax.legend();save(fig,'final_TV_vs_xM')

local=read(R/'LOCAL_EXPONENTS.csv')
fig,ax=plt.subplots(figsize=(7,4.4))
for line in ('P1','P2'):
 rr=[r for r in local if r['line']==line];ax.semilogx([(float(r['x_left'])*float(r['x_right']))**.5 for r in rr],[float(r['beta_local']) for r in rr],'o-',label=line)
ax.axhline(.55,color='gray',ls=':');ax.set(xlabel='geometric midpoint x_M',ylabel='local beta',title='Local final-TV decay exponent');ax.legend();save(fig,'local_beta_vs_xM')

fs=read(R/'FS_COMPARISON.csv') if (R/'FS_COMPARISON.csv').exists() else []
for delta,name in ((1e-3,'epsilon_vs_FS_delta1e-3'),(1e-5,'epsilon_vs_FS_delta1e-5')):
 fig,ax=plt.subplots(figsize=(7,4.4))
 for line in ('P1','P2'):
  rr=[r for r in fs if r['line']==line and int(r['k'])==2 and float(r['delta'])==delta];rr.sort(key=lambda r:int(r['M']))
  if rr:
   ax.loglog([int(r['M']) for r in rr],[float(r['epsilon_final_upper']) for r in rr],'o-',label=f'{line} final upper')
   ax.loglog([int(r['M']) for r in rr],[float(r['epsilon_FS_lower']) for r in rr],'s--',label=f'{line} FS lower')
 ax.set(xlabel='M',ylabel='epsilon',title=f'Final vs allocation baseline, delta={delta:g}');ax.legend(fontsize=8);save(fig,name)

krows=[]
for p in (R/'final_state/k_robustness').glob('*.json'):
 if p.name not in ('SUMMARY.json','EXPECTED_JOBS.json'):krows.append(json.loads(p.read_text()))
fig,ax=plt.subplots(figsize=(7,4.4))
base={(int(r['M']),2):float(r['TV_final']) for r in final if r['line']=='P1' and r['grid']=='BASE'}
for k in (1,2,4):
 if k==2:rr=[(m,v) for (m,kk),v in base.items() if m in (1024,2048,4096)]
 else:rr=[(int(x['M']),float(x['tv_grid'])) for x in krows if int(x['k'])==k]
 rr.sort();ax.loglog([x[0] for x in rr],[x[1] for x in rr],'o-',label=f'k={k}')
ax.set(xlabel='M',ylabel='final symmetric TV',title='User-packet robustness');ax.legend();save(fig,'k_robustness')

init=[]
for p in (R/'final_state/initialization').glob('*.json'):
 if p.name not in ('SUMMARY.json','EXPECTED_JOBS.json'):init.append(json.loads(p.read_text()))
fig,ax=plt.subplots(figsize=(7,4.4))
for tag in ('continuous_stationary','well_plus','well_minus'):
 rr=sorted([x for x in init if x['initialization']==tag],key=lambda x:int(x['M']))
 ax.loglog([int(x['M']) for x in rr],[float(x['tv_grid']) for x in rr],'o-',label=tag)
ax.set(xlabel='M',ylabel='final symmetric TV',title='P1 initialization sensitivity');ax.legend();save(fig,'initialization_sensitivity')

conv=[]
for line,Ms in (('P1',(1024,4096,16384)),('P2',(8192,32768,65536))):
 for M in Ms:
  ds={x['grid']:x for x in (json.load(open(p)) for p in glob.glob(str(R/'final_state/jobs/*.json'))) if x['line']==line and int(x['M'])==M}
  if len(ds)==4:
   g=float(ds['GRID']['tv_symmetric_upper_mass_defect']);g2=float(ds['GRID2']['tv_symmetric_upper_mass_defect']);basev=float(ds['BASE']['tv_symmetric_upper_mass_defect']);dom=float(ds['DOMAIN']['tv_symmetric_upper_mass_defect'])
   conv.append((line,M,abs(dom/basev-1),abs(g/g2-1)))
fig,ax=plt.subplots(figsize=(7,4.4))
for line in ('P1','P2'):
 rr=[x for x in conv if x[0]==line];ax.plot([x[1] for x in rr],[100*x[3] for x in rr],'o-',label=f'{line}: GRID vs GRID2 %')
ax.axhline(2,color='gray',ls=':');ax.set(xlabel='M anchor',ylabel='relative final-TV change (%)',title='Final-state grid convergence');ax.legend();save(fig,'numerical_convergence')

fig,ax=plt.subplots(figsize=(7,3.5));ax.axis('off');ax.text(.5,.58,'No reduced-order model used',ha='center',va='center',fontsize=16);ax.text(.5,.38,'All reported headline final-state points were computed by the direct hidden-support recurrence.',ha='center',va='center',wrap=True,fontsize=11);save(fig,'rom_validation_not_applicable')
print(f'FIGURES_WRITTEN {len(list(F.glob("*.png")))}')
