import csv,json,glob,math
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
 if rows:
  with (R/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
states=[json.load(open(p)) for p in glob.glob(str(R/'final_state/jobs/*.json'))]
base={(x['line'],int(x['M'])):x for x in states if x['grid']=='BASE'}
conv=[];bounds=[]
for line,Ms in [('P1',(1024,4096,16384)),('P2',(8192,32768,65536))]:
 for M in Ms:
  d={x['grid']:x for x in states if x['line']==line and int(x['M'])==M}
  if len(d)!=4:conv.append({'line':line,'M':M,'status':'INCOMPLETE','available_grids':','.join(d)});continue
  tv={g:float(x['tv_symmetric_upper_mass_defect']) for g,x in d.items()};b=tv['BASE'];dom=abs(tv['DOMAIN']/b-1);fine=abs(tv['GRID']/tv['GRID2']-1)
  epsrel={};epsmax={}
  for delta,key in [('1e-3','0.001'),('1e-5','1e-05')]:
   vals={g:float(x['epsilon'][key]['epsilon_final_upper_with_mass_defect']) for g,x in d.items()};vmax=max(vals.values());vmin=min(vals.values());epsrel[delta]=abs(vmax-vmin)/max(vmax,1e-300);epsmax[delta]=vmax
  passv=dom<=.02 and fine<=.02
  conv.append({'line':line,'M':M,'tv_base':tv['BASE'],'tv_grid':tv['GRID'],'tv_grid2':tv['GRID2'],'tv_domain':tv['DOMAIN'],'domain_vs_base_relative':dom,'grid_vs_grid2_relative':fine,'epsilon_delta1e-3_max_all_grids':epsmax['1e-3'],'epsilon_delta1e-3_relative_range':epsrel['1e-3'],'epsilon_delta1e-5_max_all_grids':epsmax['1e-5'],'epsilon_delta1e-5_relative_range':epsrel['1e-5'],'finite_domain_defect_max':max(float(x['present_total_mass_defect'])+float(x['absent_total_mass_defect']) for x in d.values()),'status':'PASS' if passv else 'FAIL'})
for (line,M),x in base.items():
 d=[q for q in states if q['line']==line and int(q['M'])==M]
 row={'line':line,'M':M,'TV_base_grid':x['tv_symmetric_grid'],'TV_conservative_upper_base':x['tv_symmetric_upper_mass_defect'],'epsilon_delta1e-3_base_upper':x['epsilon']['0.001']['epsilon_final_upper_with_mass_defect'],'epsilon_delta1e-5_base_upper':x['epsilon']['1e-05']['epsilon_final_upper_with_mass_defect'],'evidence_label':'DIRECT_FULL_GRID'}
 if len(d)==4:
  row['TV_upper_all_grids_max']=max(float(q['tv_symmetric_upper_mass_defect']) for q in d)
  for delta,key in [('1e-3','0.001'),('1e-5','1e-05')]:row[f'epsilon_{delta}_upper_all_grids_max']=max(float(q['epsilon'][key]['epsilon_final_upper_with_mass_defect']) for q in d)
 else:
  row['TV_upper_all_grids_max']=row['epsilon_1e-3_upper_all_grids_max']=row['epsilon_1e-5_upper_all_grids_max']=''
 bounds.append(row)
write('NUMERICAL_CONVERGENCE_FINAL.csv',conv);write('FINAL_PRIVACY_NUMERICAL_BOUNDS.csv',bounds)

mem={(x['line'],int(x['M'])):x for x in read(R/'memory/FINITE_HORIZON_MEMORY_CONTINUUM.csv')}
krows=[]
for p in glob.glob(str(R/'final_state/k_robustness/*.json')):
 if p.endswith(('SUMMARY.json','EXPECTED_JOBS.json')):continue
 x=json.load(open(p));M=int(x['M']);L=float(mem[('P1',M)]['richardson_continuum']);krows.append({**x,'L_TV_M':L,'x_M':M/L,'epsilon_1e-3_upper':x['epsilon_upper']['0.001'],'epsilon_1e-5_upper':x['epsilon_upper']['1e-05']})
write('USER_PACKET_ROBUSTNESS.csv',krows)

inits=[]
for p in glob.glob(str(R/'final_state/initialization/*.json')):
 if p.endswith(('SUMMARY.json','EXPECTED_JOBS.json')):continue
 x=json.load(open(p));M=int(x['M']);b=base[('P1',M)]
 inits.append({'M':M,'initialization':x['initialization'],'tv':x['tv_grid'],'tv_upper':x['tv_upper'],'invariant_tv':b['tv_symmetric_grid'],'absolute_TV_difference':abs(float(x['tv_grid'])-float(b['tv_symmetric_grid'])),'relative_TV_difference':abs(float(x['tv_grid'])/float(b['tv_symmetric_grid'])-1),'epsilon_delta1e-3_upper':x['epsilon_upper']['0.001'],'epsilon_delta1e-5_upper':x['epsilon_upper']['1e-05'],'evidence_label':'DIRECT_FULL_GRID'})
write('INITIALIZATION_SENSITIVITY.csv',inits)

infinite=read(R/'input/MEMFRONT2_REVIEW/MEMORY_SPECTRAL_COMPLETION.csv');fvi=[]
for row in read(R/'memory/FINITE_HORIZON_MEMORY_CONTINUUM.csv'):
 inf=next((float(q['LTV_completed_point_estimate']) for q in infinite if q['line']==row['line'] and q['grid']=='BASE'),None)
 fvi.append({'line':row['line'],'M':row['M'],'L_TV_M_continuum':row['richardson_continuum'],'L_TV_infinite_diagnostic':inf,'finite_over_infinite':float(row['richardson_continuum'])/inf if inf else None,'evidence_label_finite':'RICHARDSON_CONTINUUM_ESTIMATE','evidence_label_infinite':'POSTHOC_DIAGNOSTIC'})
write('FINITE_VS_INFINITE_MEMORY.csv',fvi)
print(json.dumps({'final_state':len(states),'numerical_anchor_rows':len(conv),'k_rows':len(krows),'initialization_rows':len(inits),'memory_rows':len(fvi)},sort_keys=True))
