import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows={}
for p in (ROOT/'final_state/jobs').glob('*.json'):
 d=json.loads(p.read_text())
 if d.get('line')=='P1' and d.get('k')==2: rows[(d['grid'],int(d['M']))]=d
out=[]
for M in (1024,4096,16384):
 vals={g:rows[(g,M)] for g in ('BASE','GRID','GRID2','DOMAIN')}
 tv={g:float(d['tv_symmetric_upper_mass_defect']) for g,d in vals.items()}
 eps={delta:max(float(d['epsilon'][delta]['epsilon_final_upper_with_mass_defect']) for d in vals.values()) for delta in ('0.001','1e-05')}
 defects=[float(d['present_total_mass_defect'])+float(d['absent_total_mass_defect']) for d in vals.values()]
 domain=abs(tv['DOMAIN']-tv['BASE'])/tv['BASE']; fine=abs(tv['GRID2']-tv['GRID'])/tv['GRID']
 out.append({'line':'P1','M':M,'tv_base':tv['BASE'],'tv_grid':tv['GRID'],'tv_grid2':tv['GRID2'],'tv_domain':tv['DOMAIN'],'domain_vs_base_relative':domain,'grid_vs_grid2_relative':fine,'epsilon_delta1e-3_max_all_grids':eps['0.001'],'epsilon_delta1e-3_relative_range':(max(float(d['epsilon']['0.001']['epsilon_final_upper_with_mass_defect']) for d in vals.values())-min(float(d['epsilon']['0.001']['epsilon_final_upper_with_mass_defect']) for d in vals.values()))/max(eps['0.001'],1e-300),'epsilon_delta1e-5_max_all_grids':eps['1e-05'],'epsilon_delta1e-5_relative_range':(max(float(d['epsilon']['1e-05']['epsilon_final_upper_with_mass_defect']) for d in vals.values())-min(float(d['epsilon']['1e-05']['epsilon_final_upper_with_mass_defect']) for d in vals.values()))/max(eps['1e-05'],1e-300),'finite_domain_defect_max':max(defects),'status':'PASS' if domain<=.02 and fine<=.02 else 'FAIL'})
with (ROOT/'NUMERICAL_CONVERGENCE_FINAL.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(out,indent=2))
