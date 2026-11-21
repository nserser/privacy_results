import json,glob,csv,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'input/MEMFRONT2_REVIEW/RAW_SPECTRAL'
EXT=ROOT/'memory/direct_extended'
OUT=ROOT/'memory';OUT.mkdir(exist_ok=True)
configs={}; original_byid={}
for f in glob.glob(str(RAW/'*.json')):
 d=json.load(open(f))
 if not isinstance(d,dict) or 'config' not in d: continue
 c=d['config'];configs[(c['line'],c['grid_label'])]=(d,Path(f[:-5]+'.npz'));original_byid[d['id']]=d
for f in glob.glob(str(EXT/'*.json')):
 d=json.load(open(f))
 if not isinstance(d,dict) or 'source_memfront2_id' not in d: continue
 orig=original_byid[d['source_memfront2_id']];c=orig['config']
 configs[(c['line'],c['grid_label'])]=(orig,Path(f[:-5]+'.npz'))
def get_series(line,grid):
 d,npzpath=configs[(line,grid)];a=np.load(npzpath);c=d['config']; y=a['tv_grid'] if 'tv_grid' in a else a['r']; up=a['tv_upper'];r0=len(y)-1
 return d,a,y,up,r0
MSET={'P1':[256,512,1024,2048,4096,8192,16384],'P2':[4096,8192,16384,32768,65536,131072,262144]}
GRIDS=['BASE','GRID','GRID2','DOMAIN']; rows=[]; continuum=[]
for line,Ms in MSET.items():
 for M in Ms:
  bygrid={}
  for grid in GRIDS:
   d,a,y,up,R=get_series(line,grid)
   if M-1<=R:
    vals=y[:M]; ups=up[:M]; split=M; label='DIRECT_FULL_GRID'
    e1=e2=0.
   else:
    modes=d['modes']; m=modes[1]; lam=m['abs_lambda']; amp=.5*m['density_l1_amplitude']
    rr=np.arange(R+1,M,dtype=np.int64); tail=amp*np.power(lam,rr)
    vals=np.concatenate([y[:R+1],tail]); ups0=up[:R+1]; ups=np.concatenate([ups0,tail]);split=R+1;label='HYBRID_MEMORY'
    e1=float(tail.sum());e2=float(np.dot(tail,tail))
   # LTV central grid response and conservative response upper; finite escape is separately quantified.
   S1=float(vals.sum());S2=float(vals@vals);L=S1*S1/S2
   S1u=float(ups.sum());S2u=float(ups@ups);Lu=S1u*S1u/S2u
   dom_def=float(np.maximum(ups-vals,0).sum())
   row={'line':line,'grid':grid,'M':M,'S1':S1,'S2':S2,'L_TV_M':L,'M_over_LTV_M':M/L,'S1_spectral_fraction':e1/S1 if S1 else 0.,'S2_spectral_fraction':e2/S2 if S2 else 0.,'direct_spectral_split_r':split,'finite_domain_defect_sum':dom_def,'finite_domain_defect_relative':dom_def/max(S1,1e-300),'L_TV_upper_response':Lu,'L_TV_upper_relative_difference':abs(Lu-L)/L,'evidence_label':label,'status':'DIRECT_MEMORY' if label=='DIRECT_FULL_GRID' else 'HYBRID_MEMORY'}
   rows.append(row);bygrid[grid]=row
  # Richardson from BASE(h), GRID(h/2), GRID2(h/4); never called certified if irregular.
  b,g,g2,dom=(bygrid[x] for x in GRIDS)
  d1=b['L_TV_M']-g['L_TV_M'];d2=g['L_TV_M']-g2['L_TV_M'];p=math.log2(abs(d1/d2)) if d1*d2>0 and d2!=0 else float('nan')
  rich=g2['L_TV_M']+(g2['L_TV_M']-g['L_TV_M'])/(2**p-1) if math.isfinite(p) and abs(2**p-1)>1e-12 else float('nan')
  domain_rel=abs(dom['L_TV_M']-b['L_TV_M'])/b['L_TV_M']
  fine_rel=abs(g2['L_TV_M']-rich)/rich if math.isfinite(rich) and rich else abs(g2['L_TV_M']-g['L_TV_M'])/g['L_TV_M']
  fd=max(x['finite_domain_defect_relative'] for x in (b,g,g2,dom))
  passed=domain_rel<=.01 and fine_rel<=.02 and fd<=.005
  hybrid=any(x['evidence_label']=='HYBRID_MEMORY' for x in (b,g,g2,dom))
  label=('HYBRID_MEMORY+' if hybrid else '')+('RICHARDSON_CONTINUUM_ESTIMATE' if math.isfinite(rich) else 'DIRECT_FULL_GRID_SEQUENCE')
  continuum.append({'line':line,'M':M,'L_BASE':b['L_TV_M'],'L_GRID':g['L_TV_M'],'L_GRID2':g2['L_TV_M'],'L_DOMAIN':dom['L_TV_M'],'p_order':p,'richardson_continuum':rich,'richardson_valid':math.isfinite(rich),'domain_vs_base_relative':domain_rel,'grid2_vs_continuum_relative':fine_rel,'grid_vs_grid2_relative':abs(g['L_TV_M']-g2['L_TV_M'])/g['L_TV_M'],'finite_domain_defect_relative_max':fd,'continuum_pass':passed,'evidence_label':label,'status':'PASS' if passed else 'FAIL_CONVERGENCE_GATE'})
fields=list(rows[0]);
with (OUT/'FINITE_HORIZON_MEMORY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (OUT/'FINITE_HORIZON_MEMORY_CONTINUUM.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(continuum[0]));w.writeheader();w.writerows(continuum)
(OUT/'FINITE_MEMORY_SUMMARY.json').write_text(json.dumps({'jobs':len(rows),'continuum_points':len(continuum),'results':continuum},indent=2,sort_keys=True)+'\n')
print(json.dumps({'rows':len(rows),'continuum':continuum},indent=2))
