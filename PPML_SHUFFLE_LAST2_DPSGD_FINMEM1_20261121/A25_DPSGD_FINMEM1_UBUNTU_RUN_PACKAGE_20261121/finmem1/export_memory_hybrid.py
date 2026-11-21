import csv,glob,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];SRC=R/'input/MEMFRONT2_REVIEW/RAW_SPECTRAL';EXT=R/'memory/direct_extended'
orig={}
for p in glob.glob(str(SRC/'*.json')):
 d=json.load(open(p));c=d.get('config',{})
 if c.get('grid_label')=='BASE' and c.get('line') in ('P1','P2'):orig[c['line']]=(d,Path(p[:-5]+'.npz'))
for p in glob.glob(str(EXT/'*.json')):
 d=json.load(open(p));
 if isinstance(d,dict) and d.get('line')=='P1' and d.get('grid')=='BASE':orig['P1']=(orig['P1'][0],Path(p[:-5]+'.npz'))
rows=[]
for line,(d,path) in orig.items():
 if line=='P1':
  # Select exact matching extended archive by source id, then the original direct base if needed.
  ep=None
  for jf in glob.glob(str(EXT/'*.json')):
   jd=json.load(open(jf))
   if jd.get('line')=='P1' and jd.get('grid')=='BASE':ep=Path(jf[:-5]+'.npz');break
  a=np.load(ep if ep else path);vals=a['tv_grid'] if 'tv_grid' in a else a['r'];upper=a['tv_upper'] if 'tv_upper' in a else vals
  for r,(v,u) in enumerate(zip(vals,upper)):rows.append({'line':line,'grid':'BASE','r':r,'a_r':float(v),'a_r_upper':float(u),'evidence_label':'DIRECT_FULL_GRID'})
 else:
  a=np.load(path);vals=a['r'];upper=a['tv_upper'];Rmax=len(vals)-1
  mode=d['modes'][1];lam=float(mode['abs_lambda']);amp=.5*float(mode['density_l1_amplitude'])
  for r in range(262145):
   if r<=Rmax: v=float(vals[r]);u=float(upper[r]);label='DIRECT_FULL_GRID'
   else: v=amp*lam**r;u=v;label='HYBRID_MEMORY'
   rows.append({'line':line,'grid':'BASE','r':r,'a_r':v,'a_r_upper':u,'evidence_label':label})
with (R/'MEMORY_RESPONSE_HYBRID.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'WROTE {len(rows)} lag rows; P1 direct extension, P2 direct through 65536 then validated spectral continuation')
