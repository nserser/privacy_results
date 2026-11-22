import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
def rows(f):return list(csv.DictReader((ROOT/f).open()))
mem=rows('FINITE_HORIZON_MEMORY_CONTINUUM.csv'); final=rows('FINAL_PRIVACY_BY_M.csv'); comp=rows('P1_FS_COMPARISON.csv'); loc=rows('LOCAL_EXPONENTS.csv'); k4p=ROOT/'USER_PACKET_K4_ROBUSTNESS.csv'
M=[int(r['M']) for r in final];x=[float(r['M_over_LTV_M']) for r in final];tv=[float(r['TV_final']) for r in final]
def save(fig,name):fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=180);fig.savefig(OUT/f'{name}.svg');plt.close(fig)
f,ax=plt.subplots();ax.plot([int(r['M']) for r in mem],[float(r['richardson_continuum']) for r in mem],marker='o');ax.set(xscale='log',xlabel='M',ylabel=r'$L_{TV}(M)$',title='P1 finite-horizon privacy memory');save(f,'P1_LTV_vs_M')
f,ax=plt.subplots();ax.loglog(M,tv,marker='o');ax.set(xlabel='M',ylabel='Final symmetric TV',title='P1 final-state privacy (fresh direct full-grid)');save(f,'P1_final_TV_vs_M')
f,ax=plt.subplots();ax.loglog(x,tv,marker='o');ax.axvline(4,color='gray',ls='--');ax.set(xlabel=r'$x_M=M/L_{TV}(M)$',ylabel='Final symmetric TV',title='P1 final TV vs finite-memory-normalized horizon');save(f,'P1_final_TV_vs_xM')
f,ax=plt.subplots();ax.plot([float(r['x_right']) for r in loc],[float(r['beta_local']) for r in loc],marker='o');ax.axhline(0,color='gray',lw=.8);ax.axvline(4,color='gray',ls='--');ax.set(xscale='log',xlabel=r'$x_M$ (right endpoint)',ylabel='Local beta',title='P1 local privacy-decay exponent');save(f,'P1_local_beta_vs_xM')
for delta,label in [('0.001','1e-3'),('1e-05','1e-5')]:
 rr=[r for r in comp if r['delta']==delta];f,ax=plt.subplots();ax.loglog([int(r['M']) for r in rr],[float(r['epsilon_final_upper_cert']) for r in rr],marker='o',label='P1 final epsilon upper');ax.loglog([int(r['M']) for r in rr],[float(r['epsilon_FS_lower']) for r in rr],marker='s',label='FS lower');ax.loglog([int(r['M']) for r in rr],[float(r['epsilon_FS_upper']) for r in rr],marker='^',label='FS upper');ax.set(xlabel='M',ylabel='epsilon',title=f'P1 vs Feldman-Shenfeld, delta={label}');ax.legend();save(f,f'P1_epsilon_vs_FS_delta_{label}')
f,ax=plt.subplots();by={}
for r in comp:by.setdefault(int(r['M']),[]).append(float(r['ratio_final_upper_over_FS_lower']))
xs=sorted(by);ax.semilogx(xs,[max(by[m]) for m in xs],marker='o');ax.axhline(.99,color='red',ls='--');ax.set(xlabel='M',ylabel='Final epsilon upper / FS lower',title='P1 strict FS separation ratio');save(f,'P1_FS_ratio_vs_M')
if k4p.exists():
 rr=rows('USER_PACKET_K4_ROBUSTNESS.csv');f,ax=plt.subplots();ax.loglog(M,tv,marker='o',label='k=2');k4m={}
 for r in rr:k4m.setdefault(int(r['M']),float(r.get('tv_final','nan')))
 if k4m:ax.loglog(sorted(k4m),[k4m[m] for m in sorted(k4m)],marker='s',label='k=4')
 ax.set(xlabel='M',ylabel='Final symmetric TV',title='P1 user-packet robustness');ax.legend();save(f,'P1_k4_robustness')
f,ax=plt.subplots(); conv=rows('NUMERICAL_CONVERGENCE_FINAL.csv');M0=[1024,4096,16384]
for grid,col in [('BASE','tv_base'),('GRID','tv_grid'),('GRID2','tv_grid2'),('DOMAIN','tv_domain')]:ax.plot(M0,[float(next(r for r in conv if int(r['M'])==m)[col]) for m in M0],marker='o',label=grid)
ax.set(xscale='log',yscale='log',xlabel='M',ylabel='Final symmetric TV',title='P1 numerical convergence');ax.legend();save(f,'P1_numerical_convergence')
print(json.dumps({'figures':sorted(p.name for p in OUT.iterdir())},indent=2))
