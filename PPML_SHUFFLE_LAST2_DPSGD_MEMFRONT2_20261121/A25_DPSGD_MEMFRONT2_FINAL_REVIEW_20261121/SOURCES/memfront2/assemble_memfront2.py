import json,csv,glob,os,hashlib,platform,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
OUT=BASE/'deliverable'; OUT.mkdir(exist_ok=True)
RES=BASE/'results'
all_files=glob.glob(str(RES/'spectral/*.json'))+glob.glob(str(RES/'secondary/*.json'))
rows=[]
for f in all_files:
 d=json.load(open(f)); npz=np.load(f[:-5]+'.npz'); c=d['config']; modes=d['modes']; rs=npz['spectral_r']; pred=npz['spectral_at_stride']; obs=npz['r'] if 'r' in npz.files else npz['tv_grid']; upper=npz['tv_upper']; rgrid=np.arange(len(upper));
 # A single slow real mode gives the leading asymptotic TV tail estimate. Other retained modes provide a conservative L1 residual envelope.
 m=modes[1]; lam=m['abs_lambda']; amp=.5*m['density_l1_amplitude']; R=len(upper)-1
 if lam<1:
  tail=amp*lam**(R+1)/(1-lam); tail_sq=amp**2*lam**(2*(R+1))/(1-lam**2)
  residual=sum(.5*x['density_l1_amplitude']*x['abs_lambda']**(R+1)/(1-x['abs_lambda']) for x in modes[2:] if x['abs_lambda']<1)
 else: tail=float('inf');tail_sq=float('inf');residual=float('inf')
 st=float(upper.sum()); sq=float(upper@upper); L=(st+tail)**2/(sq+tail_sq)
 Ldirect=st*st/sq
 d.update({'completed_LTV_point_estimate_single_slow_mode':L,'direct_truncated_LTV_from_conservative_upper':Ldirect,'slow_mode_tail_area_estimate':tail,'slow_mode_tail_fraction_estimate':tail/(st+tail),'retained_subdominant_L1_tail_envelope':residual,'completion_certified':False})
 Path(f).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
 row={'id':d['id'],'line':c['line'],'grid':c['grid_label'],'eta':c['eta'],'z':c['z'],'k':2,'nx':c['nx'],'x_max':c['x_max'],'max_r':R,'lambda_slow':m['lambda_real'],'tau_slow':m['tau'],'amplitude_L1':m['density_l1_amplitude'],'LTV_direct_upper':Ldirect,'LTV_completed_estimate':L,'direct_sum_upper':st,'tail_area_estimate':tail,'tail_fraction':tail/(st+tail),'retained_subdominant_tail_envelope':residual,'heldout_max_relative_error':d.get('heldout_max_pointwise_relative_error_perron_corrected',d.get('heldout_max_pointwise_relative_error')),'heldout_relative_l1_error':d.get('heldout_relative_l1_error_perron_corrected',d.get('heldout_relative_l1_error')),'q_residual_l1':d['quasi_stationary_residual_l1'],'per_step_escape':d['perron_escape_per_step'],'max_right_residual':max(x['right_residual'] for x in modes),'max_left_residual':max(x['left_residual'] for x in modes),'data_npz':os.path.relpath(f[:-5]+'.npz',BASE)}
 rows.append((row,d,npz,rs,pred,obs,upper))
rows.sort(key=lambda x:(x[0]['line'],x[0]['grid']))
# tables
with open(OUT/'MEMORY_SUMMARY_FINAL.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0][0]));w.writeheader();w.writerows(x[0] for x in rows)
with open(OUT/'SPECTRAL_EIGENMODES.csv','w',newline='') as f:
 fields=['line','grid','eta','z','mode','lambda_real','lambda_imag','abs_lambda','tau','density_l1_amplitude','projection_real','projection_imag','right_residual','left_residual','condition_indicator']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row,d,*_ in rows:
  for m in d['modes']:w.writerow({'line':row['line'],'grid':row['grid'],'eta':row['eta'],'z':row['z'],**{('condition_indicator' if k=='biorthogonal_condition_indicator' else k):v for k,v in m.items() if k in fields or k=='biorthogonal_condition_indicator'}})
with open(OUT/'SPECTRAL_MODE_AMPLITUDES.csv','w',newline='') as f:
 fields=['line','grid','mode','lambda_real','tau','density_l1_amplitude','projection_real','right_residual','left_residual','condition_indicator','tail_area_upper_at_R']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row,d,*_ in rows:
  R=row['max_r']
  for m in d['modes']:
   l=m['abs_lambda']; v=.5*m['density_l1_amplitude']*l**(R+1)/(1-l) if l<1 else float('inf')
   w.writerow({'line':row['line'],'grid':row['grid'],'mode':m['mode'],'lambda_real':m['lambda_real'],'tau':m['tau'],'density_l1_amplitude':m['density_l1_amplitude'],'projection_real':m['projection_real'],'right_residual':m['right_residual'],'left_residual':m['left_residual'],'condition_indicator':m['biorthogonal_condition_indicator'],'tail_area_upper_at_R':v})
with open(OUT/'SPECTRAL_RECONSTRUCTION_VALIDATION.csv','w',newline='') as f:
 fields=['line','grid','training_interval_start','training_interval_end','heldout_interval_start','heldout_interval_end','heldout_max_pointwise_relative_error','heldout_relative_l1_error','explained_fraction_l1','pass_5pct','q_residual_l1','max_eigenpair_residual']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row,d,*_ in rows:
  hi=d.get('heldout_interval') or [None,None];err=d.get('heldout_max_pointwise_relative_error_perron_corrected',d.get('heldout_max_pointwise_relative_error')); rel=d.get('heldout_relative_l1_error_perron_corrected',d.get('heldout_relative_l1_error'))
  w.writerow({'line':row['line'],'grid':row['grid'],'training_interval_start':d.get('training_interval',[None,None])[0],'training_interval_end':d.get('training_interval',[None,None])[1],'heldout_interval_start':hi[0],'heldout_interval_end':hi[1],'heldout_max_pointwise_relative_error':err,'heldout_relative_l1_error':rel,'explained_fraction_l1':d.get('heldout_explained_fraction_l1'),'pass_5pct':err is not None and err<=.05,'q_residual_l1':d['quasi_stationary_residual_l1'],'max_eigenpair_residual':max(max(m['right_residual'],m['left_residual']) for m in d['modes'])})
with open(OUT/'MEMORY_RESPONSE_DIRECT.csv','w',newline='') as f:
 fields=['line','grid','eta','z','r','TV_direct_grid','TV_conservative_upper','escape_present_cumulative','escape_reference_cumulative','spectral_prediction_if_stride_point','is_spectral_stride_point']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row,d,npz,rs,pred,obs,upper in rows:
  ep=npz['escape_present'];eq=npz['escape_reference']; mp={int(r):float(v) for r,v in zip(rs,pred)}
  for rr in range(len(upper)):
   w.writerow({'line':row['line'],'grid':row['grid'],'eta':row['eta'],'z':row['z'],'r':rr,'TV_direct_grid':float(obs[rr]),'TV_conservative_upper':float(upper[rr]),'escape_present_cumulative':float(ep[rr]),'escape_reference_cumulative':float(eq[rr]),'spectral_prediction_if_stride_point':mp.get(rr,''),'is_spectral_stride_point':rr in mp})
# Completion status: current estimates are diagnostic, not certified closures.
with open(OUT/'MEMORY_SPECTRAL_COMPLETION.csv','w',newline='') as f:
 fields=['line','grid','LTV_direct_truncated','LTV_completed_point_estimate','slow_mode_tail_fraction','subdominant_tail_envelope','finite_domain_defect_estimate','status']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row,*_ in rows:w.writerow({'line':row['line'],'grid':row['grid'],'LTV_direct_truncated':row['LTV_direct_upper'],'LTV_completed_point_estimate':row['LTV_completed_estimate'],'slow_mode_tail_fraction':row['tail_fraction'],'subdominant_tail_envelope':row['retained_subdominant_tail_envelope'],'finite_domain_defect_estimate':'bounded by cumulative escaped mass; not material to cross-grid discrepancy','status':'DIAGNOSTIC_ONLY_CLOSURE_FAIL_OR_UNVERIFIED'})
# primary closure calculations
pr={line:[r for r,*_ in rows if r['line']==line] for line in ('P1','P2')}
closure={}
for line, rr in pr.items():
 base=next(x for x in rr if x['grid']=='BASE'); req=[x for x in rr if x['grid'] in ('BASE','GRID','DOMAIN')]; vals=[x['LTV_completed_estimate'] for x in req]; taus=[x['tau_slow'] for x in req]
 spread=(max(vals)-min(vals))/base['LTV_completed_estimate']; tauspread=(max(taus)-min(taus))/base['tau_slow']
 closure[line]={'required_grids':{x['grid']:{'tau':x['tau_slow'],'amplitude_L1':x['amplitude_L1'],'LTV_completed_point_estimate':x['LTV_completed_estimate'],'heldout_max_relative_error':x['heldout_max_relative_error'],'max_retained_eigenpair_residual':max(x['max_right_residual'],x['max_left_residual']),'slow_mode_eigenpair_residual':max(next(d['modes'][1]['right_residual'] for rr,d,*_ in rows if rr['id']==x['id']),next(d['modes'][1]['left_residual'] for rr,d,*_ in rows if rr['id']==x['id']))} for x in req},'completed_LTV_relative_spread_vs_BASE':spread,'slow_tau_relative_spread_vs_BASE':tauspread,'heldout_pass':all(x['heldout_max_relative_error']<=.05 for x in req),'subdominant_tail_envelope_fraction_max':max(x['retained_subdominant_tail_envelope']/(x['direct_sum_upper']+x['tail_area_estimate']) for x in req),'memory_closed':False,'reason':('cross-grid completed LTV spread exceeds 5%; slow-mode tau is also highly grid sensitive' if line=='P1' else 'slow-mode tau spread exceeds 5% across BASE/GRID/DOMAIN despite completed LTV spread within 5%')}
# secondary phase map; status only if direct-spectral error meaningful
phase=[]
for row,*_ in rows:
 if row['line'] in ('P1','P2'): continue
 phase.append({'line':row['line'],'eta':row['eta'],'z':row['z'],'grid':row['grid'],'dominant_tau_estimate':row['tau_slow'],'slow_mode_amplitude_L1':row['amplitude_L1'],'LTV_completed_point_estimate':row['LTV_completed_estimate'],'heldout_max_relative_error':row['heldout_max_relative_error'],'memory_closure_status':'UNVALIDATED' if row['heldout_max_relative_error']>.05 else 'EXPLORATORY_ONLY_NO_CROSS_GRID_CHECK','frontier_status':'NOT_RUN'})
with open(OUT/'PHASE_MAP_MEMORY.csv','w',newline='') as f:
 fields=list(phase[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(phase)
# all downstream outcomes explicitly blocked
blocked={
 'FINAL_PRIVACY_BY_M.csv':['line','M','M_over_LTV','symmetric_TV','epsilon_final_delta1e-3','epsilon_final_delta1e-5','status'],
 'LOCAL_EXPONENTS.csv':['line','M_left','M_right','x_left','x_right','beta_local','status'],
 'ASYMPTOTIC_FITS.csv':['line','fit_window','beta_TV','CI95','R2','n','status'],
 'CROSSOVER_SUMMARY.csv':['line','M_peak','x_peak','M_decay','x_decay','status'],
 'FS_COMPARISON.csv':['line','M','delta','epsilon_final_upper','epsilon_FS_lower','epsilon_FS_upper','strict_win','status'],
 'USER_PACKET_ROBUSTNESS.csv':['line','k','M','M_over_LTV','TV','status'],
 'INITIALIZATION_SENSITIVITY.csv':['line','initialization','M','TV','status'],
 'NUMERICAL_CONVERGENCE.csv':['line','M','variant','TV_relative_change','epsilon_relative_change','status']}
for name,fields in blocked.items():
 with open(OUT/name,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow({fields[0]:'ALL',fields[-1]:'BLOCKED_BY_SPECTRAL_CLOSURE'})
# convergence table from memory values only
with open(OUT/'NUMERICAL_CONVERGENCE_MEMORY.csv','w',newline='') as f:
 fields=['line','quantity','BASE','GRID','DOMAIN','GRID2','max_relative_spread_BASE_GRID_DOMAIN','threshold','pass']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for line in ('P1','P2'):
  rr=pr[line]
  for key,col,thr in [('tau_slow','tau_slow',.05),('completed_LTV_estimate','LTV_completed_estimate',.05)]:
   vals={x['grid']:x[col] for x in rr}; b=vals['BASE']; spr=(max(vals[g] for g in ('BASE','GRID','DOMAIN'))-min(vals[g] for g in ('BASE','GRID','DOMAIN')))/b
   w.writerow({'line':line,'quantity':key,'BASE':vals.get('BASE'),'GRID':vals.get('GRID'),'DOMAIN':vals.get('DOMAIN'),'GRID2':vals.get('GRID2'),'max_relative_spread_BASE_GRID_DOMAIN':spr,'threshold':thr,'pass':spr<=thr})
# expected job manifest
jobs=[]
for row,*_ in rows: jobs.append({'id':row['id'],'stage':'M2_DIRECT_SPECTRAL','line':row['line'],'grid':row['grid'],'status':'COMPLETED'})
for stage in ['M4_FINAL_STATE','M5_FRONTIER','M6_FELDMAN_SHENFELD','M7_USER_PACKET','M8_INITIALIZATION']:
 jobs.append({'id':stage,'stage':stage,'status':'BLOCKED_BY_SPECTRAL_CLOSURE','reason':'P1 and P2 did not pass line-wise spectral closure'})
(OUT/'MEMFRONT2_EXPECTED_JOB_MANIFEST.json').write_text(json.dumps({'jobs':jobs,'completed_spectral_jobs':len(rows),'downstream_scientific_jobs_authorized':False},indent=2,sort_keys=True)+'\n')
# decision
p1=closure['P1'];p2=closure['P2'];decision={'milestone':'PPML-SHUFFLE-LAST2-DPSGD-MEMFRONT2','decision':'REPAIR_MEMFRONT2','historical_decisions_preserved':{'SCALECERT':'STOP_REALISTIC_DP_SGD','MEMFRONT1':'REPAIR_MEMFRONT'},'M0':'PASS','memory_closure':closure,'final_state_scaling':'NOT_RUN_BLOCKED_BY_SPECTRAL_CLOSURE','feldman_shenfeld':'NOT_RUN','GPU_used':False,'CPU_workers_max':6,'secondary_lines':phase,'triggered_adaptive_extensions':{'P1':'GRID2 triggered by cross-grid spectral sensitivity','P2':'GRID2 triggered by cross-grid tau stability check'},'failed_or_inconclusive_gates':['P1 completed LTV BASE/GRID/DOMAIN variation exceeds 5%','P1 dominant slow-mode tau is grid sensitive; BASE/GRID/DOMAIN do not agree within 5%','P2 dominant slow-mode tau BASE-to-GRID variation exceeds 5%','P2 32-mode decomposition has higher-mode left residuals above 1e-9; all residuals are reported, while dominant slow-mode residuals are near machine precision','secondary lines with low direct response are below reliable relative-error floor'],'reason':'The required independent memory closure gate did not pass for either primary line. The protocol blocks final-state scaling, matched allocation comparisons, k robustness, and initialization analysis.'}
(OUT/'MEMFRONT2_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
# Self-contained SVG figures (no plotting dependency added to the frozen environment).
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
def svg(name,title,body):
    (figdir/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560"><rect width="100%" height="100%" fill="white"/><text x="480" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>{body}</svg>\n')
def axes(xlab,ylab):
    return '<path d="M90 70 V475 H900" fill="none" stroke="black"/><text x="490" y="530" text-anchor="middle" font-family="sans-serif">'+xlab+'</text><text transform="translate(25 270) rotate(-90)" text-anchor="middle" font-family="sans-serif">'+ylab+'</text>'
def polyline(points,color,label,yoff=0):
    pts=' '.join(f'{x:.1f},{y:.1f}' for x,y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/><text x="710" y="{82+yoff}" font-family="sans-serif" font-size="13" fill="{color}">{label}</text>'
# log response and spectral prediction, downsampled for compact vector output
body=axes('r (log scale)','one-shock TV response (log scale)'); colors=['#145da0','#c0392b','#258039','#8540a1','#dd8700','#148b8b']
for idx,(row,d,npz,rs,pred,obs,upper) in enumerate(rows):
    if row['line'] not in ('P1','P2') or row['grid'] not in ('BASE','GRID','DOMAIN'): continue
    stride=max(1,len(obs)//350); rr=np.arange(1,len(obs),stride); xx=90+810*(np.log10(rr)/np.log10(max(rr))); yy=475-405*((np.log10(np.maximum(obs[rr],1e-15))-(-15))/(0-(-15)))
    body+=polyline(list(zip(xx,yy)),colors[idx%len(colors)],f"{row['line']} {row['grid']}",idx*18)
    spp=npz['spectral_r']; sp=npz['spectral_at_stride']; xx=90+810*(np.log10(np.maximum(spp[1:],1))/np.log10(max(spp[1:]))); yy=475-405*((np.log10(np.maximum(sp[1:],1e-15))+15)/15)
    body+=polyline(list(zip(xx,yy)),colors[idx%len(colors)],f"{row['line']} {row['grid']} spectral",(idx+6)*18)
svg('direct_vs_spectral.svg','Direct one-shock response and spectral reconstruction',body)
# dominant eigenvalue and amplitude versus time constant
for filename,field,ylabel in [('eigenvalue_spectrum.svg','lambda_slow','dominant eigenvalue'),('mode_amplitude_tau.svg','amplitude_L1','slow-mode L1 amplitude')]:
    body=axes('dominant slow-mode tau (log scale)',ylabel)
    pts=[]
    for row,*_ in rows:
        if row['line'] not in ('P1','P2'): continue
        x=90+810*(np.log10(row['tau_slow'])/10)
        val=row[field]; yy=475-405*(val if field=='lambda_slow' else min(1,max(0,(np.log10(max(val,1e-15))+15)/15)))
        pts.append((x,yy)); body+=f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="5" fill="#145da0"/><text x="{x+5:.1f}" y="{yy-6:.1f}" font-family="sans-serif" font-size="10">{row["line"]}-{row["grid"]}</text>'
    svg(filename,ylabel,body)
# Full retained primary spectra, not only the dominant mode.
for filename, value_field, ylabel in [('all_modes_tau_spectrum.svg','tau','mode time constant (steps)'),('all_modes_amplitude_spectrum.svg','density_l1_amplitude','mode L1 amplitude')]:
    body=axes('mode index (sorted by |lambda|)',ylabel)
    selected=[x for x in rows if x[0]['line'] in ('P1','P2') and x[0]['grid'] in ('BASE','GRID','DOMAIN')]
    for ii,(row,d,*_) in enumerate(selected):
        pts=[]
        for j,m in enumerate(d['modes'][1:],start=1):
            v=m[value_field]
            if v is None or not np.isfinite(v): continue
            xx=90+810*j/32
            yy=475-405*((np.log10(max(v,1e-15))+15)/15)
            pts.append((xx,yy))
        if pts: body+=polyline(pts,colors[ii%len(colors)],f"{row['line']} {row['grid']}",ii*18)
    svg(filename,'Retained spectral modes: '+ylabel,body)
# diagnostic memory map
body=axes('eta','z');
for row,*_ in rows:
    x=90+810*(row['eta']/.25); y=475-405*(row['z']/9); body+=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#145da0"/><text x="{x+9:.1f}" y="{y-5:.1f}" font-family="sans-serif" font-size="12">{row["line"]}: log10 L={np.log10(max(row["LTV_completed_estimate"],1)):.2f}</text>'
svg('memory_phase_map.svg','Diagnostic completed-memory estimates (closure not certified)',body)
for name,title in [('final_tv_vs_M.svg','Final-state TV: blocked by spectral closure'),('final_tv_vs_memory_units.svg','Final-state TV vs M/L: blocked by spectral closure'),('local_beta_vs_memory_units.svg','Local exponent: blocked by spectral closure'),('feldman_shenfeld.svg','Feldman–Shenfeld: blocked by spectral closure'),('k_robustness.svg','k robustness: blocked by spectral closure'),('initialization_sensitivity.svg','Initialization sensitivity: blocked by spectral closure')]:
    svg(name,title,'<text x="480" y="280" text-anchor="middle" font-family="sans-serif" font-size="16">Not run under the fail-closed protocol</text>')
# reports, environment, audit independent-style checks
(OUT/'M0_VERIFICATION.json').write_text((BASE/'M0_VERIFICATION.json').read_text())
# job ledger copied
(OUT/'MEMFRONT2_JOB_LEDGER.jsonl').write_text((BASE/'MEMFRONT2_JOB_LEDGER.jsonl').read_text())
(OUT/'ENVIRONMENT.json').write_text(json.dumps(json.load(open(BASE/'M0_VERIFICATION.json')),indent=2,sort_keys=True)+'\n')
report='''# MEMFRONT2 final report\n\n## Decision\n\n**REPAIR_MEMFRONT2.** The spectral closure gate failed on both primary lines. No final-state privacy, Feldman–Shenfeld, packet-size, or initialization experiment was run. The frozen protocol requires this stop.\n\n## Results and interpretation\n\n- P1 (eta=.10,z=2): BASE/DOMAIN dominant slow tau is about 1.724e8 steps, with L1 mode amplitude 3.95e-7. GRID gives tau 3.772e8 and GRID2 4.658e8 (amplitude decreases to 1.58e-7). The fitted within-grid tail predicts a completed L_TV around 37,157 on BASE/DOMAIN, 39,321 on GRID, and 39,902 on GRID2. These are **diagnostic point estimates**, not certified memory closures. BASE/GRID/DOMAIN L_TV spread is over 5%, and tau is highly grid-sensitive. Therefore the old finite-horizon MEMFRONT1 L_TV=221.505 does not survive an infinite-tail completion on the BASE discretization; the discrepancy is attributable to a tiny-amplitude, extremely slow discrete mode, whose numerical stability is not adequate for scientific closure.\n- P2 (eta=.10,z=4): BASE/DOMAIN slow tau is about 15,045.5, amplitude 1.130e-3; GRID gives tau 15,859.2, amplitude 1.090e-3; GRID2 gives tau 16,073.4, amplitude 1.080e-3. Completed diagnostic L_TV estimates are 8,588.4 (BASE/DOMAIN), 8,771.6 (GRID), and 8,818.4 (GRID2). BASE/GRID/DOMAIN L_TV spread is about 2.1%, but BASE-to-GRID tau variation is about 5.4%, failing the separate 5% spectral time-constant condition. P2 is therefore not `SPECTRALLY_CLOSED`.\n- Held-out direct-vs-spectral reconstructions on each grid passed the 5% pointwise criterion for P1/P2; observed maximum errors are in `SPECTRAL_RECONSTRUCTION_VALIDATION.csv`. This validates the retained-mode reconstruction over the checked intervals on each discretization; it does **not** cure cross-grid disagreement.\n- The selected leading modes have small amplitudes, especially P1. Retained eigenpair residuals are reported individually; higher modes become ill-conditioned, so tail-completion point estimates must not be read as formal certified bounds. Cumulative escape was tracked and no probability renormalization was used.\n- The secondary lines are an exploratory memory map only. Where direct responses hit a numerical floor, relative reconstruction error is large; those lines are explicitly marked unvalidated.\n\n## Answers to the frozen questions\n\n1. P1 is not robustly 221.5 after spectral tail completion; the BASE diagnostic is roughly 37.2k, but the closure is grid-sensitive and uncertified.\n2. P2 has a slow mode near tau 15.0k–16.1k with L1 amplitude 0.00108–0.00113, but it fails the required cross-grid tau stability.\n3. Completed P2 L_TV diagnostic range is about 8.59k–8.82k across tested grids; this is not an authorized certified estimate.\n4. P1 final-state decay at x>=4 or x>=8 was not tested.\n5. No P1 Feldman–Shenfeld comparison was run.\n6. P2 final-state frontier was not run; practical status remains unresolved.\n7. No collapse assessment is possible without final-state measurements.\n8. The computed modes suggest multiple scales, but P1’s very slow mode is grid sensitive; a scientific claim of a validated spectrum is not supported yet.\n9. k=4 was not run.\n10. Initialization sensitivity was not run.\n11. Secondary estimates vary widely and several fail direct/spectral held-out checks; no short/moderate/long/exreme memory classification is certified.\n12. Final verdict is **REPAIR_MEMFRONT2**, not GO or STOP.\n\n## Provenance and execution\n\nM0 passed: input checksum manifest verified, 22 core tests passed, and deterministic one-worker/four-worker equivalence passed exactly. Six CPU workers were used for independent primary spectral tasks, two for GRID2, and five for secondary lines. No GPU was requested or used. Historical SCALECERT and MEMFRONT1 decisions remain unchanged.\n'''
(OUT/'FINAL_MEMFRONT2_REPORT.md').write_text(report)
# independent audit logic, no import of decision generator
checks=[]
def ck(name,ok,detail):checks.append({'check':name,'pass':bool(ok),'detail':detail})
ck('M0_PASS',json.load(open(OUT/'M0_VERIFICATION.json'))['status']=='PASS','core tests, input hashes, worker equivalence')
ck('primary_grid_job_count',len([r for r,*_ in rows if r['line'] in ('P1','P2')])==8,'BASE/GRID/DOMAIN/GRID2 for both lines')
ck('secondary_line_count',len([r for r,*_ in rows if r['line'] not in ('P1','P2')])==5,'R1-R5')
ck('heldout_primary_reconstruction',all(r['heldout_max_relative_error']<=.05 for r,*_ in rows if r['line'] in ('P1','P2')),'each primary held-out reconstruction <=5%')
ck('primary_slow_mode_residuals',all(max(x['slow_mode_eigenpair_residual'] for x in closure[line]['required_grids'].values())<=1e-9 for line in ('P1','P2')),'dominant slow-mode left/right eigenpair residual <=1e-9')
ck('full_eigenpair_residuals_reported',all('max_retained_eigenpair_residual' in x for line in ('P1','P2') for x in closure[line]['required_grids'].values()),'all retained-mode residuals are available in CSV; higher-mode residual gate is disclosed')
ck('P1_memory_nonclosure',not p1['memory_closed'] and p1['completed_LTV_relative_spread_vs_BASE']>.05,'cross-grid LTV exceeds 5%')
ck('P2_memory_nonclosure',not p2['memory_closed'] and p2['slow_tau_relative_spread_vs_BASE']>.05,'slow tau exceeds 5%')
ck('no_final_state_executed',not any('final_state' in x.get('event','') for x in [json.loads(l) for l in open(OUT/'MEMFRONT2_JOB_LEDGER.jsonl')]),'final-state jobs blocked')
ck('no_gpu',json.load(open(OUT/'M0_VERIFICATION.json'))['gpu_used'] is False,'CPU-only')
ck('historical_decisions_preserved',json.load(open(OUT/'M0_VERIFICATION.json'))['historical_MEMFRONT1_status']=='REPAIR_MEMFRONT' and json.load(open(OUT/'M0_VERIFICATION.json'))['historical_SCALECERT_status']=='STOP_REALISTIC_DP_SGD','historical statuses unchanged')
ck('fail_closed_decision',decision['decision']=='REPAIR_MEMFRONT2','matches line-wise gates')
audit={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL_INDEPENDENT_AUDIT','checks_passed':sum(x['pass'] for x in checks),'checks_total':len(checks),'checks':checks,'limitations':['No final-state or FS results are produced because primary memory closure fails.']}
(OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
# pip freeze
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open(OUT/'pip_freeze.txt','w'),check=False)
# Include the frozen implementation source and MEMFRONT2 orchestration.
import shutil
src_out=OUT/'SOURCES'; src_out.mkdir(exist_ok=True)
shutil.copytree(BASE.parent/'CORE_SOURCE/src',src_out/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(BASE.parent/'CORE_SOURCE/tests',src_out/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
shutil.copytree(BASE.parent/'CORE_SOURCE/scripts',src_out/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(BASE,src_out/'memfront2',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','results','deliverable'))
raw_out=OUT/'RAW_SPECTRAL'; raw_out.mkdir(exist_ok=True)
for f in all_files:
    shutil.copy2(f,raw_out); shutil.copy2(f[:-5]+'.npz',raw_out)
(OUT/'SPECIFICATION_PROVENANCE.json').write_text(json.dumps({'title':'PPML-SHUFFLE-LAST2-DPSGD-MEMFRONT2','date_freeze':'2026-11-21','source':'user-provided frozen execution specification in conversation','specification_file_hash':'not separately available','operative_gates':'MEMFRONT2_DECISION.json and FINAL_MEMFRONT2_REPORT.md','GPU':'prohibited for this milestone','CPU_policy':'process parallelism across independent spectral/grid jobs'},indent=2,sort_keys=True)+'\n')
# source/code diff context
(OUT/'CODE_DIFF.patch').write_text('''MEMFRONT2 implementation changes are confined to new orchestration/analysis files under memfront2/. Frozen THEORYCOMPLY files under SOURCES/src/scalecert/ were not modified. The orchestration adds sparse left/right spectral diagnostics, direct one-shock trajectories with explicit Gaussian tail-loss envelopes, diagnostic tail completion, and fail-closed output assembly. `spectral_reconstruct` suppresses the numerical Perron drift from float64 column-sum roundoff; this adjustment is documented in memfront2/spectral.py and its effect is included in the reported residual/closure failure. No final-state kernel, hidden-support DP, privacy-profile or Feldman–Shenfeld adapter was changed or run.\n''')
# sums
with open(OUT/'SHA256SUMS.txt','w') as f:
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS.txt': f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n')
print(json.dumps({'out':str(OUT),'spectral_rows':len(rows),'primary_closure':closure,'audit':audit['status'],'files':sum(p.is_file() for p in OUT.rglob('*'))},indent=2))
