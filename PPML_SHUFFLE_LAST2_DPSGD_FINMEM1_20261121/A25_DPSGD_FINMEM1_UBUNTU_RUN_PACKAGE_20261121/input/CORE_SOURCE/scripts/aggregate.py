#!/usr/bin/env python3
import argparse,json,math,pathlib,sys,yaml
from collections import Counter,defaultdict
import numpy as np
import pandas as pd

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scalecert.privacy import fit_power_exponent
from scalecert.plan import build_plan
from scalecert.experiments import canonical_id


def load_records(path):
    out=[]
    for p in pathlib.Path(path).rglob('*.json'):
        try:r=json.loads(p.read_text())
        except Exception: continue
        if isinstance(r,dict) and 'family' in r and 'config' in r:
            r['_source_file']=str(p); out.append(r)
    return out


def flat(r):
    x={'experiment_id':r.get('experiment_id'),'family':r.get('family'),'status':r.get('status')}
    for k,v in r.get('config',{}).items():
        if isinstance(v,(str,int,float,bool)): x[k]=v
    for k,v in r.get('metrics',{}).items():
        if isinstance(v,(str,int,float,bool)): x[k]=v
    d=r.get('diagnostics',{})
    for k in ('present_edge_mass_5','absent_edge_mass_5','escaped_present','escaped_absent'):
        if k in d and isinstance(d[k],(int,float)): x[k]=d[k]
    return x


def rel(a,b): return abs(float(a)-float(b))/max(abs(float(b)),1e-300)

def fit_both(g):
    return {
        'forward':fit_power_exponent(g.M,g.final_chi2_forward),
        'reverse':fit_power_exponent(g.M,g.final_chi2_reverse),
        'transcript':fit_power_exponent(g.M,g.transcript_chi2),
    }

def fit_good(f,threshold=1.1):
    return np.isfinite(f['beta']) and f['beta']>threshold and f['r2']>0.95


def expected_completeness(config_path,records):
    C=yaml.safe_load(open(config_path)); jobs=build_plan(C); expected={canonical_id(j['family'],j['config']):j for j in jobs}
    counts=Counter(r.get('experiment_id') for r in records)
    actual=set(k for k in counts if k)
    missing=sorted(set(expected)-actual); extra=sorted(actual-set(expected)); duplicates=sorted(k for k,v in counts.items() if k and v>1)
    return {'pass':not missing and not extra and not duplicates,'expected_count':len(expected),'observed_unique_count':len(actual),
            'missing':missing,'extra':extra,'duplicates':duplicates}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); ap.add_argument('--config',required=True); ap.add_argument('--baseline',default='')
    a=ap.parse_args(); out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rec=load_records(a.input); df=pd.DataFrame([flat(r) for r in rec]); df.to_csv(out/'all_results.csv',index=False)
    checks={}; fits=[]
    checks['expected_manifest']=expected_completeness(a.config,rec)
    failures=[r for r in rec if r.get('status')!='OK']
    byid={r.get('experiment_id'):r for r in rec if r.get('experiment_id')}

    # 1. Independent continuous numerical convergence + expm anchor.
    conv=df[(df.get('stage',pd.Series(dtype=str))=='convergence')&(df.status=='OK')].copy() if not df.empty else pd.DataFrame()
    details=[]; cpass=not conv.empty
    required_basic={'reference','domain','grid','time','joint'}
    if not conv.empty:
        for (grp,M),g in conv.groupby(['convergence_group','M']):
            mp={str(r.variant_label):r for _,r in g.iterrows()}
            if not required_basic.issubset(mp): cpass=False; details.append({'group':grp,'M':int(M),'error':'missing basic variants'}); continue
            ref=mp['reference']; joint=mp['joint']; row={'group':grp,'M':int(M)}; ok=True
            for label in ['domain','grid','time']:
                for metric in ['final_chi2_forward','final_chi2_reverse']:
                    d=rel(mp[label][metric],ref[metric]); row[f'{label}_{metric}_rel']=d; ok &= d<=0.02
            edge=max(float(joint.get('present_edge_mass_5',0)),float(joint.get('absent_edge_mass_5',0)))
            row['edge_mass']=edge; ok &= edge<=1e-5 and np.isfinite(joint.final_chi2_reverse)
            if 'expm' in mp:
                for metric in ['final_chi2_forward','final_chi2_reverse']:
                    d=rel(joint[metric],mp['expm'][metric]); row[f'joint_expm_{metric}_rel']=d; ok &= d<=0.02
                row['expm_anchor']=True
            row['pass']=bool(ok); cpass &= ok; details.append(row)
        # Every convergence group must contain at least one expm anchor.
        for grp,g in conv.groupby('convergence_group'):
            if 'expm' not in set(g.variant_label.astype(str)):
                cpass=False; details.append({'group':grp,'error':'missing expm reference anchor'})
            gj=g[g.variant_label=='joint']
            if len(gj.M.unique())>=3:
                for metric in ['final_chi2_forward','final_chi2_reverse']:
                    f=fit_power_exponent(gj.M,gj[metric]); fits.append({'kind':'convergence','group':grp,'variant':'joint','metric':metric,**f})
                    cpass &= np.isfinite(f['beta'])
    checks['convergence']={'pass':bool(cpass),'details':details}

    # 2. Main symmetric nonlinear scaling.
    nl=df[(df.family=='nonlinear')&(df.status=='OK')&(df.get('stage','')=='nonlinear')].copy() if not df.empty else pd.DataFrame()
    anchor=nl[(nl.b==1)&(nl.k==2)&np.isclose(nl.m,.5)&np.isclose(nl.A,.55)&np.isclose(nl.B,.05)&np.isclose(nl.tau,1)] if not nl.empty else pd.DataFrame()
    if len(anchor.M.unique())>=3:
        fs=fit_both(anchor); fits += [{'kind':'main','metric':k,**v} for k,v in fs.items()]
        ok=fit_good(fs['forward'],1.7) and fit_good(fs['reverse'],1.7) and 0.8<=fs['transcript']['beta']<=1.2 and fs['transcript']['r2']>0.98
        growing=min(fs['forward']['beta'],fs['reverse']['beta'])>fs['transcript']['beta']+0.1
        checks['main_scaling']={'pass':bool(ok),'growing':bool(growing),'fits':fs}
    else: checks['main_scaling']={'pass':False,'growing':False,'error':'missing anchor'}

    # 3. FINITE1/profile and one-shock theory certificates.
    theory_rows=[]; tpass=True
    for r in rec:
        if r.get('family')=='nonlinear' and r.get('status')=='OK' and r.get('config',{}).get('stage') in ('nonlinear','stress','usercap','convergence'):
            m=r.get('metrics',{}); ratio=m.get('finite1_bound_ratio_to_observed'); prof=m.get('theory_profile_certificate',{})
            ok=ratio is not None and float(ratio)>=1-1e-8 and prof.get('pass') is True
            tpass &= ok; theory_rows.append({'id':r['experiment_id'],'ratio':ratio,'profile_pass':prof.get('pass'),'pass':bool(ok)})
    mem_raw=[r for r in rec if r.get('family')=='memory']
    for r in mem_raw:
        ok=r.get('status')=='OK' and r.get('metrics',{}).get('theory_envelope_pass') is True and r.get('diagnostics',{}).get('memory_tail',{}).get('tail_complete') is True
        tpass &= ok; theory_rows.append({'id':r.get('experiment_id'),'memory_envelope':r.get('metrics',{}).get('theory_envelope_pass'),'tail_complete':r.get('diagnostics',{}).get('memory_tail',{}).get('tail_complete'),'pass':bool(ok)})
    checks['theory_certificates']={'pass':bool(tpass and theory_rows),'rows':theory_rows}

    # 4. k packet scaling and at-most-k cap sweep.
    prows=[]; ppass=True
    for kk in [1,2,4]:
        g=nl[(nl.b==1)&(nl.k==kk)&np.isclose(nl.m,.5)&np.isclose(nl.A,.55)&np.isclose(nl.B,.05)&np.isclose(nl.tau,1)]
        if len(g.M.unique())<3: ppass=False; continue
        f=fit_both(g); good=fit_good(f['forward']) and fit_good(f['reverse']); ppass &= good; prows.append({'k':kk,'fits':f,'pass':bool(good)})
    checks['k_packet']={'pass':bool(ppass),'rows':prows}

    uc=df[(df.family=='nonlinear')&(df.status=='OK')&(df.get('stage','')=='usercap')].copy() if not df.empty else pd.DataFrame()
    ucrows=[]; ucpass=not uc.empty
    if not uc.empty:
        for cap,gc in uc.groupby('user_cap'):
            maxrows=[]; complete=True
            for M,gm in gc.groupby('M'):
                got=set(int(x) for x in gm.k); required=set(range(1,int(cap)+1)); complete &= got==required
                gm=gm.copy(); gm['sym']=np.maximum(gm.final_chi2_forward,gm.final_chi2_reverse); rr=gm.loc[gm.sym.idxmax()]
                maxrows.append({'M':int(M),'max_chi2':float(rr.sym),'worst_m':int(rr.k)})
            d=pd.DataFrame(maxrows); f=fit_power_exponent(d.M,d.max_chi2) if len(d)>=3 else {'beta':float('nan'),'r2':float('nan')}
            good=complete and len(d)>=3 and fit_good(f); ucpass &= good; ucrows.append({'cap_k':int(cap),'complete':bool(complete),'fit':f,'rows':maxrows,'pass':bool(good)})
    checks['at_most_k_adjacency']={'pass':bool(ucpass),'caps':ucrows}

    # 5. Collision structural and fixed-G tests.
    cinfo={}; cpass=False
    if not nl.empty:
        gc=nl[(nl.b==8)&(nl.k==4)&np.isclose(nl.A,.55)&np.isclose(nl.B,.05)&np.isclose(nl.tau,1)&(nl.collision_parameterization=='fixed_B')]
        gr=nl[(nl.b==1)&(nl.k==4)&np.isclose(nl.A,.55)&np.isclose(nl.B,.05)&np.isclose(nl.tau,1)&(nl.collision_parameterization=='fixed_B')]
        if len(gc.M.unique())>=3 and len(gr.M.unique())>=3:
            fc=fit_both(gc); fr=fit_both(gr); cpass=fit_good(fc['forward']) and fit_good(fc['reverse']) and min(fc['forward']['beta'],fc['reverse']['beta'])>=min(fr['forward']['beta'],fr['reverse']['beta'])-.5
            cinfo={'fixed_B_collision':fc,'fixed_B_free':fr}
        fg=nl[(nl.k==4)&np.isclose(nl.A,.55)&np.isclose(nl.tau,1)&(nl.collision_parameterization=='fixed_G')]
        fgr=fg[fg.b==1]; fgc=fg[fg.b==8]
        if len(fgr.M.unique())>=3 and len(fgc.M.unique())>=3:
            a1=fit_both(fgr); a8=fit_both(fgc); fgok=fit_good(a1['forward']) and fit_good(a8['forward']) and fit_good(a1['reverse']) and fit_good(a8['reverse']); cpass &= fgok
            cinfo.update({'fixed_G_b1':a1,'fixed_G_b8':a8,'fixed_G_pass':bool(fgok)})
        else: cpass=False; cinfo['fixed_G_error']='missing fixed-G M sweep'
    checks['collision_stress']={'pass':bool(cpass),**cinfo}

    # 6. Finite amplitude.
    arows=[]; apass=True
    for BB in [.025,.05,.10,.20]:
        g=nl[(nl.b==1)&(nl.k==2)&np.isclose(nl.A,.55)&np.isclose(nl.B,BB)&np.isclose(nl.tau,1)]
        if len(g.M.unique())<3: apass=False; continue
        f=fit_both(g); good=fit_good(f['forward']) and fit_good(f['reverse']); apass &= good; arows.append({'B':BB,'fits':f,'pass':bool(good)})
    checks['finite_amplitude']={'pass':bool(apass),'rows':arows}

    # 7. Hidden versus known support.
    ks=[r for r in rec if r.get('family')=='nonlinear' and r.get('status')=='OK' and 'known_support_mean_chi2' in r.get('metrics',{})]
    kv=[]
    for r in ks:
        h=float(r['metrics']['final_chi2_forward']); ksv=float(r['metrics']['known_support_mean_chi2']); kv.append({'hidden':h,'known':ksv,'pass':h<=ksv*(1+1e-8)})
    checks['known_vs_hidden_support']={'pass':bool(kv and all(x['pass'] for x in kv)),'rows':kv}

    # 8. No-forgetting negative control.
    gg=df[(df.family=='gaussian')&(df.status=='OK')] if not df.empty else pd.DataFrame(); nf=gg[(gg.b==1)&(gg.k==2)&np.isclose(gg.mu,.25)&np.isclose(gg.rho,1)] if not gg.empty else pd.DataFrame()
    if len(nf.M.unique())>=3:
        f1=fit_power_exponent(nf.M,nf.final_chi2); f2=fit_power_exponent(nf.M,nf.transcript_chi2); ok=.8<=f1['beta']<=1.2 and .8<=f2['beta']<=1.2
        checks['no_forgetting']={'pass':bool(ok),'final_fit':f1,'transcript_fit':f2}
    else: checks['no_forgetting']={'pass':False,'error':'missing rho=1 control'}

    # 9. Independent memory and chi^2 scaling collapse; metastability.
    mem=df[(df.family=='memory')&(df.status=='OK')] if not df.empty else pd.DataFrame(); collapse=[]
    if not mem.empty and not nl.empty:
        for _,mr in mem.iterrows():
            cand=nl[np.isclose(nl.m,mr.m)&np.isclose(nl.A,mr.A)&np.isclose(nl.B,mr.B)&np.isclose(nl.tau,mr.tau)&(nl.b==1)&(nl.k==2)]
            if cand.empty: continue
            rr=cand.sort_values('M').iloc[-1]; norm=float(rr.gap_chi2*mr.l_emp/rr.M)
            collapse.append({'A':float(mr.A),'M':int(rr.M),'l_emp':float(mr.l_emp),'gap_chi2':float(rr.gap_chi2),'normalized':norm})
    if len(collapse)>=3:
        vals=np.array([x['normalized'] for x in collapse]); cv=float(vals.std()/max(abs(vals.mean()),1e-300)); checks['memory_collapse']={'pass':bool(cv<=.5),'cv':cv,'rows':collapse}
    else: checks['memory_collapse']={'pass':False,'rows':collapse,'error':'need >=3 matched settings'}
    if len(mem)>=3:
        mm=mem[np.isclose(mem.B,.05)&np.isclose(mem.tau,1)].sort_values('A'); arr=mm.l_emp.to_numpy(float); trend=len(arr)>=3 and bool(np.all(np.diff(arr)>=-.05*np.maximum(arr[:-1],1e-12)))
        checks['metastability']={'pass':trend,'A':mm.A.tolist(),'l_emp':arr.tolist()}
    else: checks['metastability']={'pass':False,'error':'missing memory sweep'}

    # 10. Multi-epoch EM/L normalization using independently measured L_emp.
    me=[r for r in rec if r.get('family')=='multiepoch' and r.get('status')=='OK']; merows=[]; mepass=bool(me)
    for r in me:
        c=r['config']; matches=mem[np.isclose(mem.m,c['m'])&np.isclose(mem.A,c['A'])&np.isclose(mem.B,c['B'])&np.isclose(mem.tau,c['tau'])] if not mem.empty else pd.DataFrame()
        if matches.empty: mepass=False; continue
        L=float(matches.iloc[0].l_emp); rows=r['metrics']['rows']; norm=[]
        for x in rows: norm.append(float(x['gap_d2']*L/(x['E']*c['M'])))
        final=np.array([x['final_d2_forward'] for x in rows]); plateau=len(final)>=3 and final[-1]/max(final[-2],1e-300)<=1.30
        cv=float(np.std(norm)/max(abs(np.mean(norm)),1e-300)) if len(norm)>=3 else float('inf'); good=plateau and cv<=.5
        mepass &= good; merows.append({'config':c,'l_emp':L,'normalized_EM':norm,'cv':cv,'plateau':bool(plateau),'pass':bool(good)})
    checks['multi_epoch']={'pass':bool(mepass),'rows':merows}

    # 11. SGLD convergence.
    sg=df[(df.family=='bridge_sgld')&(df.status=='OK')&(df.get('stage','')=='bridges')] if not df.empty else pd.DataFrame(); sgrows=[]; sgpass=not sg.empty and not nl.empty
    if sgpass:
        for (M,k),g in sg.groupby(['M','k']):
            ref=nl[(nl.M==M)&(nl.b==1)&(nl.k==k)&np.isclose(nl.m,.5)&np.isclose(nl.A,.55)&np.isclose(nl.B,.05)&np.isclose(nl.tau,1)]
            if ref.empty: sgpass=False; continue
            target=float(ref.iloc[0].final_d2_forward); g=g.sort_values('eta',ascending=False); errs=[rel(x,target) for x in g.final_d2_forward]
            monotone=all(errs[i+1]<=errs[i]*1.10+1e-12 for i in range(len(errs)-1)); ok=errs[-1]<=.15 and monotone; sgpass &= ok
            sgrows.append({'M':int(M),'k':int(k),'etas':g.eta.tolist(),'relative_errors':errs,'pass':bool(ok)})
    checks['sgld_bridge']={'pass':bool(sgpass and sgrows),'rows':sgrows}

    # 12. DP-SGD discrete spatial/domain convergence.  Use privacy-profile
    # quantities rather than chi^2 because extreme Gaussian tails are numerically
    # hostile to reverse Renyi while (epsilon,delta) is stable at the target deltas.
    dc=df[(df.family=='bridge_dpsgd')&(df.status=='OK')&(df.get('stage','')=='bridge_convergence')] if not df.empty else pd.DataFrame(); dcrows=[]; dcpass=not dc.empty
    certified_lines=set(); numeric_eps_envelope={}
    if not dc.empty:
        for (grp,M),g in dc.groupby(['discrete_convergence_group','M']):
            mp={str(r.variant_label):r for _,r in g.iterrows()}; ok={'dbase','dgrid','ddomain'}.issubset(mp)
            row={'group':grp,'M':int(M)}
            if ok:
                for metric in ['tv_symmetric_upper','epsilon_upper_delta_1e5']:
                    b=float(mp['dbase'][metric]); gg=float(mp['dgrid'][metric]); dd=float(mp['ddomain'][metric])
                    if metric=='tv_symmetric_upper':
                        rg=rel(gg,b); rd=rel(dd,b); tol=.05
                    else:
                        rg=abs(gg-b); rd=abs(dd-b); tol=.02
                    row[f'grid_{metric}_change']=rg; row[f'domain_{metric}_change']=rd; ok &= np.isfinite(gg) and np.isfinite(dd) and rg<=tol and rd<=tol
                # Numerical upper envelope used later for strict external-baseline
                # separation.  max + observed refinement spread is deliberately
                # conservative; only the pre-registered convergence lines use it.
                ck=mp['dbase']; key=f"k{int(ck['k'])}_eta{float(ck['eta']):g}_z{float(ck['noise_multiplier']):g}"
                for delta,metric in [(1e-3,'epsilon_upper_delta_1e3'),(1e-5,'epsilon_upper_delta_1e5')]:
                    vals=[float(mp[v][metric]) for v in ['dbase','dgrid','ddomain']]
                    if all(np.isfinite(vals)):
                        numeric_eps_envelope[(key,int(M),delta)]=max(vals)+(max(vals)-min(vals))
            row['pass']=bool(ok); dcpass &= ok; dcrows.append(row)
        for grp,g in dc.groupby('discrete_convergence_group'):
            if all(x['pass'] for x in dcrows if x['group']==grp) and len(g.M.unique())>=3: certified_lines.add(str(grp))
    checks['dpsgd_numerical_convergence']={'pass':bool(dcpass and certified_lines),'certified_lines':sorted(certified_lines),'rows':dcrows}

    # 13. Robust multi-point DP-SGD extrapolation.  Pre-baseline signal uses
    # symmetric TV decay (stable under tail underflow); final GO additionally
    # requires strict Feldman--Shenfeld lower-bound separation below.
    dp=df[(df.family=='bridge_dpsgd')&(df.status=='OK')&(df.get('stage','')=='bridges')] if not df.empty else pd.DataFrame(); dprows=[]; line_good={}
    if not dp.empty:
        for (k,eta,z),g in dp.groupby(['k','eta','noise_multiplier']):
            if len(g.M.unique())<3: continue
            ftv=fit_power_exponent(g.M,g.tv_symmetric_upper); ftr=fit_power_exponent(g.M,g.transcript_chi2)
            good=int(k)>=2 and np.isfinite(ftv['beta']) and ftv['beta']>.55 and ftv['r2']>.95 and .8<=ftr['beta']<=1.2
            key=f"k{int(k)}_eta{float(eta):g}_z{float(z):g}"; line_good[key]=bool(good)
            dprows.append({'key':key,'k':int(k),'eta':float(eta),'noise_multiplier':float(z),'tv_fit':ftv,'transcript_chi2_fit':ftr,'good':bool(good)})
    required_keys=['k2_eta0.1_z2','k2_eta0.1_z4']; robust=all(line_good.get(k,False) for k in required_keys) and {'z2','z4'}.issubset(certified_lines)
    checks['dpsgd_bridge']={'pass':bool(robust),'required_lines':required_keys,'fits':dprows}

    # 14. Feldman-Shenfeld strict lower-bound separation on both required lines, both deltas, >=3 M.
    bsummary={'pass':False,'error':'baseline missing'}; bpass=False
    if a.baseline and pathlib.Path(a.baseline).exists():
        base=json.loads(pathlib.Path(a.baseline).read_text()); brows=[x for x in base.get('rows',[]) if x.get('status')!='FAILED']; comps=[]; grouped=defaultdict(list)
        for br in brows:
            src=byid.get(br.get('source_experiment_id'))
            if not src or src.get('family')!='bridge_dpsgd' or src.get('config',{}).get('stage')!='bridges': continue
            c=src['config']; delta_val=float(br['delta']); delta=str(delta_val); fe=src.get('metrics',{}).get('epsilon_upper_at_delta',{}).get(delta)
            if fe is None or br.get('epsilon_lower') is None: continue
            key=f"k{int(c['k'])}_eta{float(c['eta']):g}_z{float(c['noise_multiplier']):g}"
            num_env=numeric_eps_envelope.get((key,int(c['M']),delta_val),float('inf'))
            final_upper=max(float(fe),float(num_env))
            strict=np.isfinite(final_upper) and final_upper < .99*float(br['epsilon_lower'])
            row={'key':key,'M':int(c['M']),'delta':delta_val,'final_epsilon_grid_upper':float(fe),'final_epsilon_numerical_upper':float(final_upper),'fs_epsilon_lower':float(br['epsilon_lower']),'fs_epsilon_upper':float(br['epsilon_upper']),'strict_better':bool(strict)}
            comps.append(row); grouped[(key,float(br['delta']))].append(row)
        req=[]
        for key in required_keys:
            for delta in [1e-3,1e-5]:
                rows=grouped.get((key,delta),[]); ok=len({r['M'] for r in rows})>=3 and all(r['strict_better'] for r in rows if r['M'] in sorted({x['M'] for x in rows})[-3:])
                req.append({'key':key,'delta':delta,'pass':bool(ok),'rows':rows})
        bpass=all(x['pass'] for x in req); bsummary={'pass':bool(bpass),'required':req,'comparisons':comps,'probe':base.get('probe'),'runtime_lock':base.get('runtime_lock')}
    checks['feldman_shenfeld']=bsummary

    pd.DataFrame(fits).to_csv(out/'scaling_fits.csv',index=False); (out/'checks.json').write_text(json.dumps(checks,indent=2,sort_keys=True))
    core_keys=['expected_manifest','convergence','main_scaling','theory_certificates','k_packet','at_most_k_adjacency','collision_stress','finite_amplitude','known_vs_hidden_support','no_forgetting','memory_collapse','metastability','multi_epoch','sgld_bridge','dpsgd_numerical_convergence']
    core=all(checks[k]['pass'] for k in core_keys)
    if failures or not checks['expected_manifest']['pass']:
        decision='FAIL_NUMERICAL_VALIDATION'; reason=f"failed/non-OK records={len(failures)} or manifest mismatch"
    elif not core:
        decision='REPAIR_SCALECERT' if checks['main_scaling'].get('growing',False) else 'STOP_REALISTIC_DP_SGD'; reason='one or more theory/numerical compliance gates failed'
    elif not checks['dpsgd_bridge']['pass']:
        decision='STOP_REALISTIC_DP_SGD'; reason='continuous theory is certified but prespecified user-level DP-SGD lines do not robustly show a growing gap'
    elif not checks['feldman_shenfeld']['pass']:
        decision='REPAIR_SCALECERT'; reason='DP-SGD bridge passes but strict lower-bound separation from Feldman-Shenfeld is not certified'
    else:
        decision='GO_REALISTIC_ULDP_ML'; reason='all theory-compliance, numerical, user-level, DP-SGD, and matched baseline gates passed'
    payload={'decision':decision,'reason':reason,'failed_records':len(failures),'checks':{k:v.get('pass') for k,v in checks.items()}}
    (out/'decision.json').write_text(json.dumps(payload,indent=2,sort_keys=True))
    lines=['# SCALECERT Theory-Compliant Aggregate Report','',f'**Decision:** `{decision}`',f'**Reason:** {reason}','','## Gate summary']
    for k,v in checks.items(): lines.append(f"- {k}: {'PASS' if v.get('pass') else 'FAIL'}")
    lines += ['','See `checks.json`, `scaling_fits.csv`, `all_results.csv`, and the exact expected-job manifest for inspectable evidence.']
    (out/'FINAL_SCALECERT_REPORT.md').write_text('\n'.join(lines)+'\n'); print(decision)

if __name__=='__main__': main()
