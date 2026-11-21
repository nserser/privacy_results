#!/usr/bin/env python3
import argparse, json, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


def save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--summary',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    s=pathlib.Path(a.summary); out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(s/'all_results.csv'); checks=json.loads((s/'checks.json').read_text())

    # 1: main continuous scaling, forward/reverse plus exact transcript.
    d=df[(df.family=='nonlinear')&(df.status=='OK')&(df.stage=='nonlinear')&(df.b==1)&(df.k==2)&np.isclose(df.A,.55)&np.isclose(df.B,.05)&np.isclose(df.tau,1.0)].sort_values('M')
    if len(d):
        fig,ax=plt.subplots()
        ax.loglog(d.M,d.transcript_chi2,'o-',label='transcript')
        ax.loglog(d.M,d.final_chi2_forward,'s-',label='final forward')
        ax.loglog(d.M,d.final_chi2_reverse,'^-',label='final reverse')
        ax.set_xlabel('M'); ax.set_ylabel('chi-square leakage'); ax.legend(); ax.set_title('Symmetric transcript vs final scaling')
        save(fig,out/'01_scaling_symmetric.png')

    # 2: no-forgetting control.
    if 'rho' in df.columns:
        g=df[(df.family=='gaussian')&(df.status=='OK')&np.isclose(df.rho,1.0)].sort_values('M')
        if len(g):
            fig,ax=plt.subplots(); ax.loglog(g.M,g.transcript_chi2,'o-',label='transcript'); ax.loglog(g.M,g.final_chi2,'s-',label='final')
            ax.set_xlabel('M');ax.set_ylabel('chi-square');ax.legend();ax.set_title('No-forgetting control');save(fig,out/'02_no_forgetting.png')

    # 3: independent-memory scaling collapse.
    rows=checks.get('memory_collapse',{}).get('rows',[])
    if rows:
        rr=pd.DataFrame(rows); fig,ax=plt.subplots()
        if 'A' in rr: ax.plot(rr.A,rr.normalized,'o-')
        else: ax.plot(np.arange(len(rr)),rr.normalized,'o-')
        ax.set_xlabel('memory setting'); ax.set_ylabel('chi2 gap * L_emp / M'); ax.set_title('Privacy-memory scaling collapse')
        save(fig,out/'03_memory_collapse.png')

    # 4: multi-epoch normalized collapse.
    rows=checks.get('multi_epoch',{}).get('rows',[])
    if rows:
        fig,ax=plt.subplots()
        for i,r in enumerate(rows):
            vals=r.get('normalized_EM',[])
            if vals: ax.plot(np.arange(1,len(vals)+1),vals,'o-',label=f'config {i+1}')
        ax.set_xlabel('registered epoch index'); ax.set_ylabel('D2 gap * L_emp / (E M)'); ax.set_title('Multi-epoch normalized separation')
        if len(rows)>1: ax.legend()
        save(fig,out/'04_multi_epoch_collapse.png')

    # 5: DP-SGD pre-baseline robust lines use symmetric-TV exponent.
    fits=checks.get('dpsgd_bridge',{}).get('fits',[])
    if fits:
        ff=pd.DataFrame([{
            'key':r.get('key'),
            'tv_beta':r.get('tv_fit',{}).get('beta',np.nan),
            'tv_r2':r.get('tv_fit',{}).get('r2',np.nan),
            'tr_beta':r.get('transcript_chi2_fit',{}).get('beta',np.nan),
            'good':r.get('good',False),
        } for r in fits])
        fig,ax=plt.subplots();
        good=ff.good.astype(bool)
        ax.scatter(ff.tr_beta[~good],ff.tv_beta[~good],marker='x',label='not certified')
        ax.scatter(ff.tr_beta[good],ff.tv_beta[good],marker='o',label='pre-baseline certified')
        ax.axhline(.55,linestyle='--'); ax.set_xlabel('transcript chi2 exponent'); ax.set_ylabel('final symmetric-TV exponent'); ax.set_title('DP-SGD extrapolation scaling')
        ax.legend(); save(fig,out/'05_dpsgd_exponents.png')

    # 6: strict Feldman lower-bound comparisons if present.
    comps=checks.get('feldman_shenfeld',{}).get('comparisons',[])
    if comps:
        cc=pd.DataFrame(comps)
        for (key,delta),g in cc.groupby(['key','delta']):
            g=g.sort_values('M'); fig,ax=plt.subplots()
            ax.plot(g.M,g.final_epsilon_numerical_upper,'o-',label='final epsilon numerical upper')
            ax.plot(g.M,g.fs_epsilon_lower,'s-',label='FS lower epsilon')
            if 'fs_epsilon_upper' in g: ax.plot(g.M,g.fs_epsilon_upper,'^-',label='FS upper epsilon')
            ax.set_xlabel('M'); ax.set_ylabel('epsilon'); ax.set_title(f'{key}, delta={delta:g}'); ax.legend()
            safe=key.replace('.','p').replace('/','_'); save(fig,out/f'06_fs_{safe}_delta_{delta:g}.png')

    print(out)

if __name__=='__main__': main()
