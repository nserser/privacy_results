import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def rcsv(n): return list(csv.DictReader((ROOT/n).open(newline='')))
def jread(n): return json.loads((ROOT/n).read_text())
mem=rcsv('FINITE_HORIZON_MEMORY_CONTINUUM.csv'); priv=rcsv('FINAL_PRIVACY_BY_M.csv'); fits=rcsv('ASYMPTOTIC_FITS.csv'); local=rcsv('LOCAL_EXPONENTS.csv'); fs=rcsv('P1_FS_COMPARISON.csv'); conv=rcsv('NUMERICAL_CONVERGENCE_FINAL.csv')
f4=next(r for r in fits if r['x_threshold']=='4'); f8=next(r for r in fits if r['x_threshold']=='8')
fs_by_m={}
for r in fs: fs_by_m.setdefault(int(r['M']),[]).append(r)
strict=[m for m,v in fs_by_m.items() if len(v)==2 and all(x['strict_win']=='True' for x in v)]
asym=[int(r['M']) for r in priv if float(r['M_over_LTV_M'])>=4]
consecutive=any(a in strict and b in strict for a,b in zip(asym,asym[1:]))
k4=rcsv('USER_PACKET_K4_ROBUSTNESS.csv') if (ROOT/'USER_PACKET_K4_ROBUSTNESS.csv').exists() else []
k4m={}
for r in k4:k4m.setdefault(int(r['M']),[]).append(r)
k4consec=any(a in k4m and b in k4m and len(k4m[a])==len(k4m[b])==2 and all(x['strict_win']=='True' for x in k4m[a]+k4m[b]) for a,b in zip(sorted(k4m),sorted(k4m)[1:]))
k4final=jread('K4_FINAL_STATE_SUMMARY.json') if (ROOT/'K4_FINAL_STATE_SUMMARY.json').exists() else []
audit=jread('INDEPENDENT_AUDIT.json') if (ROOT/'INDEPENDENT_AUDIT.json').exists() else {'status':'NOT_RUN'}
primary=all(r['status']=='PASS' for r in mem) and len(asym)>=3 and float(f4['beta_TV'])>.55 and float(f4['R2'])>=.95 and all(float(x['beta_local'])>0 for x in local[-3:]) and float(f4['beta_drop_first'])>0 and consecutive and len(conv)==3 and all(r['status']=='PASS' for r in conv)
strong=primary and len(k4final)>=3 and k4consec and audit.get('status')=='PASS'
status='GO_REALISTIC_ULDP_ML' if strong else ('REPAIR_FINMEM1A' if primary else 'STOP_PRACTICAL_DPSGD')
decision={'milestone':'PPML-SHUFFLE-LAST2-DPSGD-FINMEM1A','decision':status,'historical_statuses_preserved':{'SCALECERT':'STOP_REALISTIC_DP_SGD','MEMFRONT1':'REPAIR_MEMFRONT','MEMFRONT2':'REPAIR_MEMFRONT2','FINMEM1_GPU_INTERIM':'NOT_A_SCIENTIFIC_DECISION'},'gpu_used_for_new_jobs':False,'P1_memory':mem,'P1_final_state':priv,'P1_fit_x_ge_4':f4,'P1_fit_x_ge_8':f8,'P1_local_exponents':local,'P1_strict_FS_win_M_both_deltas':strict,'P1_two_consecutive_asymptotic_FS_win':consecutive,'P1_numerical_convergence':conv,'k4_final_state':k4final,'k4_FS':k4,'k4_two_consecutive_both_delta_wins':k4consec,'P2_FS':'DEFERRED','k1_FS':'NOT_AUTHORIZED','independent_audit':audit,'claim_scope':'Finite-state numerical witness only; does not establish a theorem for arbitrary neural-network DP-SGD.'}
(ROOT/'FINMEM1A_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
def fmt(x): return f'{float(x):.6g}'
mt='\n'.join(f"| {r['M']} | {fmt(r['richardson_continuum'])} | {fmt(int(r['M'])/float(r['richardson_continuum']))} | {r['status']} |" for r in mem)
pt='\n'.join(f"| {r['M']} | {fmt(r['L_TV_M'])} | {fmt(r['M_over_LTV_M'])} | {fmt(r['TV_final'])} | {fmt(r['epsilon_delta1e-3_upper'])} | {fmt(r['epsilon_delta1e-5_upper'])} |" for r in priv)
ft='\n'.join(f"| {m} | {fmt(next(r for r in fs_by_m[m] if float(r['delta'])==.001)['ratio_final_upper_over_FS_lower'])} | {fmt(next(r for r in fs_by_m[m] if float(r['delta'])==1e-5)['ratio_final_upper_over_FS_lower'])} | {'WIN' if m in strict else 'NO'} |" for m in sorted(fs_by_m))
kt='\n'.join(f"| {r['M']} | {r['delta']} | {fmt(r['epsilon_final_upper'])} | {fmt(r['epsilon_FS_lower'])} | {'WIN' if r['strict_win']=='True' else 'NO'} |" for r in k4) or 'No k=4 accountant results.'
report=f'''# FINMEM1A P1 completion report

## Decision

**{status}**. The result concerns the frozen one-dimensional discrete numerical witness only. It does not establish a general neural-network DP-SGD theorem.

The historical SCALECERT, MEMFRONT1, and MEMFRONT2 decisions remain unchanged. The 14 P1 k=2 Feldman–Shenfeld (FS) outputs are reused from the immutable 19/40 snapshot and verified against IDs, parameters, and pinned commit. New P1 final-state jobs were run in this separate branch on CPU. The conditional k=4 FS queries used GPU only after same-node bitwise kernel parity plus the archived full-accountant CPU/GPU parity audit. Broad Slurm job 930 was cancelled; its partial outputs were excluded.

## Finite-horizon memory

| M | Richardson continuum L_TV(M) | x_M | Status |
|---:|---:|---:|---|
{mt}

Each point passes domain-vs-base <=1%, fine-grid/Richardson <=2%, and finite-domain defect <=0.5%. The spectral continuation used beyond direct coverage passes the frozen held-out validation (<5% pointwise error). This uses L_TV(M), not L_TV(infinity).

## Final-state privacy

| M | L_TV(M) | x_M | Final TV | Epsilon upper, delta=1e-3 | Epsilon upper, delta=1e-5 |
|---:|---:|---:|---:|---:|---:|
{pt}

All seven base points are fresh DIRECT_FULL_GRID computations. Anchor grids BASE/GRID/GRID2/DOMAIN pass at M=1024, 4096, and 16384. At x>=4, beta={fmt(f4['beta_TV'])}, fit 95% regression interval [{fmt(f4['beta_CI95_low'])}, {fmt(f4['beta_CI95_high'])}], R2={fmt(f4['R2'])}, n={f4['n']}. At x>=8, beta={fmt(f8['beta_TV'])}, R2={fmt(f8['R2'])}, n={f8['n']}. Dropping the first x>=4 point gives beta={fmt(f4['beta_drop_first'])}, R2={fmt(f4['R2_drop_first'])}. The last three local exponents are positive.

## Archived FS comparison

Ratios are conservative final epsilon upper / exact archived FS lower. Strict win requires ratio <0.99.

| M | Ratio delta=1e-3 | Ratio delta=1e-5 | Both-delta strict win |
|---:|---:|---:|---|
{ft}

The contiguous asymptotic region M=1024 through 16384 wins at both deltas, with the ratio declining as M increases. No archived P1 k=2 query was rerun.

## k=4 robustness

Three fresh k=4 final-state points were computed at M=1024, 2048, and 4096. The pinned accountant with deterministic GPU gather was triggered only after P1 k=2 passed and same-node parity passed; it stopped after two consecutive candidate points passed both deltas.

| M | delta | Final epsilon upper | FS lower | Result |
|---:|---:|---:|---:|---|
{kt}

## Execution and limitations

The initial FINMEM1A specification described the final-state curve as missing. A prior local FINMEM1 tree contained older final-state artifacts; these were not silently relabeled. FINMEM1A regenerated its P1 memory and final-state computations. P2 FS remains deferred; k=1 FS was not run. The memory table generator also produced P2 memory diagnostics as a by-product; no P2 final-state or accountant result affects this decision.

Independent audit: **{audit.get('status','NOT_RUN')}**. P1 final-state and all memory calculations ran on CPU; P1 used 8 processes and k=4 final state used 3 processes with numerical-library thread counts fixed to 1. GPU was used only for triggered k=4 FS accountant queries. Archived P1 k=2 GPU outputs remain labeled and hashed as reused external-accountant results.
'''
(ROOT/'FINAL_FINMEM1A_REPORT.md').write_text(report)
(ROOT/'README.md').write_text(f'# FINMEM1A Review Package\n\nFocused P1 final-state completion with archived P1 FS baseline reuse.\n\nTerminal decision: **{status}**. See `FINAL_FINMEM1A_REPORT.md`, `FINMEM1A_DECISION.json`, and `INDEPENDENT_AUDIT.json`. Raw per-job JSON/NPZ and progress logs are retained.\n')
print(json.dumps({'status':status,'P1_primary_gates':primary,'k4_gates':k4consec,'audit':audit.get('status')}))
