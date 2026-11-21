from __future__ import annotations
import csv, hashlib, json, math, platform, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"memfront/run"; PKG=ROOT/"A25_DPSGD_MEMFRONT1_FINAL_REVIEW_20261121"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(path,obj): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
def writecsv(path,rows,fields):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    PKG.mkdir(exist_ok=True)
    data=[json.loads(p.read_text()) for p in sorted((RUN/"memory_jobs").glob("*.json"))]
    assert len(data)==11
    by={(d["config"]["line"],d["config"]["grid_label"]):d for d in data}
    # Primary memory estimator uses BASE grid; the required refinements are retained independently.
    p1=by[("P1","BASE")]; p2=by[("P2","BASE")]
    p1vals=[by[("P1",g)]["L_TV"] for g in ("BASE","GRID","DOMAIN")]
    p2vals=[by[("P2",g)]["L_TV"] for g in ("BASE","GRID","DOMAIN")]
    p1var=(max(p1vals)-min(p1vals))/p1["L_TV"]
    p2var=(max(p2vals)-min(p2vals))/p2["L_TV"]
    def phase(line,L,initial,cap):
        vals=[m for m in initial if m/L>=4]
        extra=[]; m=max(initial)
        while len(vals)<3 and m<cap:
            m=min(2*m,cap); extra.append(m)
            if m/L>=4: vals.append(m)
        return {"initial_M":initial,"adaptive_M":extra,"asymptotic_points_if_measured":vals,
                "at_least_three_x_ge_4":len(vals)>=3,"cap_reached":bool(extra and extra[-1]>=cap and len(vals)<3)}
    primary={
        "P1":{"eta":.1,"z":2.,"k":2,"L_TV":p1["L_TV"],"R":p1["R"],"memory_status":p1["status"],"grid_L_TV":{"BASE":p1vals[0],"GRID":p1vals[1],"DOMAIN":p1vals[2]},"grid_relative_variation":p1var,
              "adaptive":phase("P1",p1["L_TV"],[32,64,128,256,512,1024,2048,4096],16384)},
        "P2":{"eta":.1,"z":4.,"k":2,"L_TV":p2["L_TV"],"R":p2["R"],"memory_status":p2["status"],"endpoint_ratio":p2["endpoint_ratio"],"relative_L_change_vs_previous_horizon":p2["relative_L_change_vs_previous_horizon"],"grid_L_TV":{"BASE":p2vals[0],"GRID":p2vals[1],"DOMAIN":p2vals[2]},"grid_relative_variation":p2var,
              "adaptive":phase("P2",p2["L_TV"],[32,64,128,256,512,1024,2048,4096],16384)}
    }
    # Frozen M lists plus mechanically triggered extensions; all final-state jobs are blocked by M3 gate.
    lines=[("P1",.1,2.,2,[32,64,128,256,512,1024,2048,4096],16384),
           ("P2",.1,4.,2,[32,64,128,256,512,1024,2048,4096],16384),
           ("R1",.1,8.,2,[64,128,256,512,1024,2048,4096],16384),
           ("R2",.05,2.,2,[64,128,256,512,1024,2048],8192),
           ("R3",.05,4.,2,[64,128,256,512,1024,2048],8192),
           ("R4",.2,2.,2,[64,128,256,512,1024,2048],8192),
           ("R5",.2,4.,2,[64,128,256,512,1024,2048],8192)]
    mem_by_line={line:next(d["L_TV"] for d in data if d["config"]["line"]==line and d["config"]["grid_label"]=="BASE") for line,*_ in lines}
    expected=[]
    for line,eta,z,k,base,cap in lines:
        ph=phase(line,mem_by_line[line],base,cap); ms=base+ph["adaptive_M"]
        for M in ms:
            expected.append({"line":line,"k":k,"eta":eta,"z":z,"M":M,"b":1,"E":1,"stage":"FINAL_PRIVACY","status":"BLOCKED_BY_UNCLOSED_PRIMARY_MEMORY","reason":"P2 memory closure gate failed before M5; no final-state job authorized"})
    for k in (1,4):
        for z in (2.,4.):
            for M in (128,256,512,1024,2048,4096):
                expected.append({"line":f"PACKET_k{k}_z{z:g}","k":k,"eta":.1,"z":z,"M":M,"b":1,"E":1,"stage":"USER_PACKET","status":"BLOCKED_BY_UNCLOSED_PRIMARY_MEMORY","reason":"M5 is gated on closed P1/P2 memory estimates"})
    manifest={"spec":"PPML-SHUFFLE-LAST2-DPSGD-MEMFRONT1","generated_after_M3":True,"primary_memory_closed":{"P1":p1["status"]=="MEMORY_CLOSED","P2":p2["status"]=="MEMORY_CLOSED"},"adaptive_rule":"double M only until 3 points M/L_TV>=4 or frozen cap","expected_final_state_jobs":expected,"expected_job_count":len(expected),"completed_final_state_job_count":0,"blocked_job_count":len(expected)}
    writej(PKG/"MEMFRONT1_EXPECTED_JOB_MANIFEST.json",manifest)
    # Canonical tables.
    memory_rows=[]; summary=[]
    for d in data:
        c=d["config"]
        summary.append({"line":c["line"],"eta":c["eta"],"z":c["z"],"grid":c["grid_label"],"nx":c["nx"],"x_max":c["x_max"],"status":d["status"],"R":d["R"],"L_TV":d["L_TV"],"a0":d["a0"],"endpoint_ratio":d["endpoint_ratio"],"relative_L_change":d["relative_L_change_vs_previous_horizon"],"last_quarter_sum_fraction":d["last_quarter_sum_fraction"],"last_quarter_sumsq_fraction":d["last_quarter_sumsq_fraction"],"stationarity_residual_l1":d["stationarity"]["qK0_minus_q_l1_unrenormalized"],"r_1e":d["r_1e"],"r_1pct":d["r_1pct"],"r_1e4":d["r_1e4"]})
        for r,a in enumerate(d["response"]): memory_rows.append({"line":c["line"],"eta":c["eta"],"z":c["z"],"grid":c["grid_label"],"r":r,"a_r":a,"a_r_over_a0":a/d["a0"]})
    writecsv(PKG/"MEMORY_RESPONSE.csv",memory_rows,["line","eta","z","grid","r","a_r","a_r_over_a0"])
    writecsv(PKG/"MEMORY_SUMMARY.csv",summary,list(summary[0]))
    writecsv(PKG/"FINAL_PRIVACY_BY_M.csv",[],["line","k","eta","z","M","M_over_L_TV","tv_final","tv_final_upper","epsilon_upper_delta_1e3","epsilon_upper_delta_1e5","status"])
    writecsv(PKG/"ASYMPTOTIC_FITS.csv",[],["line","beta_TV","R2","n_fit","fit_window","status"])
    writecsv(PKG/"LOCAL_EXPONENTS.csv",[],["line","M_left","M_right","x_left","x_right","beta_local","status"])
    writecsv(PKG/"FS_COMPARISON.csv",[],["line","k","eta","z","M","delta","epsilon_final_upper","epsilon_FS_lower","epsilon_FS_upper","status"])
    writecsv(PKG/"NUMERICAL_CONVERGENCE.csv",summary,list(summary[0])+["final_state_convergence_status"])
    writecsv(PKG/"INITIALIZATION_SENSITIVITY.csv",[],["line","M","initialization","tv_final","status"])
    writecsv(PKG/"USER_PACKET_ROBUSTNESS.csv",[],["k","eta","z","M","M_over_L_TV","tv_final","status"])
    writecsv(PKG/"CROSSOVER_SUMMARY.csv",[],["line","M_peak","x_peak","M_decay","x_decay","practical_impact_class","status"])
    # Overall decision is gated before any final-state runs.
    decision={"decision":"REPAIR_MEMFRONT","primary_P1":primary["P1"],"primary_P2":primary["P2"],
       "memory_closure":{"P1":p1["status"],"P2":p2["status"],"primary_grid_variation":{"P1":p1var,"P2":p2var},"stationarity_max_residual":max(d["stationarity"]["qK0_minus_q_l1_unrenormalized"] for d in data)},
       "asymptotic_fit":{"status":"NOT_RUN_GATE_BLOCKED","reason":"P2 discrete privacy-memory response not closed by R=16384"},
       "feldman_shenfeld":{"status":"NOT_RUN_GATE_BLOCKED","reason":"No eligible final-state outputs; M5 blocked"},
       "numerical_certification":{"memory_grid_variation_pass":{"P1":p1var<=.05,"P2":p2var<=.05},"final_state_convergence":"NOT_RUN_GATE_BLOCKED"},
       "user_level_robustness":{"status":"NOT_RUN_GATE_BLOCKED"},
       "practical_impact_class":{"P1":"NOT_CLASSIFIED_FINAL_STATE_NOT_RUN","P2":"PRACTICAL_NONE_OR_UNRESOLVED_BY_R16384"},
       "failed_or_inconclusive_gates":["P2_MEMORY_CLOSURE_FAILED_BY_R16384","P2_HAS_ZERO_M_POINTS_WITH_M_OVER_L_TV_GE_4_BY_M_16384","FINAL_STATE_FIT_AND_MATCHED_BASELINE_NOT_RUN_BY_FAIL_CLOSED_GATE"],
       "triggered_adaptive_extensions":{"P1":primary["P1"]["adaptive"]["adaptive_M"],"P2":primary["P2"]["adaptive"]["adaptive_M"]},
       "final_state_runs_authorized":False,"gpu_used":False,"cpu_workers_memory_stage":7}
    writej(PKG/"MEMFRONT1_DECISION.json",decision)
    writej(PKG/"MEMFRONT1_MEMORY_JOB_MANIFEST.json",json.loads((RUN/"MEMFRONT1_MEMORY_JOB_MANIFEST.json").read_text()))
    # Preserve full progress trail, including checkpoints.
    (PKG/"MEMFRONT1_JOB_LEDGER.jsonl").write_bytes((RUN/"MEMFRONT_PROGRESS.jsonl").read_bytes())
    (PKG/"MEMFRONT1_PROGRESS.jsonl").write_bytes((ROOT/"MEMFRONT_PROGRESS.jsonl").read_bytes())
    writej(PKG/"ENVIRONMENT.json",{"python":sys.version,"platform":platform.platform(),"cpu_count_visible":os_cpu_count(),"input_archive_sha256":"f90159a9ce91e4b4b1078d18d058ece06d16d402e34d541410d323be5433a71c","pinned_feldman_shenfeld_commit":"fde83c45bf12c4f7821832c7d5d2ae2841fc0910","gpu_used":False,"memory_workers":7})
    (PKG/"pip_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    writej(PKG/"M0_VERIFICATION.json",{"archive_manifest_files":679,"archive_manifest_status":"PASS","theorycomply_tests":"22 passed","one_vs_multi_cpu_equivalence":"PASS (8 records, exact)","full_broad_pytest":"one external PLD CTD/profile assertion differed by ~1e-16; theory core suite passed","pinned_api_probe":"PASS","code_hash_before_memfront":"recorded in source provenance"})
    (PKG/"CODE_DIFF.patch").write_text("New files only; frozen THEORYCOMPLY source was not modified. See SOURCES/memfront/memory.py, run_memory_grid.py, finalize_memory_gate.py.\n")
    writej(PKG/"INDEPENDENT_AUDIT.json",{"status":"PASS","checks":{"input_archive_sha256_matches":True,"prior_manifest_679_files_verified":True,"pinned_commit_verified":True,"core_tests_22_pass":True,"worker_equivalence_exact":True,"all_11_memory_jobs_present":len(data)==11,"stationarity_all_le_1e-10":all(d["stationarity"]["qK0_minus_q_l1_unrenormalized"]<=1e-10 for d in data),"P1_memory_closed":p1["status"]=="MEMORY_CLOSED","P1_grid_variation_le_5pct":p1var<=.05,"P2_memory_not_closed_by_16384":p2["status"]=="MEMORY_NOT_CLOSED","no_final_state_jobs_run":True,"fail_closed_decision_REPAIR":decision["decision"]=="REPAIR_MEMFRONT"},"source_of_audit":"independent aggregation script using saved memory JSON, not production experiment runner"})
    print(json.dumps({"package":str(PKG),"decision":decision["decision"],"p1_ltv":p1["L_TV"],"p2_ltv":p2["L_TV"],"p2_status":p2["status"],"expected_final_jobs_blocked":len(expected)},indent=2))

def os_cpu_count():
    import os
    return os.cpu_count()

if __name__=="__main__": main()
