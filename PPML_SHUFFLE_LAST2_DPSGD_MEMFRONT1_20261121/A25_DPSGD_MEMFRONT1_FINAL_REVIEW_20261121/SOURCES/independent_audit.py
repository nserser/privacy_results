"""Separate post-run gate audit; intentionally imports no production runner."""
import csv, glob, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
pkg=root/"A25_DPSGD_MEMFRONT1_FINAL_REVIEW_20261121"
def j(p): return json.loads(Path(p).read_text())
def main():
    files=glob.glob(str(root/"memfront/run/memory_jobs/*.json")); ds=[j(p) for p in files]
    assert len(ds)==11
    by={(d["config"]["line"],d["config"]["grid_label"]):d for d in ds}
    p1,p2=by[("P1","BASE")],by[("P2","BASE")]
    vals1=[by[("P1",g)]["L_TV"] for g in ("BASE","GRID","DOMAIN")]
    vals2=[by[("P2",g)]["L_TV"] for g in ("BASE","GRID","DOMAIN")]
    var1=(max(vals1)-min(vals1))/p1["L_TV"]; var2=(max(vals2)-min(vals2))/p2["L_TV"]
    checks={
      "all_11_memory_files":len(ds)==11,
      "all_stationarity_residuals_le_1e-10":all(d["stationarity"]["qK0_minus_q_l1_unrenormalized"]<=1e-10 for d in ds),
      "P1_closure_all_conditions":p1["status"]=="MEMORY_CLOSED" and p1["R"]<=16384,
      "P1_grid_variation_le_5pct":var1<=.05,
      "P2_unclosed_by_frozen_cap":p2["status"]=="MEMORY_NOT_CLOSED" and p2["R"]==16384,
      "P2_tail_endpoint_exceeds_1e-4":p2["endpoint_ratio"]>1e-4,
      "P2_L_change_exceeds_1pct":p2["relative_L_change_vs_previous_horizon"]>.01,
      "P2_grid_variation_le_5pct":var2<=.05,
      "P2_adaptive_M_cap_insufficient":16384/p2["L_TV"]<4,
      "all_final_jobs_blocked":all(x["status"]=="BLOCKED_BY_UNCLOSED_PRIMARY_MEMORY" for x in j(pkg/"MEMFRONT1_EXPECTED_JOB_MANIFEST.json")["expected_final_state_jobs"]),
      "no_final_privacy_rows":sum(1 for _ in csv.DictReader((pkg/"FINAL_PRIVACY_BY_M.csv").open()))==0,
      "decision_is_repair":j(pkg/"MEMFRONT1_DECISION.json")["decision"]=="REPAIR_MEMFRONT",
    }
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"P1_LTV_grid":vals1,"P1_variation":var1,"P2_LTV_grid":vals2,"P2_variation":var2,"auditor":"memfront.independent_audit (separate code path)"}
    (pkg/"INDEPENDENT_AUDIT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))
    if not all(checks.values()): raise SystemExit(1)
if __name__=="__main__": main()
