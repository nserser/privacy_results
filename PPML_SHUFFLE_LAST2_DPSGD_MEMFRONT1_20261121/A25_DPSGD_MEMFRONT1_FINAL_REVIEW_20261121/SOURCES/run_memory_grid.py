from __future__ import annotations
import argparse, concurrent.futures, json, os
from pathlib import Path
from memfront.memory import all_memory_configs, memory_response, append_jsonl, canonical_id

def _one(args):
    cfg, out, prog = args
    # Do not let BLAS spawn nested workers; orchestration owns CPU parallelism.
    os.environ.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    return memory_response(cfg, Path(out), Path(prog))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=1); ap.add_argument("--root",default="memfront/run"); ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args(); root=Path(a.root); out=root/"memory_jobs"; out.mkdir(parents=True,exist_ok=True); prog=root/"MEMFRONT_PROGRESS.jsonl"
    configs=all_memory_configs()
    jobs=[(c,str(out),str(prog)) for c in configs]
    manifest={"stage":"M3_DISCRETE_MEMORY","workers":a.workers,"job_ids":[canonical_id(c) for c in configs],"jobs":configs}
    (root/"MEMFRONT1_MEMORY_JOB_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    append_jsonl(prog,{"event":"memory_grid_start","jobs":len(jobs),"workers":a.workers})
    if a.limit: jobs=jobs[:a.limit]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs=[pool.submit(_one,j) for j in jobs]
        for f in concurrent.futures.as_completed(futs):
            r=f.result(); results.append({"id":canonical_id(r["config"]),"line":r["config"]["line"],"grid":r["config"]["grid_label"],"status":r["status"],"L_TV":r["L_TV"],"R":r["R"]})
            append_jsonl(prog,{"event":"memory_grid_progress","completed":len(results),"total":len(jobs),"result":results[-1]})
    results.sort(key=lambda x:(x["line"],x["grid"]))
    (root/"MEMORY_GRID_SUMMARY.json").write_text(json.dumps({"results":results,"complete":len(results)==len(jobs)},indent=2,sort_keys=True)+"\n")
    append_jsonl(prog,{"event":"memory_grid_complete","completed":len(results),"total":len(jobs)})

if __name__=="__main__": main()
