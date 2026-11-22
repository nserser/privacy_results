#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import argparse, concurrent.futures, json, pathlib, sys, traceback
import yaml

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scalecert.plan import build_plan
from scalecert.experiments import run_job, canonical_id


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--workers",default="auto")
    ap.add_argument("--stages",default="")
    ap.add_argument("--shard-index",type=int,default=0)
    ap.add_argument("--num-shards",type=int,default=1)
    ap.add_argument("--fail-fast",action="store_true")
    a=ap.parse_args()
    C=yaml.safe_load(open(a.config))
    stages=[x.strip() for x in a.stages.split(",") if x.strip()] or None
    jobs=build_plan(C,stages)
    jobs=[j for i,j in enumerate(jobs) if i%a.num_shards==a.shard_index]
    out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
    workers=int(os.environ.get("SLURM_CPUS_PER_TASK",os.cpu_count() or 1)) if a.workers=="auto" else int(a.workers)
    print(f"jobs={len(jobs)} workers={workers} shard={a.shard_index}/{a.num_shards} stages={stages or 'default'}")
    failures=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        fs={ex.submit(run_job,j):j for j in jobs}
        for f in concurrent.futures.as_completed(fs):
            j=fs[f]; rid=canonical_id(j["family"],j["config"])
            try:
                r=f.result()
            except Exception as e:
                failures+=1
                r={"experiment_id":rid,"family":j["family"],"config":j["config"],"status":"FAILED","error":repr(e),"traceback":traceback.format_exc()}
            (out/f"{j['family']}_{rid}.json").write_text(json.dumps(r,indent=2,sort_keys=True))
            print(j["family"],rid,r["status"])
            if a.fail_fast and r["status"]=="FAILED":
                raise SystemExit(2)
    if failures:
        print(f"WARNING failures={failures}")
        raise SystemExit(2)

if __name__=="__main__": main()
