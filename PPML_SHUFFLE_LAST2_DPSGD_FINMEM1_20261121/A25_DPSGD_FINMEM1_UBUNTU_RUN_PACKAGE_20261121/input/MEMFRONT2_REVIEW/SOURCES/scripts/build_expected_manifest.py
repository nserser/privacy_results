#!/usr/bin/env python3
import argparse,json,pathlib,sys,yaml
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scalecert.plan import build_plan
from scalecert.experiments import canonical_id

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); ap.add_argument('--stages',default='')
    a=ap.parse_args(); C=yaml.safe_load(open(a.config)); stages=[x.strip() for x in a.stages.split(',') if x.strip()] or None
    jobs=build_plan(C,stages); rows=[]
    for j in jobs:
        rows.append({'experiment_id':canonical_id(j['family'],j['config']),'family':j['family'],'stage':j['config'].get('stage'),'config':j['config']})
    ids=[r['experiment_id'] for r in rows]
    if len(ids)!=len(set(ids)): raise SystemExit('duplicate expected experiment IDs')
    payload={'config':str(pathlib.Path(a.config).resolve()),'count':len(rows),'rows':rows}
    p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(len(rows))
if __name__=='__main__': main()
