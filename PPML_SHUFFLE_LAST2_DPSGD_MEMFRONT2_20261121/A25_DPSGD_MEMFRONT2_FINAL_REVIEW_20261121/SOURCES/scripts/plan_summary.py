#!/usr/bin/env python3
import argparse,collections,pathlib,sys,yaml
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scalecert.plan import build_plan
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
C=yaml.safe_load(open(a.config));jobs=build_plan(C);c=collections.Counter(j['family'] for j in jobs)
print('total',len(jobs));
for k,v in sorted(c.items()): print(k,v)
