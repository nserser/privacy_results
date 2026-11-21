#!/usr/bin/env python3
import argparse,json,math,pathlib

def load(d):
 out={}
 for p in pathlib.Path(d).glob('*.json'):
  r=json.loads(p.read_text()); out[(r.get('family'),r.get('experiment_id'))]=r
 return out

def strip_runtime(x):
 if isinstance(x,dict): return {k:strip_runtime(v) for k,v in x.items() if k not in {'start_time','end_time','elapsed_s','pid','host'}}
 if isinstance(x,list): return [strip_runtime(v) for v in x]
 return x

def compare(a,b,path=''):
 if type(a)!=type(b): return [(path,a,b)]
 if isinstance(a,dict):
  if set(a)!=set(b): return [(path+'/keys',set(a),set(b))]
  out=[]
  for k in a: out+=compare(a[k],b[k],path+'/'+str(k))
  return out
 if isinstance(a,list):
  if len(a)!=len(b): return [(path+'/len',len(a),len(b))]
  out=[]
  for i,(x,y) in enumerate(zip(a,b)): out+=compare(x,y,path+f'/{i}')
  return out
 if isinstance(a,float):
  if math.isnan(a) and math.isnan(b): return []
  return [] if a==b else [(path,a,b)]
 return [] if a==b else [(path,a,b)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('a');ap.add_argument('b');a=ap.parse_args()
 A,B=load(a.a),load(a.b)
 if set(A)!=set(B): raise SystemExit(f'key mismatch {set(A)^set(B)}')
 diffs=[]
 for k in A: diffs += [(k,)+d for d in compare(strip_runtime(A[k]),strip_runtime(B[k]))]
 if diffs:
  print('DIFF',diffs[:20]); raise SystemExit(2)
 print(f'PASS exact scientific equivalence: {len(A)} records')
if __name__=='__main__':main()
