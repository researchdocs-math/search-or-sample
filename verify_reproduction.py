#!/usr/bin/env python3
"""Compare deterministic scientific outputs while excluding timing measurements."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from sos.experiment import run_one
ROOT=Path(__file__).resolve().parent

def scientific(x):
 if isinstance(x,dict):return {k:scientific(v) for k,v in x.items() if k not in ['initial_seconds','elapsed_seconds','seconds']}
 if isinstance(x,list):return [scientific(v) for v in x]
 return x

def compare(a,b,path=''):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(u,v) in enumerate(zip(a,b)):compare(u,v,path+f'/{i}')
 elif isinstance(a,float):assert np.isclose(a,b,rtol=1e-10,atol=1e-12),f'{path}: {a} != {b}'
 else:assert a==b,f'{path}: {a} != {b}'

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results');p.add_argument('--limit',type=int,default=18);p.add_argument('--out',type=Path,default=ROOT/'verification.json');p.add_argument('--audit-cases',type=int,default=1);a=p.parse_args()
 selected=[]
 # Explicit ordering spans acquisition rules, both noise levels, and all prices.
 cases=[('main','poly3','adaptive',0.,.25),('main','rational','adaptive',.1,1.),('main','oscillator','sample_always',.1,4.),('main','interaction','search_only',0.,1.),('main','rational3d','uncorrected',.1,.25),('main','drag','fixed_50_corrected',0.,4.),('main','sin2','adaptive_random',.1,1.),('main','poly2d','fixed_25',0.,.25),('main','sinmix','sample_random',.1,1.),('main','sincos','sample_maximin',.1,4.),('engineering','airfoil','adaptive',0.,1.),('engineering','concrete','sample_always',0.,1.),('main','poly3','fixed_50',.1,1.),('main','rational','fixed_75',.1,1.),('main','oscillator','resample_only',.1,1.),('sensitivity_large','rational','adaptive',.1,.25),('sensitivity_large','rational','sample_always',.1,.25),('sensitivity_b5r3','rational','adaptive',.1,1.)]
 for directory,problem,policy,noise,price in cases[:a.limit]:
  matches=[]
  for path in sorted((a.results/directory).glob('*.json')):
   x=json.loads(path.read_text());s=x['spec']
   if (s['problem'],s['policy'],s['noise'],s['query_cost'])==(problem,policy,noise,price):matches.append((s['seed'],path,x))
  assert matches,f'No stored run for {problem}/{policy}'
  seed,path,x=min(matches,key=lambda z:z[0]);fresh=run_one(x['spec']);compare(scientific(x),scientific(fresh));selected.append({'directory':directory,'file':path.name,'spec':x['spec'],'verified':True});print('Verified',problem,policy,seed,flush=True)
 audited=[]
 if a.audit_cases:
  from audit_actions import audit
  paths=sorted((a.results/'audit').glob('*.json'))
  assert len(paths)>=a.audit_cases,'Insufficient audit trajectories'
  with tempfile.TemporaryDirectory(prefix='.repeatability-',dir=ROOT) as folder:
   for path in paths[:a.audit_cases]:
    saved=json.loads(path.read_text());key=audit(saved['spec'],folder);fresh=json.loads((Path(folder)/(key+'.json')).read_text());compare(saved,fresh);audited.append({'file':path.name,'spec':saved['spec'],'verified':True});print('Verified audit',saved['spec'],flush=True)
 a.out.write_text(json.dumps({'audit_trajectories':audited,'float_rtol':1e-10,'float_atol':1e-12,'verified_runs':selected},indent=2))
if __name__=='__main__':main()
